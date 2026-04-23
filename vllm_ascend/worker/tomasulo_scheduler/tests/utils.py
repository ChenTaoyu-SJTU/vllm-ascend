import math
import os
import traceback
from multiprocessing.queues import Queue

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import KVCacheConfig, get_kv_cache_configs
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.logger import logger
import multiprocessing as mp
import socket

from typing import Callable, TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch
    from vllm_ascend.worker.worker import NPUWorker


class SimpleBlockAllocator:
    def __init__(self, num_gpu_blocks: int, block_size: int):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.next_block_id = 0

    def allocate(self, num_new_blocks: int) -> list[int]:
        block_ids = list(range(self.next_block_id, self.next_block_id + num_new_blocks))
        self.next_block_id += num_new_blocks
        assert self.next_block_id <= self.num_gpu_blocks, (
            f"Allocated {self.next_block_id} blocks but only {self.num_gpu_blocks} available"
        )
        return block_ids


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def build_engine_args(cli_args_dict: dict) -> EngineArgs:
    return EngineArgs(
        model=cli_args_dict["model"],
        tensor_parallel_size=cli_args_dict["tensor_parallel_size"],
        data_parallel_size=cli_args_dict["data_parallel_size"],
        enable_expert_parallel=cli_args_dict["enable_expert_parallel"],
        max_num_seqs=cli_args_dict["max_num_seqs"],
        max_model_len=cli_args_dict["max_model_len"],
        compilation_config=cli_args_dict.get("compilation_config", {}),
        enforce_eager=cli_args_dict.get("enforce_eager", True),
        load_format=cli_args_dict.get("load_format", "dummy"),
    )


def init_worker_env_and_config(
    local_rank: int,
    tp_size: int,
    dp_rank: int,
    vllm_config: VllmConfig,
) -> VllmConfig:
    """Set up env vars and adjust VllmConfig per-rank.

    Matches native vllm-ascend behavior where each DP replica runs in its
    own engine process with ASCEND_RT_VISIBLE_DEVICES narrowed to the
    tp_size cards that belong to that replica. Inside the replica,
    local_rank indexes into those visible cards (0..tp_size-1).

    Receives a VllmConfig created in the parent process (shared via pickle),
    so all workers have the same _data_parallel_master_port_list and avoid
    port conflicts during init_distributed_environment().

    For Dense models, each DP replica is an independent engine with
    world_size=tp_size and data_parallel_size reset to 1.

    For MoE models, world_size=tp_size (single DP replica view) and
    data_parallel_rank is preserved so that init_distributed_environment()
    can expand to the global rank via dp_rank * world_size + local_rank.

    Must be called at the start of every spawned worker process,
    before any NPU call and before set_current_vllm_config /
    create_and_init_worker.
    """
    # Narrow visible NPUs to this DP replica's slice before any NPU op,
    # so that torch.device(f"npu:{local_rank}") maps to a unique physical
    # card. This mirrors native vllm-ascend's DPEngineCoreProc behavior
    # (vllm/v1/engine/core.py: _set_cuda_visible_devices).
    device_ids = range(dp_rank * tp_size, (dp_rank + 1) * tp_size)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)

    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(tp_size)
    os.environ["VLLM_USE_V1"] = "1"

    is_moe = vllm_config.parallel_config.enable_expert_parallel
    if is_moe:
        vllm_config.parallel_config.data_parallel_rank = dp_rank
    else:
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

    return vllm_config


def create_and_init_worker(
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
) -> tuple["NPUWorker", KVCacheConfig]:
    """Run the full NPUWorker init sequence (Phase 1-5).

    Caller must wrap this in ``set_current_vllm_config(vllm_config)``.

    Returns (worker, kv_cache_config) ready for execute_model or
    direct model_runner access.
    """
    from vllm_ascend.worker.worker import NPUWorker

    worker = NPUWorker(
        vllm_config=vllm_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    worker.init_device()
    worker.load_model()

    available_memory = worker.determine_available_memory()
    num_gpu_blocks = 256
    worker.initialize_cache(num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0)

    kv_cache_spec = worker.get_kv_cache_spec()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )
    kv_cache_config = kv_cache_configs[0]
    worker.initialize_from_config(kv_cache_config)

    return worker, kv_cache_config


def build_scheduler_output(
    prompt_token_ids: list[int],
    num_gpu_blocks: int,
    block_size: int,
    req_id: str = "req-0",
) -> SchedulerOutput:
    """Build a minimal SchedulerOutput for a single new prefill request."""
    num_blocks_needed = math.ceil(len(prompt_token_ids) / block_size)
    assert num_blocks_needed <= num_gpu_blocks, (
        f"Need {num_blocks_needed} blocks but only {num_gpu_blocks} available"
    )
    block_ids = list(range(num_blocks_needed))

    new_req = NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_ids=(block_ids,),
        num_computed_tokens=0,
        lora_request=None,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: len(prompt_token_ids)},
        total_num_scheduled_tokens=len(prompt_token_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def worker_init_and_execute(
    local_rank: int,
    tp_size: int,
    dp_rank: int,
    distributed_init_method: str,
    vllm_config: VllmConfig,
    error_queue: Queue,
):
    """
    Target function for each spawned worker process.
    Runs the full NPUWorker initialization sequence and then
    execute_model + sample_tokens on a dummy prefill request.
    """
    try:
        vllm_config = init_worker_env_and_config(
            local_rank, tp_size, dp_rank, vllm_config,
        )

        with set_current_vllm_config(vllm_config):
            worker, kv_cache_config = create_and_init_worker(
                vllm_config, local_rank, local_rank,
                distributed_init_method,
            )

            block_size = (
                kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            )

            # Do execute_model + sample_tokens test
            prompt_token_ids = list(range(1, 33))  # 32 dummy tokens
            scheduler_output = build_scheduler_output(
                prompt_token_ids=prompt_token_ids,
                num_gpu_blocks=kv_cache_config.num_blocks,
                block_size=block_size,
            )

            exec_output = worker.execute_model(scheduler_output)
            sample_output = worker.sample_tokens(grammar_output=None)

            assert exec_output is None, \
                f"DP{dp_rank}-Rank{local_rank}: execute_model should return None"
            assert sample_output is not None, "sample_tokens should return non-None"
            logger.info(
                f"DP{dp_rank}-Rank{local_rank}: sample_tokens output: "
                f"{sample_output.get_output().sampled_token_ids}"
            )

    except Exception as e:
        error_queue.put((
            (dp_rank, local_rank),
            f"{type(e).__name__}: {e}",
            traceback.format_exc(),
        ))

def register_forward_test_requests(
    model_runner,
    num_scheduled_tokens: dict[str, int],
    allocator: SimpleBlockAllocator,
    block_size: int,
) -> tuple[list[str], dict[str, CachedRequestState]]:
    req_ids = []
    req_states = {}
    for req_id, num_prompt_tokens in num_scheduled_tokens.items():
        prompt_token_ids = list(range(1, num_prompt_tokens + 1))
        num_blocks = math.ceil(num_prompt_tokens / block_size)
        block_ids = (allocator.allocate(num_blocks),)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=SamplingParams(),
            generator=None,
            block_ids=block_ids,
            num_computed_tokens=0,
            output_token_ids=[0],
        )
        model_runner.input_batch.add_request(req_state)
        model_runner.requests[req_id] = req_state
        req_ids.append(req_id)
        req_states[req_id] = req_state
    return req_ids, req_states


def prepare_forward_step(
    model_runner: "NPUModelRunner",
    num_scheduled_tokens: dict[str, int],
):
    # Example: num_scheduled_tokens = {"0": 3, "1": 2, "2": 5}
    # Assume All requests start from num_computed_tokens = 0 in this example (pure prefill).
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    # num_reqs = 3
    num_reqs = model_runner.input_batch.num_reqs
    req_ids = list(model_runner.input_batch.req_ids)

    # num_scheduled_tokens_np = [3, 2, 5]
    num_scheduled_tokens_np = np.array(
        [num_scheduled_tokens[rid] for rid in req_ids], dtype=np.int32
    )

    # total_num_tokens = 3 + 2 + 5 = 10
    total_num_tokens = int(num_scheduled_tokens_np.sum())

    # max_query_len = max([3, 2, 5]) = 5
    max_query_len = int(num_scheduled_tokens_np.max())

    # req_indices repeats each request index according to its token count.
    # [3, 2, 5] -> [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    req_indices = np.repeat(np.arange(num_reqs), num_scheduled_tokens_np)

    # cu_num_tokens is the cumulative sum: [3, 5, 10]
    # arange is the intra-request position offset: [0, 1, 2, 0, 1, 0, 1, 2, 3, 4]
    cu_num_tokens, arange = model_runner._get_cumsum_and_arange(
        num_scheduled_tokens_np
    )

    # positions = num_computed_tokens[req_indices] + arange
    #           = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 2, 0, 1, 0, 1, 2, 3, 4]
    #           = [0, 1, 2, 0, 1, 0, 1, 2, 3, 4]
    positions_np = model_runner.positions.np[:total_num_tokens]
    np.add(
        model_runner.input_batch.num_computed_tokens_cpu[req_indices],
        arange,
        out=positions_np,
    )

    # token_indices = positions + req_indices * max_model_len
    # Gather input token IDs from token_ids_cpu.
    # input_ids = [tok_0_0, tok_0_1, tok_0_2,
    #              tok_1_0, tok_1_1,
    #              tok_2_0, tok_2_1, tok_2_2, tok_2_3, tok_2_4]
    token_indices = (
        positions_np
        + req_indices * model_runner.input_batch.token_ids_cpu.shape[1]
    )
    torch.index_select(
        model_runner.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        torch.from_numpy(token_indices),
        out=model_runner.input_ids.cpu[:total_num_tokens],
    )

    # query_start_loc marks where each request starts in the flat batch.
    # [0, 3, 5, 10]  (length = num_reqs + 1)
    model_runner.query_start_loc.np[0] = 0
    model_runner.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    model_runner.query_start_loc.copy_to_gpu()
    model_runner.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

    # seq_lens records the total sequence length after this step.
    # [3, 2, 5] = num_computed_tokens + num_scheduled_tokens_np
    model_runner.seq_lens.np[:num_reqs] = (
        model_runner.input_batch.num_computed_tokens_cpu[:num_reqs]
        + num_scheduled_tokens_np
    )
    model_runner.seq_lens.np[num_reqs:] = 0
    model_runner.seq_lens.copy_to_gpu()

    # max_query_len == 5 > 1 -> ChunkedPrefill (prefill step)
    # max_query_len == 1     -> DecodeOnly (decode step)
    if max_query_len == 1:
        model_runner.attn_state = AscendAttentionState.DecodeOnly
    else:
        model_runner.attn_state = AscendAttentionState.ChunkedPrefill

    # Commit block table to GPU and compute KV-cache slot mapping.
    model_runner.input_batch.block_table.commit_block_table(num_reqs)
    model_runner.input_batch.block_table.compute_slot_mapping(
        req_indices, positions_np
    )
    model_runner.input_batch.block_table.commit_slot_mapping(total_num_tokens)

    # Copy input_ids and positions from CPU to GPU.
    model_runner.input_ids.copy_to_gpu(total_num_tokens)
    model_runner.positions.copy_to_gpu(total_num_tokens)

    # Decide batch execution mode and padding.
    # force_eager=True disables CUDA-graph padding.
    cudagraph_mode, batch_desc, _, num_tokens_across_dp, _ = (
        model_runner._determine_batch_execution_and_padding(
            num_tokens=total_num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            force_eager=True,
        )
    )
    num_tokens_padded = batch_desc.num_tokens

    # Build per-layer attention metadata (masks, block tables, etc.).
    attn_metadata, _ = model_runner._build_attention_metadata(
        num_tokens=total_num_tokens,
        num_tokens_padded=num_tokens_padded,
        num_reqs=num_reqs,
        max_query_len=max_query_len,
    )

    return (
        attn_metadata,
        total_num_tokens,
        num_tokens_padded,
        cudagraph_mode,
        batch_desc,
        num_tokens_across_dp,
    )


def run_forward_step(
    model_runner,
    attn_metadata,
    total_num_tokens: int,
    num_tokens_padded: int,
    cudagraph_mode,
    batch_desc,
    num_tokens_across_dp,
) -> torch.Tensor:
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.rotary_embedding import update_cos_sin

    positions_gpu = model_runner.positions.gpu[:num_tokens_padded]
    update_cos_sin(positions_gpu)

    input_ids = model_runner.input_ids.gpu[:num_tokens_padded]
    positions = positions_gpu

    with set_ascend_forward_context(
        attn_metadata,
        model_runner.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        aclgraph_runtime_mode=cudagraph_mode,
        batch_descriptor=batch_desc,
        num_actual_tokens=total_num_tokens,
        model_instance=model_runner.model,
        max_tokens_across_pcp=0,
        skip_compiled=False,
    ):
        hidden_states = model_runner._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
    return hidden_states


def update_state_after_model_forward(
    model_runner,
    num_scheduled_tokens: dict[str, int],
    allocator: SimpleBlockAllocator,
    block_size: int,
):
    req_ids = list(model_runner.input_batch.req_ids)
    for i, req_id in enumerate(req_ids):
        num_new_tokens = num_scheduled_tokens[req_id]
        old_computed = model_runner.input_batch.num_computed_tokens_cpu[i]
        new_computed = old_computed + num_new_tokens
        model_runner.input_batch.num_computed_tokens_cpu[i] = new_computed
        model_runner.requests[req_id].num_computed_tokens = new_computed

        dummy_output_token = 1
        model_runner.input_batch.token_ids_cpu[i, new_computed] = (
            dummy_output_token
        )
        model_runner.requests[req_id].output_token_ids.append(
            dummy_output_token
        )

        next_seq_len = new_computed + 1
        current_blocks = math.ceil(new_computed / block_size) if new_computed > 0 else 0
        needed_blocks = math.ceil(next_seq_len / block_size)
        if needed_blocks > current_blocks:
            new_block_ids = allocator.allocate(needed_blocks - current_blocks)
            model_runner.input_batch.block_table.append_row(
                (new_block_ids,), i
            )


def model_runner_forward(
    local_rank: int,
    tp_size: int,
    dp_rank: int,
    distributed_init_method: str,
    vllm_config: VllmConfig,
    error_queue: Queue,
):
    try:
        vllm_config = init_worker_env_and_config(
            local_rank, tp_size, dp_rank, vllm_config,
        )

        with set_current_vllm_config(vllm_config):
            worker, kv_cache_config = create_and_init_worker(
                vllm_config, local_rank, local_rank,
                distributed_init_method,
            )

            model_runner = worker.model_runner
            block_size = (
                kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            )
            allocator = SimpleBlockAllocator(
                kv_cache_config.num_blocks, block_size,
            )

            prefill_tokens = {"req_0": 4, "req_1": 3}
            register_forward_test_requests(
                model_runner, prefill_tokens, allocator, block_size,
            )

            (
                attn_metadata,
                total_num_tokens,
                num_tokens_padded,
                cudagraph_mode,
                batch_desc,
                num_tokens_across_dp,
            ) = prepare_forward_step(model_runner, prefill_tokens)

            hidden_states = run_forward_step(
                model_runner,
                attn_metadata,
                total_num_tokens,
                num_tokens_padded,
                cudagraph_mode,
                batch_desc,
                num_tokens_across_dp,
            )
            assert isinstance(hidden_states, torch.Tensor)
            assert hidden_states.shape[0] >= total_num_tokens
            assert not torch.isnan(hidden_states).any()
            assert not torch.isinf(hidden_states).any()
            logger.info(
                f"DP{dp_rank}-Rank{local_rank}: prefill passed, "
                f"hidden_states.shape={hidden_states.shape}"
            )

            update_state_after_model_forward(
                model_runner, prefill_tokens, allocator, block_size,
            )

            decode_tokens = {"req_0": 1, "req_1": 1}
            (
                attn_metadata,
                total_num_tokens,
                num_tokens_padded,
                cudagraph_mode,
                batch_desc,
                num_tokens_across_dp,
            ) = prepare_forward_step(model_runner, decode_tokens)

            hidden_states = run_forward_step(
                model_runner,
                attn_metadata,
                total_num_tokens,
                num_tokens_padded,
                cudagraph_mode,
                batch_desc,
                num_tokens_across_dp,
            )
            assert isinstance(hidden_states, torch.Tensor)
            assert hidden_states.shape[0] >= total_num_tokens
            assert not torch.isnan(hidden_states).any()
            assert not torch.isinf(hidden_states).any()
            logger.info(
                f"DP{dp_rank}-Rank{local_rank}: decode passed, "
                f"hidden_states.shape={hidden_states.shape}"
            )

    except Exception as e:
        error_queue.put((
            (dp_rank, local_rank),
            f"{type(e).__name__}: {e}",
            traceback.format_exc(),
        ))


def build_tomasulo_input_batch(
    model_runner,
    attn_metadata,
    total_num_tokens: int,
    num_tokens_padded: int,
    cudagraph_mode,
    batch_desc,
    num_tokens_across_dp,
):
    """Pack outputs of ``prepare_forward_step`` into a ``TomasuloInputBatch``.

    No new compute; just slices the GPU buffers already filled by
    ``prepare_forward_step`` and forwards the metadata tuple.
    """
    from vllm_ascend.worker.tomasulo_scheduler.input_batch import (
        TomasuloInputBatch,
    )

    return TomasuloInputBatch(
        input_ids=model_runner.input_ids.gpu[:num_tokens_padded],
        positions=model_runner.positions.gpu[:num_tokens_padded],
        attn_metadata=attn_metadata,
        num_tokens_padded=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_mode=cudagraph_mode,
        batch_desc=batch_desc,
        num_actual_tokens=total_num_tokens,
    )


def build_logits_indices(model_runner, num_reqs: int) -> torch.Tensor:
    """Last token offset of each sequence in the concatenated batch.

    Mirrors the default branch in ``NPUModelRunner._prepare_inputs``
    (``model_runner_v1.py:828``):
        ``logits_indices = query_start_loc.gpu[1 : num_reqs + 1] - 1``
    """
    return model_runner.query_start_loc.gpu[1 : num_reqs + 1] - 1


def executor_forward(
    local_rank: int,
    tp_size: int,
    dp_rank: int,
    distributed_init_method: str,
    vllm_config: VllmConfig,
    error_queue: Queue,
):
    """Spawn target: drive ``TomasuloExecutor`` through prefill + decode.

    Initialisation is identical to ``model_runner_forward``; the forward /
    compute_logits / sample triplet is exercised on both a prefill step
    ({"req_0": 4, "req_1": 3}) and a decode step ({"req_0": 1, "req_1": 1}).
    """
    from vllm_ascend.worker.tomasulo_scheduler.executor import TomasuloExecutor

    try:
        vllm_config = init_worker_env_and_config(
            local_rank, tp_size, dp_rank, vllm_config,
        )

        with set_current_vllm_config(vllm_config):
            worker, kv_cache_config = create_and_init_worker(
                vllm_config, local_rank, local_rank,
                distributed_init_method,
            )

            model_runner = worker.model_runner
            executor = TomasuloExecutor(model_runner)
            block_size = (
                kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            )
            allocator = SimpleBlockAllocator(
                kv_cache_config.num_blocks, block_size,
            )
            vocab_size = model_runner.input_batch.vocab_size

            def _run_step(num_scheduled_tokens: dict[str, int], tag: str):
                (
                    attn_metadata,
                    total_num_tokens,
                    num_tokens_padded,
                    cudagraph_mode,
                    batch_desc,
                    num_tokens_across_dp,
                ) = prepare_forward_step(model_runner, num_scheduled_tokens)

                # Rebuild SamplingMetadata now that input_batch state reflects
                # the current requests (add_request bypasses _update_states).
                model_runner.input_batch.refresh_metadata()

                input_batch = build_tomasulo_input_batch(
                    model_runner,
                    attn_metadata,
                    total_num_tokens,
                    num_tokens_padded,
                    cudagraph_mode,
                    batch_desc,
                    num_tokens_across_dp,
                )

                hidden_states = executor.forward(input_batch)
                assert isinstance(hidden_states, torch.Tensor)
                assert hidden_states.shape[0] >= total_num_tokens
                assert not torch.isnan(hidden_states).any()
                assert not torch.isinf(hidden_states).any()

                num_reqs = model_runner.input_batch.num_reqs
                logits_indices = build_logits_indices(model_runner, num_reqs)
                logits = executor.compute_logits(hidden_states, logits_indices)
                assert isinstance(logits, torch.Tensor)
                assert logits.shape[0] == num_reqs
                assert not torch.isnan(logits).any()
                assert not torch.isinf(logits).any()

                sampled = executor.sample(
                    logits, model_runner.input_batch.sampling_metadata,
                )
                assert isinstance(sampled, torch.Tensor)
                assert sampled.shape == (num_reqs, 1)
                assert sampled.dtype == torch.int32
                assert (sampled >= 0).all()
                assert (sampled < vocab_size).all()

                logger.info(
                    f"DP{dp_rank}-Rank{local_rank}: {tag} passed, "
                    f"hidden_states.shape={hidden_states.shape}, "
                    f"logits.shape={logits.shape}, "
                    f"sampled.shape={sampled.shape}"
                )

            prefill_tokens = {"req_0": 4, "req_1": 3}
            register_forward_test_requests(
                model_runner, prefill_tokens, allocator, block_size,
            )
            _run_step(prefill_tokens, "prefill")

            update_state_after_model_forward(
                model_runner, prefill_tokens, allocator, block_size,
            )

            decode_tokens = {"req_0": 1, "req_1": 1}
            _run_step(decode_tokens, "decode")

    except Exception as e:
        error_queue.put((
            (dp_rank, local_rank),
            f"{type(e).__name__}: {e}",
            traceback.format_exc(),
        ))


def basic_test(engine_args_dict: dict, test_func: Callable):
    """
    Spawn worker(s) via multiprocessing, matching vllm serve behavior:

    The VllmConfig is created once in the parent process so that all
    workers share the same _data_parallel_master_port_list (populated
    by ParallelConfig.__post_init__). This avoids port conflicts that
    would occur if each worker called create_engine_config() independently.

    Dense models: each DP replica is an independent engine with its own
    distributed world (world_size=tp_size, separate init method per replica).

    MoE models: all DP replicas share one global distributed world;
    init_distributed_environment() expands rank via
    global_rank = dp_rank * tp_size + local_rank.
    """
    tp_size = engine_args_dict["tensor_parallel_size"]
    dp_size = engine_args_dict["data_parallel_size"]
    is_moe = engine_args_dict.get("enable_expert_parallel", False)

    engine_args = build_engine_args(engine_args_dict)
    vllm_config = engine_args.create_engine_config()

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()
    processes = []

    for dp_rank in range(dp_size):
        init_method = f"tcp://127.0.0.1:{find_free_port()}"
        for local_rank in range(tp_size):
            p = ctx.Process(
                target=test_func,
                args=(
                    local_rank,
                    tp_size,
                    dp_rank,
                    init_method,
                    vllm_config,
                    error_queue,
                ),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join(timeout=600)

    errors = []
    while not error_queue.empty():
        identity, msg, tb = error_queue.get_nowait()
        errors.append(f"--- {identity} ---\n{msg}\n{tb}")
    if errors:
        pytest.fail("Worker process(es) failed:\n" + "\n".join(errors))

    total = tp_size * dp_size
    mode = "MoE (shared world)" if is_moe else "Dense (independent replicas)"
    print(f"\nAll {total} workers passed ({mode}, tp={tp_size}, dp={dp_size}).")
