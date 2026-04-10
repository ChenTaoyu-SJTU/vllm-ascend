import math
import os
import traceback
from multiprocessing.queues import Queue

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
    rank: int,
    world_size: int,
    cli_args_dict: dict,
) -> tuple[VllmConfig, int]:
    """Set up env vars, create VllmConfig, and compute local_rank.

    Must be called at the start of every spawned worker process,
    before set_current_vllm_config and create_and_init_worker.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["VLLM_USE_V1"] = "1"

    engine_args = build_engine_args(cli_args_dict)
    vllm_config = engine_args.create_engine_config()
    local_rank = rank % torch.npu.device_count()
    return vllm_config, local_rank


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


def worker_process_fn(
    rank: int,
    world_size: int,
    distributed_init_method: str,
    cli_args_dict: dict,
    error_queue: Queue,
    result_queue: Queue,
):
    """
    Target function for each spawned worker process.
    Runs the full NPUWorker initialization sequence and then
    execute_model + sample_tokens on a dummy prefill request.
    """
    try:
        vllm_config, local_rank = init_worker_env_and_config(
            rank, world_size, cli_args_dict,
        )

        with set_current_vllm_config(vllm_config):
            worker, kv_cache_config = create_and_init_worker(
                vllm_config, local_rank, rank, distributed_init_method,
            )
            result_queue.put(("init_device_ok", rank))
            result_queue.put(("load_model_ok", rank))
            result_queue.put(("memory_ok", rank))
            result_queue.put(("kv_cache_ok", rank))

            block_size = (
                kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            )
            prompt_token_ids = list(range(1, 33))  # 32 dummy tokens
            scheduler_output = build_scheduler_output(
                prompt_token_ids=prompt_token_ids,
                num_gpu_blocks=kv_cache_config.num_blocks,
                block_size=block_size,
            )

            exec_output = worker.execute_model(scheduler_output)
            result_queue.put(("execute_model_ok", rank, exec_output is None))

            sample_output = worker.sample_tokens(grammar_output=None)
            result_queue.put((
                "sample_tokens_ok", rank, sample_output is not None
            ))

            result_queue.put(("all_done", rank))

    except Exception as e:
        error_queue.put((rank, f"{type(e).__name__}: {e}", traceback.format_exc()))
