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
from vllm.logger import logger
import multiprocessing as mp
import socket

from typing import Callable
import pytest


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
    # This is a workaround when using data parallel in vllm parallel_state.py
    vllm_config.parallel_config._data_parallel_master_port_list = [
        "32135"
    ]
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


def worker_init_and_execute(
    rank: int,
    world_size: int,
    distributed_init_method: str,
    cli_args_dict: dict,
    error_queue: Queue,
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
            # Do worker initialization
            worker, kv_cache_config = create_and_init_worker(
                vllm_config, local_rank, rank, distributed_init_method,
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
                f"Rank {rank}: execute_model should return None"
            assert sample_output is not None, "sample_tokens should return non-None"
            logger.info(f"Rank {rank}: sample_tokens output: \
                        {sample_output.get_output().sampled_token_ids}")

    except Exception as e:
        error_queue.put((rank, f"{type(e).__name__}: {e}", traceback.format_exc()))

# TODO: Implement the model_runner_forward test, which is similar to 
# worker_init_and_execute() but use the model_runner._model_forward directly
def model_runner_forward():
    pass


def basic_test(engine_args_dict: dict, test_func: Callable):
    """
    Spawn worker(s) via multiprocessing, run the full NPUWorker
    initialization sequence, then execute_model + sample_tokens
    on a dummy prefill request.
    """
    tp_size = engine_args_dict["tensor_parallel_size"]
    dp_size = engine_args_dict["data_parallel_size"]
    world_size = tp_size * dp_size

    distributed_init_method = f"tcp://127.0.0.1:{find_free_port()}"

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=test_func,
            args=(
                rank,
                world_size,
                distributed_init_method,
                engine_args_dict,
                error_queue,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=600)

    # Collect errors
    errors = []
    while not error_queue.empty():
        rank, msg, tb = error_queue.get_nowait()
        errors.append(f"--- Rank {rank} ---\n{msg}\n{tb}")
    if errors:
        pytest.fail("Worker process(es) failed:\n" + "\n".join(errors))

    print(f"\nAll {world_size} workers passed initialization and execution.")
