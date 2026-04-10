import multiprocessing as mp
import socket

import pytest

from vllm_ascend.worker.tomasulo_scheduler.tests.utils import worker_process_fn


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_worker_init_and_execute(engine_args_dict):
    """
    Spawn worker(s) via multiprocessing, run the full NPUWorker
    initialization sequence, then execute_model + sample_tokens
    on a dummy prefill request.
    """
    tp = engine_args_dict["tensor_parallel_size"]
    dp = engine_args_dict["data_parallel_size"]
    world_size = tp * dp

    distributed_init_method = f"tcp://127.0.0.1:{find_free_port()}"

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_process_fn,
            args=(
                rank,
                world_size,
                distributed_init_method,
                engine_args_dict,
                error_queue,
                result_queue,
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

    # Collect and verify results
    results = {}
    while not result_queue.empty():
        item = result_queue.get_nowait()
        tag, rank = item[0], item[1]
        results.setdefault(rank, []).append(item)

    for rank in range(world_size):
        rank_results = results.get(rank, [])
        tags = [r[0] for r in rank_results]
        print(f"\nRank {rank} results: {tags}")

        assert "init_device_ok" in tags, f"Rank {rank}: init_device failed"
        assert "load_model_ok" in tags, f"Rank {rank}: load_model failed"
        assert "memory_ok" in tags, f"Rank {rank}: determine_available_memory failed"
        assert "kv_cache_ok" in tags, f"Rank {rank}: initialize_from_config failed"
        assert "execute_model_ok" in tags, f"Rank {rank}: execute_model failed"
        assert "all_done" in tags, f"Rank {rank}: did not complete"

        # execute_model should return None for all ranks
        exec_result = [r for r in rank_results if r[0] == "execute_model_ok"][0]
        assert exec_result[2] is True, (
            f"Rank {rank}: execute_model should return None"
        )

        # sample_tokens: driver worker (rank 0) should return non-None
        sample_result = [r for r in rank_results if r[0] == "sample_tokens_ok"][0]
        if rank == 0:
            assert sample_result[2] is True, (
                "Rank 0 (driver): sample_tokens should return non-None"
            )

    print(f"\nAll {world_size} workers passed initialization and execution.")
