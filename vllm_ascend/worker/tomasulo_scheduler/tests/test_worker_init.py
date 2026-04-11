import multiprocessing as mp
import socket

import pytest

from vllm_ascend.worker.tomasulo_scheduler.tests.utils import worker_init_and_execute


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
    tp_size = engine_args_dict["tensor_parallel_size"]
    dp_size = engine_args_dict["data_parallel_size"]
    world_size = tp_size * dp_size

    distributed_init_method = f"tcp://127.0.0.1:{find_free_port()}"

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_init_and_execute,
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
