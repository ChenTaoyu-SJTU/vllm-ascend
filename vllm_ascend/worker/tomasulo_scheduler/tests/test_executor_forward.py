from vllm_ascend.worker.tomasulo_scheduler.tests.utils import (
    basic_test,
    executor_forward,
)


def test_executor_forward(engine_args_dict):
    basic_test(engine_args_dict, executor_forward)
