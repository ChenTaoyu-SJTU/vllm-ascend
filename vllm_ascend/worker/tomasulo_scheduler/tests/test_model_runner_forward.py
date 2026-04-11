import multiprocessing as mp
import socket

import pytest

from vllm_ascend.worker.tomasulo_scheduler.tests.utils import model_runner_forward, basic_test
from typing import Callable


def test_model_runner_forward(engine_args_dict):
    basic_test(engine_args_dict, model_runner_forward)
