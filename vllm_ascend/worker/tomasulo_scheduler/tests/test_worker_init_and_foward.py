import multiprocessing as mp
import socket

import pytest

from vllm_ascend.worker.tomasulo_scheduler.tests.utils import worker_init_and_execute, basic_test
from typing import Callable


def test_worker_init_and_execute(engine_args_dict):
    basic_test(engine_args_dict, worker_init_and_execute)
