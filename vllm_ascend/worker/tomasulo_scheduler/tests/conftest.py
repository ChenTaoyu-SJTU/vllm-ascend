import json

import pytest


def pytest_addoption(parser):
    parser.addoption("--tp", type=int, default=1,
                     help="tensor_parallel_size")
    parser.addoption("--dp", type=int, default=1,
                     help="data_parallel_size")
    parser.addoption("--enable-ep", action="store_true", default=False,
                     help="enable_expert_parallel")
    parser.addoption("--max-num-seqs", type=int, default=256,
                     help="max_num_seqs")
    parser.addoption("--max-model-len", type=int, default=32768,
                     help="max_model_len")
    parser.addoption("--model-dir", type=str, default="Qwen/Qwen3-0.6B",
                     help="model path")
    parser.addoption("--compilation-config", type=str, default=None,
                     help="JSON string for CompilationConfig")


@pytest.fixture(scope="session")
def engine_args_dict(request):
    comp_cfg = request.config.getoption("--compilation-config")
    return {
        "model": request.config.getoption("--model-dir"),
        "tensor_parallel_size": request.config.getoption("--tp"),
        "data_parallel_size": request.config.getoption("--dp"),
        "enable_expert_parallel": request.config.getoption("--enable-ep"),
        "max_num_seqs": request.config.getoption("--max-num-seqs"),
        "max_model_len": request.config.getoption("--max-model-len"),
        "compilation_config": json.loads(comp_cfg) if comp_cfg else {},
        "enforce_eager": True,
        "load_format": "auto",
    }
