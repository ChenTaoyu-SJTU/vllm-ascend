python -m pytest -skv /vllm-workspace/vllm-ascend/vllm_ascend/worker/tomasulo_scheduler/tests/test_executor_forward.py \
    --model-dir /data01/cty/models/Qwen3-0___6B/ \
    --tp 2 --dp 2 --max-model-len 512
