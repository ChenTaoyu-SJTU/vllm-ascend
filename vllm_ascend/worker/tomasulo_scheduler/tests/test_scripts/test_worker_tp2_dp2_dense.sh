python -m pytest -skv /vllm-workspace/vllm-ascend/vllm_ascend/worker/tomasulo_scheduler/tests/test_worker_init_and_foward.py \
    --model-dir /data01/cty/modelscope/hub/models/Qwen/Qwen3-0___6B/ \
    --tp 2 --dp 2 --max-model-len 512
