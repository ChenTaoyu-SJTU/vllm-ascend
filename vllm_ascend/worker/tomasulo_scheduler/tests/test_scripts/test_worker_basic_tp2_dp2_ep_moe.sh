python -m pytest -skv /vllm-workspace/vllm-ascend/vllm_ascend/worker/tomasulo_scheduler/tests/test_worker_init_and_foward.py \
    --model-dir /data01/cty/models/qwen3-30b-moe \
    --tp 2 --dp 2 --max-model-len 512 --enable-ep
