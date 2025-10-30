SHELL := /bin/bash
MODAL ?= modal

# Config
PORT ?= 8000
IMAGE ?= vllm/vllm-openai:latest
MODEL ?= mixedbread-ai/mxbai-rerank-base-v2
GPU ?= all
DOCS ?= data/hf_harper_lee.json
QUERY ?= Who wrote 'To Kill a Mockingbird'?

# vLLM reranker hf_overrides mapping required for Mixedbread v2
HF_OVERRIDES := '{"architectures":["Qwen2ForSequenceClassification"],"classifier_from_token":["0","1"],"method":"from_2_way_softmax"}'

.PHONY: help
help:
	@echo "Targets:"
	@echo "  vllm-up-docker   - Run vLLM OpenAI server (Docker)"
	@echo "  vllm-health      - Check server models endpoint"
	@echo "  vllm-client      - Call /v1/rerank with example data"
	@echo "  gpu-check        - Verify Docker can access NVIDIA GPU"
	@echo "  local-run        - Run local wrapper once"
	@echo "  batch-run        - Run batch sample cases"
	@echo "  local-run-mps    - Run local wrapper on Apple MPS"
	@echo "  local-run-cpu    - Run local wrapper on CPU"
	@echo "  modal-deploy     - Deploy Modal app"
	@echo "  modal-serve      - Run Modal dev server (requires 'modal' CLI on PATH)"
	@echo "  modal-serve-uvx  - Run Modal dev server via 'uvx' (no install)"

.PHONY: vllm-up-docker
vllm-up-docker:
	docker run --gpus $(GPU) --rm -p $(PORT):8000 \
	  -e HF_TOKEN=$$HF_TOKEN \
	  $(IMAGE) \
	  python -m vllm.entrypoints.openai.api_server \
	    --host 0.0.0.0 --port 8000 \
	    --model $(MODEL) \
	    --task score \
	    --dtype auto \
	    --gpu-memory-utilization 0.90 \
	    --max-num-seqs 512 \
	    --hf-overrides $(HF_OVERRIDES)

.PHONY: vllm-health
vllm-health:
	curl -s http://localhost:$(PORT)/v1/models | jq . || curl -s http://localhost:$(PORT)/v1/models

.PHONY: vllm-client
vllm-client:
	uv run python src/client_vllm_rerank.py \
	  --url http://localhost:$(PORT) \
	  --query "$(QUERY)" \
	  --docs-file $(DOCS) \
	  --top-k 3 \
	  --model-id $(MODEL)

.PHONY: gpu-check
gpu-check:
	docker run --rm --gpus $(GPU) nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi || true

.PHONY: local-run
local-run:
	uv run --with mxbai-rerank --with transformers --with torch \
	  python src/run_local.py \
	  --query "$(QUERY)" \
	  --docs-file $(DOCS) \
	  --top-k 3 \
	  --model-id $(MODEL) \
	  --device auto \
	  --quiet

.PHONY: batch-run
batch-run:
	uv run --with mxbai-rerank --with transformers --with torch \
	  python src/run_batch.py \
	  --cases-file data/rerank_samples.json \
	  --model-id $(MODEL) \
	  --device auto \
	  --quiet

.PHONY: local-run-mps
local-run-mps:
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	uv run --with mxbai-rerank --with transformers --with torch \
	  python src/run_local.py \
	  --query "$(QUERY)" \
	  --docs-file $(DOCS) \
	  --top-k 3 \
	  --model-id $(MODEL) \
	  --device mps \
	  --quiet

.PHONY: local-run-cpu
local-run-cpu:
	OMP_NUM_THREADS=$$(sysctl -n hw.ncpu) \
	uv run --with mxbai-rerank --with transformers --with torch \
	  python src/run_local.py \
	  --query "$(QUERY)" \
	  --docs-file $(DOCS) \
	  --top-k 3 \
	  --model-id $(MODEL) \
	  --device cpu \
	  --quiet

.PHONY: modal-deploy
modal-deploy:
	$(MODAL) deploy modal_app.py

.PHONY: modal-serve
modal-serve:
	$(MODAL) serve modal_app.py

.PHONY: modal-serve-uvx
modal-serve-uvx:
	uvx modal serve modal_app.py
