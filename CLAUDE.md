# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Mixbread Reranker deployment repo for `mixedbread-ai/mxbai-rerank-base-v2`. Supports local testing (Python wrapper), vLLM GPU server (Docker/Modal), and CPU ONNX deployment (Modal). All paths use `uv run` ad-hoc dependency injection to avoid lock files.

## Key Commands

### Local Testing
```bash
# Python wrapper (uses mxbai_rerank)
make local-run QUERY="your query" DOCS=data/example_docs.json
make local-run-mps    # Apple Silicon GPU
make local-run-cpu    # CPU only

# Batch test multiple cases
make batch-run
```

### vLLM Docker (GPU required)
```bash
make vllm-up-docker   # Start server
make vllm-health      # Check server
make vllm-client      # Test /v1/rerank endpoint
make gpu-check        # Verify Docker GPU access
```

### ONNX CPU Path
```bash
# Export + quantize
uv run --with optimum[onnxruntime] --with transformers --with torch \
  python scripts/export_onnx.py --model-id mixedbread-ai/mxbai-rerank-base-v2 --out-dir onnx/mxbai-base

uv run --with onnxruntime --with onnxruntime-tools \
  python scripts/quantize_onnx.py --model-path onnx/mxbai-base/model.onnx --out-path onnx/mxbai-base/model-int8.onnx

# Run local ONNX inference
uv run --with onnxruntime --with transformers \
  python src/run_onnx.py --query "your query" --docs-file data/example_docs.json --model-dir onnx/mxbai-base --model-file model-int8.onnx --top-k 3
```

### Modal Deployment
```bash
# vLLM GPU deployment
make modal-serve      # Dev server (auto-detects modal CLI or uses uvx)
make modal-deploy     # Production

# ONNX CPU deployment
make modal-serve-onnx
make modal-deploy-onnx

# Environment toggle: FAST_BOOT=true for faster cold starts, FAST_BOOT=false (default) for better throughput
FAST_BOOT=true uvx modal serve modal_app.py
```

### Benchmarking
```bash
# Build BEIR test dataset
make bench-build BENCH_DATASET=scifact BENCH_LIMIT=100

# Run async benchmark
make bench-run BENCH_URL=https://<modal-url> CONCURRENCY=16 BENCH_METRICS=1
```

## Architecture

### Key Model Requirements
- Mixedbread v2 reranker requires `--hf-overrides` to map Qwen2 classifier correctly:
  ```json
  {"architectures":["Qwen2ForSequenceClassification"],"classifier_from_token":["0","1"],"method":"from_2_way_softmax"}
  ```
- vLLM needs `--task score` to enable rerank/scoring mode
- Modal vLLM uses `@modal.web_server` pattern (not ASGI) to spawn vLLM subprocess

### File Roles
- `modal_app.py`: vLLM GPU Modal deployment (L4 default). Preloads weights during image build. Mounts HF+vLLM cache volumes. Configurable via `FAST_BOOT` env var
- `modal_app_onnx.py`: CPU-only Modal deployment using int8 quantized ONNX. Builds/quantizes model during image build. Serves `/rerank` + `/health` via FastAPI
- `src/run_local.py`: Local Python wrapper using `mxbai_rerank` package. Supports device selection (auto/cpu/mps/cuda)
- `src/onnx_reranker.py`: Shared ONNX inference logic with prompt formatting, tokenization, and batched scoring
- `src/client_vllm_rerank.py`: HTTP client for `/v1/rerank` endpoint
- `bench/build_beir_subset.py`: BEIR dataset sampling for benchmarks
- `bench/bench_rerank_async.py`: Async throughput/latency testing with optional `/metrics` polling

### Modal Patterns
- Image building uses `uv_pip_install` with pinned versions (vllm==0.11.0, torch==2.8.0)
- Weights preloaded via `.run_function(preload_model, volumes=...)` during image build to minimize cold start latency
- Two cache volumes: `/root/.cache/huggingface` and `/root/.cache/vllm`
- `TORCH_CUDA_ARCH_LIST=89` targets L4 GPUs for faster compilation
- ONNX app builds/quantizes model once into Modal volume, reuses across containers

### ONNX Implementation Details
- Prompt format: Qwen chat template with custom task prompt for binary relevance
- Scoring: extracts yes_loc/no_loc logits, computes `yes_logits - no_logits`
- Padding: ensures multiple-of-8 for performance, up to model max length (8192 default)
- Tokenizer: left-padding with fast tokenizer, handles long docs via truncation

## Development Tips

### Testing New Ideas
Always test with throwaway `uv run` one-liners before editing code:
```bash
uv run --with <libs> python -c "<code>"
```
For longer experiments, write to scratch files but don't commit clutter.

### Dependency Management
All deps installed ad-hoc via `uv run --with <package>`. No requirements.txt. Modal apps pin versions in `.uv_pip_install()`. Update versions in `modal_app.py` and `modal_app_onnx.py` when upgrading.

### Performance Notes
- Batch size: tune via `--max-num-seqs` (vLLM) and client-side chunking
- Dtype: `--dtype auto` selects bf16/fp16 on GPU
- Concurrency: `@modal.concurrent(max_inputs=...)` controls container parallelism
- Cold starts: `FAST_BOOT=true` disables compilation (`--enforce-eager`), faster boot but lower throughput
- Prewarm: keep replicas warm via `scaledown_window` or use `FAST_BOOT=false` for compiler opts

### GPU Requirements
- vLLM Docker path requires NVIDIA GPU + NVIDIA Container Toolkit
- macOS/Windows: use local Python wrapper or Modal deployment
- Apple Silicon: use `make local-run-mps` with MPS fallback enabled

### Data Files
- `data/example_docs.json`: minimal test case
- `data/hf_harper_lee.json`: HF example (mirrors official docs)
- `data/rerank_samples.json`: multilingual, code, long-context cases
- `data/bench/*.json`: generated via `make bench-build`
