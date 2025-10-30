# Mixbread Reranker (mxbai-rerank-base-v2)

Minimal, fast path to test and deploy the Mixedbread cross-encoder reranker `mixedbread-ai/mxbai-rerank-base-v2` locally and on Modal.

**What you get:**
- Local Python script using the official `mxbai_rerank` wrapper (fast sanity-check)
- vLLM server setup for the cross-encoder reranker with OpenAI-compatible `/v1/rerank` endpoint
- Simple HTTP client to call the vLLM rerank endpoint
- Modal app using `@modal.web_server`, HF/vLLM cache volumes, and a fast-boot toggle
- ONNX export + quantization scripts plus CPU-only Modal service for GPU-free deployment
- Tips for inference-time optimization (dtype, batching, packing, prewarming)

---

## At a Glance

Four ways to run this reranker:

| Path | GPU Required | Complexity | Best For |
|------|-------------|------------|----------|
| **1. Local Python Wrapper** | No (CPU/MPS/CUDA) | Lowest | Quick testing, development |
| **2. Local vLLM (Docker)** | Yes (NVIDIA) | Medium | Local GPU inference, higher throughput |
| **3. Modal vLLM (Cloud)** | Yes (auto-provisioned) | Medium | Production, auto-scaling |
| **4. ONNX CPU** | No | Low-Medium | CPU-only environments, cost optimization |

---

## Prerequisites

**All paths:**
- Python 3.10+

**Option 1 (Local Python):**
- No additional requirements

**Option 2 (Local vLLM GPU):**
- Linux with NVIDIA GPU and drivers
- NVIDIA Container Toolkit for Docker GPU passthrough
- CUDA 12.x
- macOS/Windows users: use Option 1, 3, or 4 instead

**Option 3 (Modal vLLM):**
- Modal account (free tier available)
- Modal CLI: `uv tool install modal` or `pipx install modal`

**Option 4 (ONNX CPU):**
- No additional requirements for local use
- Modal account for cloud deployment

---

## Quick Start

Test the reranker in under 30 seconds with the local Python wrapper:

```bash
uv run --with mxbai-rerank --with transformers --with torch \
  python src/run_local.py \
    --query "what is vLLM?" \
    --docs-file data/example_docs.json \
    --top-k 3 \
    --model-id mixedbread-ai/mxbai-rerank-base-v2 \
    --quiet
```

Or use the shortcut:
```bash
make local-run
```

This downloads the model (~500MB first run), ranks documents by relevance, and prints JSON results with scores.

---

## Deployment Options

### Option 1: Local Python Wrapper

**When to use:** Quick testing, development, or when you don't need high throughput.

**Basic usage:**
```bash
uv run --with mxbai-rerank --with transformers --with torch \
  python src/run_local.py \
    --query "Who wrote 'To Kill a Mockingbird'?" \
    --docs-file data/hf_harper_lee.json \
    --top-k 3 \
    --model-id mixedbread-ai/mxbai-rerank-base-v2 \
    --quiet
```

**Device selection:**
```bash
# Auto-detect best device (default)
make local-run

# Force CPU (sets OMP threads)
make local-run-cpu

# Apple Silicon GPU (MPS with safe fallback)
make local-run-mps

# CUDA GPU (if available)
uv run ... python src/run_local.py --device cuda ...
```

**Batch processing:**

Test multiple queries at once:
```bash
uv run --with mxbai-rerank --with transformers --with torch \
  python src/run_batch.py \
    --cases-file data/rerank_samples.json \
    --model-id mixedbread-ai/mxbai-rerank-base-v2 \
    --quiet
```

Or: `make batch-run`

**Python API example** (mirrors HF docs):
```python
from mxbai_rerank import MxbaiRerankV2

model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")
query = "Who wrote 'To Kill a Mockingbird'?"
documents = [
    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960...",
    "The novel 'Moby-Dick' was written by Herman Melville...",
    "Harper Lee, an American novelist widely known for her novel...",
]
print(model.rank(query, documents, return_documents=True, top_k=3))
```

**Note on warnings:** The Transformers stack may emit advisories like "torch_dtype is deprecated; use dtype" and a Qwen2TokenizerFast tip. These are safe to ignore. Use `--quiet` to suppress.

---

### Option 2: Local vLLM Server (GPU)

**When to use:** You have an NVIDIA GPU and want higher throughput with OpenAI-compatible API.

**GPU prerequisites:**
- Linux with NVIDIA GPU and drivers installed
- NVIDIA Container Toolkit for Docker GPU passthrough
- Quick check: `make gpu-check` should print `nvidia-smi` output from inside a container
- macOS/Windows without NVIDIA GPU: use Option 1, 3, or 4

**Start the server:**

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 \
    --model mixedbread-ai/mxbai-rerank-base-v2 \
    --task score \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 512 \
    --hf-overrides '{"architectures":["Qwen2ForSequenceClassification"],"classifier_from_token":["0","1"],"method":"from_2_way_softmax"}'
```

Or: `make vllm-up-docker`

**Why these flags?**
- `--task score`: Enables the scoring/rerank path (exposes `/v1/rerank`)
- `--hf-overrides`: Required for Mixedbread v2 to map the Qwen2 classifier correctly

**Test the server:**

```bash
# Health check
curl http://localhost:8000/v1/models | jq .
# Or: make vllm-health

# Call /v1/rerank endpoint
uv run python src/client_vllm_rerank.py \
  --url http://localhost:8000 \
  --query "what is vLLM?" \
  --docs-file data/example_docs.json \
  --top-k 3 \
  --model-id mixedbread-ai/mxbai-rerank-base-v2
# Or: make vllm-client
```

**Customization:**
```bash
# Change model or port
make vllm-up-docker MODEL=mixedbread-ai/mxbai-rerank-large-v2 PORT=8080
```

---

### Option 3: Production Deployment (Modal + vLLM)

**When to use:** Production workloads requiring auto-scaling, minimal ops, or cloud GPU access.

**Setup:**

Install Modal CLI (choose one):
```bash
uv tool install modal      # Recommended
# or: pipx install modal
# or: pip install modal
```

Authenticate:
```bash
modal token new
```

**Deploy:**

```bash
# Dev server (hot-reload, temporary URL)
modal serve modal_app.py
# Prints URL like https://yourname--mxbai-rerank-vllm-serve-dev.modal.run
# Ctrl+C to stop

# Production (stable URL)
modal deploy modal_app.py
```

**No CLI installed?** Use `uvx`:
```bash
uvx modal serve modal_app.py    # Dev
uvx modal deploy modal_app.py   # Production
```

**Call the deployed endpoint:**
```bash
uv run python src/client_vllm_rerank.py \
  --url https://<your-app>.modal.run \
  --query "what is vLLM?" \
  --docs-file data/example_docs.json \
  --top-k 3 \
  --model-id mixedbread-ai/mxbai-rerank-base-v2
```

**Configuration:**

The Modal app (`modal_app.py`) provides several knobs:

**FAST_BOOT toggle** (cold-start speed vs throughput):
```bash
# Default (FAST_BOOT=false): enables Torch compilation + CUDA graphs
# Better throughput, slower cold starts (~60-90s)
modal deploy modal_app.py

# Fast boot (FAST_BOOT=true): disables compilation (--enforce-eager)
# Faster cold starts (~20-30s), lower throughput
FAST_BOOT=true modal serve modal_app.py
```

**GPU type** (edit `modal_app.py`):
```python
gpu=f"L4:1"      # Default (cost-effective)
gpu=f"H100:1"    # Faster inference
gpu=f"A10G:1"    # Alternative
```

**Concurrency** (edit `modal_app.py`):
```python
@modal.concurrent(max_inputs=32)  # 16-64 depending on batch size
```

**Model size** (edit `modal_app.py` or set env):
```python
MODEL_ID = "mixedbread-ai/mxbai-rerank-large-v2"
# Or: MODEL_ID=mixedbread-ai/mxbai-rerank-large-v2 modal deploy modal_app.py
```

**How it works:**
- Uses `@modal.web_server` to launch vLLM once per container
- Mounts two volumes: `/root/.cache/huggingface` and `/root/.cache/vllm` for faster cold starts
- Pre-downloads model weights during image build via `Image.run_function(preload_model, ...)`
- Pins dependencies: `vllm==0.11.0`, `torch==2.8.0`, `huggingface_hub[hf_transfer]==0.35.0`, `flashinfer-python==0.3.1`
- Sets `TORCH_CUDA_ARCH_LIST=89` to target L4 GPUs and speed compilation

**Preloading details:**

The build step calls `huggingface_hub.snapshot_download(MODEL_ID)` while mounting the HF cache Volume. This minimizes first-cold-start latency. To skip preloading (faster image builds), comment out `.run_function(preload_model, ...)` in `modal_app.py`; weights will download on first start instead.

**macOS notes:**
- Apple Silicon GPUs work via PyTorch MPS (decent for small batches)
- For large batches/production throughput, prefer Modal cloud GPUs (H100/L4)

---

### Option 4: CPU-Only ONNX

**When to use:** No GPU available, cost optimization, or edge deployment.

#### Local ONNX Workflow

**1. Export to ONNX:**
```bash
uv run --with optimum[onnxruntime] --with transformers --with torch \
  python scripts/export_onnx.py \
    --model-id mixedbread-ai/mxbai-rerank-base-v2 \
    --out-dir onnx/mxbai-base
```

Downloads the HF model and exports to ONNX format.

**2. Quantize (optional but recommended):**
```bash
uv run --with onnxruntime --with onnxruntime-tools \
  python scripts/quantize_onnx.py \
    --model-path onnx/mxbai-base/model.onnx \
    --out-path onnx/mxbai-base/model-int8.onnx
```

Applies dynamic int8 quantization for faster CPU inference.

**3. Run CPU inference:**
```bash
uv run --with onnxruntime --with transformers \
  python src/run_onnx.py \
    --query "Who wrote 'To Kill a Mockingbird'?" \
    --docs-file data/hf_harper_lee.json \
    --model-dir onnx/mxbai-base \
    --model-file model-int8.onnx \
    --top-k 3
```

Set `--model-file model.onnx` to compare fp16 vs int8 performance. Results mirror the JSON structure from the Python wrapper.

#### Modal ONNX Deployment

Deploy the quantized ONNX reranker on Modal CPU containers:

```bash
# Dev server (temporary URL)
make modal-serve-onnx
# Or: modal serve modal_app_onnx.py

# Production
make modal-deploy-onnx
# Or: modal deploy modal_app_onnx.py
```

**How it works:**
- `modal_app_onnx.py` builds and quantizes the model once in a Modal volume during image build
- Serves `/rerank` (POST) and `/health` (GET) endpoints
- Uses FastAPI with 4 CPU cores, 4GB RAM per container
- Auto-scales from 0 to 20 containers

**Request format:**
```json
{
  "query": "your search query",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 5
}
```

**Response:**
```json
{
  "results": [
    {"document": "doc2", "score": 0.95},
    {"document": "doc1", "score": 0.78},
    ...
  ]
}
```

URLs are logged after `serve`/`deploy` finishes.

---

## Advanced Usage

### Batch Processing

Process multiple test cases in one run:

```bash
uv run --with mxbai-rerank --with transformers --with torch \
  python src/run_batch.py \
    --cases-file data/rerank_samples.json \
    --model-id mixedbread-ai/mxbai-rerank-base-v2 \
    --quiet
```

Or: `make batch-run`

The input JSON should contain a `cases` array with `query` and `documents` fields.

---

### Benchmarking

**1. Build a BEIR benchmark dataset:**

```bash
make bench-build BENCH_DATASET=scifact BENCH_LIMIT=100
# Or:
uv run --with beir --with datasets --with tqdm \
  python bench/build_beir_subset.py \
    --dataset scifact \
    --split test \
    --limit 100 \
    --top-k 20 \
    --negatives 20 \
    --out data/bench/scifact_rerank.json
```

Supported datasets: `scifact`, `nq`, `trec-covid`, `fiqa`, etc. See `bench/README.md` for details.

**2. Measure throughput/latency:**

```bash
make bench-run BENCH_URL=https://<modal-url> CONCURRENCY=16
# Or:
uv run --with httpx \
  python bench/bench_rerank_async.py \
    --cases-file data/bench/scifact_rerank.json \
    --url https://<modal-or-local-url> \
    --concurrency 16 \
    --batch-size 1
```

Add `BENCH_METRICS=1` to poll `/metrics` and capture GPU utilization:
```bash
make bench-run BENCH_URL=https://<modal-url> CONCURRENCY=16 BENCH_METRICS=1
```

Optionally set `BENCH_METRICS_URL` if metrics endpoint is separate.

---

### Web Content Collection

Extract web pages as Markdown for reranking input:

```bash
uv run --with trafilatura trafilatura --markdown -u "https://example.com" > output.md
```

Then use `output.md` content as documents in your rerank queries.

---

### Optimization Tips

**Batch candidates:** Larger batches improve GPU utilization. Adjust `max_inputs` in Modal app and client-side chunking.

**Dtype:** `--dtype auto` selects bf16/fp16 on GPU for speed/quality balance.

**Packing:** Tune `--max-num-seqs` and `--gpu-memory-utilization` to keep device busy.

**Prewarm:** Keep at least one warm replica or set `FAST_BOOT=false` to allow compiler optimizations.

**Text length:** Rerank on trimmed snippets first; refine on top-K full passages if needed.

**CPU fallback:** Use ONNX int8 workflow (local or Modal) when GPUs aren't available.

---

## Reference

### Make Targets (Complete List)

**Local Python wrapper:**
```bash
make local-run          # Auto-detect device
make local-run-cpu      # Force CPU
make local-run-mps      # Apple Silicon GPU
make batch-run          # Batch test cases
```

**vLLM Docker:**
```bash
make vllm-up-docker     # Start server
make vllm-health        # Health check
make vllm-client        # Call /v1/rerank
make gpu-check          # Verify Docker GPU access
```

**Modal deployment:**
```bash
make modal-serve        # vLLM dev server
make modal-deploy       # vLLM production
make modal-serve-onnx   # ONNX dev server
make modal-deploy-onnx  # ONNX production
```

**Benchmarking:**
```bash
make bench-build BENCH_DATASET=scifact BENCH_LIMIT=100
make bench-run BENCH_URL=<url> CONCURRENCY=16
make bench-run BENCH_URL=<url> CONCURRENCY=16 BENCH_METRICS=1
```

**Customization via env vars:**
```bash
make vllm-up-docker MODEL=mixedbread-ai/mxbai-rerank-large-v2 PORT=8080
make bench-build BENCH_DATASET=nq BENCH_LIMIT=500 BENCH_TOP_K=10
```

---

### Sample Datasets

- `data/example_docs.json`: Small example document list (quick sanity check)
- `data/hf_harper_lee.json`: HF Harper Lee example passages (mirrors official docs)
- `data/rerank_samples.json`: Multilingual, code, and long-context test cases
- `data/bench/*.json`: Generated via `make bench-build` (BEIR workloads)

---

### File Descriptions

**Python scripts:**
- `src/run_local.py`: Local Python reranker using `mxbai_rerank`
- `src/run_batch.py`: Batch runner for multiple test cases
- `src/client_vllm_rerank.py`: Simple HTTP client for `/v1/rerank`
- `src/run_onnx.py`: CLI wrapper around ONNX reranker (CPU)
- `src/onnx_reranker.py`: Shared ONNX inference logic (tokenization, scoring)

**Modal apps:**
- `modal_app.py`: Modal web server launching vLLM with caches and FAST_BOOT
- `modal_app_onnx.py`: Modal CPU service serving ONNX int8 reranker

**Scripts:**
- `scripts/export_onnx.py`: Exports Mixedbread reranker to ONNX
- `scripts/quantize_onnx.py`: Applies dynamic int8 quantization to ONNX model

**Benchmarking:**
- `bench/build_beir_subset.py`: Generate BEIR benchmark datasets
- `bench/bench_rerank_async.py`: Async throughput/latency tester with metrics polling
- `bench/README.md`: Detailed benchmarking docs

---

### Model Configuration Details

**vLLM HF Overrides (required for Mixedbread v2):**
```json
{
  "architectures": ["Qwen2ForSequenceClassification"],
  "classifier_from_token": ["0", "1"],
  "method": "from_2_way_softmax"
}
```

This mapping is necessary because the Mixedbread reranker uses a Qwen2-based architecture with custom classifier configuration.

**Modal dependencies (pinned in `modal_app.py`):**
- `vllm==0.11.0`
- `torch==2.8.0`
- `huggingface_hub[hf_transfer]==0.35.0`
- `flashinfer-python==0.3.1`

**Environment variables:**
- `HF_TOKEN`: Hugging Face token (optional for public models, required for gated models)
- `HF_HUB_ENABLE_HF_TRANSFER=1`: Faster model downloads
- `TORCH_CUDA_ARCH_LIST=89`: Target L4 GPU architecture for faster compilation
- `FAST_BOOT`: Toggle cold-start speed vs throughput (`true` or `false`)
- `MODEL_ID`: Override model (e.g., `mixedbread-ai/mxbai-rerank-large-v2`)
