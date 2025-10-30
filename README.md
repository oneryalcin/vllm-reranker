# Mixbread Reranker (mxbai-rerank-base-v2)

This repo provides a minimal, fast path to test and deploy the Mixedbread cross-encoder reranker `mixedbread-ai/mxbai-rerank-base-v2` locally and on Modal.

What you get:
- Local Python script using the official `mxbai_rerank` wrapper (fast sanity-check).
- vLLM server setup for the cross-encoder reranker with an OpenAI-compatible `/v1/rerank` endpoint.
- Simple HTTP client to call the vLLM rerank endpoint.
- Modal app using a proper `@modal.web_server`, HF/vLLM cache volumes, and a fast-boot toggle.
- Tips for inference-time optimization (dtype, batching, packing, prewarming).

Prereqs
- Python 3.10+
- GPU recommended for vLLM path (CUDA 12.x). CPU-only is OK for the local Python sanity check.
- Optional: Modal account for deployment.

Quick Start (Local, Python wrapper)
1) Install deps:
   One-off with uv per run:  
   `uv run --with mxbai-rerank --with transformers --with torch`  
   or install once: `pip install mxbai-rerank transformers torch`
2) Run:
   `uv run python src/run_local.py --query "what is vLLM?" --docs-file data/example_docs.json --top-k 3 --model-id mixedbread-ai/mxbai-rerank-base-v2 --quiet`

Relevant sample datasets
- HF example (Harper Lee): `data/hf_harper_lee.json`
- Small multi-case set: `data/rerank_samples.json`
- BEIR workloads: run `make bench-build` to write JSON under `data/bench/` (see `bench/README.md`).

Batch runner
- `uv run --with mxbai-rerank --with transformers --with torch python src/run_batch.py --cases-file data/rerank_samples.json --model-id mixedbread-ai/mxbai-rerank-base-v2 --quiet`

About warnings
- The Transformers stack may emit advisories like "torch_dtype is deprecated; use dtype" and a Qwen2TokenizerFast tip. These are safe to ignore. Use `--quiet` in scripts to suppress.

Example (mirrors HF docs)
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

vLLM Reranker (Local)
vLLM supports cross-encoder rerankers and exposes `/v1/rerank`. The Mixedbread v2 model requires `--hf-overrides` to map the classifier correctly.

GPU prerequisites
- Linux with an NVIDIA GPU and drivers installed.
- NVIDIA Container Toolkit installed so Docker can pass through GPUs.
- Quick check: `make gpu-check` should print `nvidia-smi` output from inside a container.
- On macOS/Windows without an NVIDIA GPU, use the local Python wrapper or deploy to Modal.

Start the server (GPU):

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

Note: `--task score` enables the scoring/rerank path. The server exposes OpenAI-compatible APIs including `/v1/rerank`.

Call it:
`uv run python src/client_vllm_rerank.py --url http://localhost:8000 --query "what is vLLM?" --docs-file data/example_docs.json --top-k 3 --model-id mixedbread-ai/mxbai-rerank-base-v2`

Make targets (shortcut)
- Start server: `make vllm-up-docker`  (customize MODEL/PORT via env)
- Health check: `make vllm-health`
- Call /v1/rerank: `make vllm-client`
- Verify Docker GPU access: `make gpu-check`
- Local wrapper: `make local-run`
- Batch tests: `make batch-run`
 - Apple Silicon GPU: `make local-run-mps` (uses MPS with safe fallback)
 - CPU only: `make local-run-cpu` (sets OMP threads)
- Build BEIR subset: `make bench-build BENCH_DATASET=scifact BENCH_LIMIT=100`
- Benchmark endpoint: `make bench-run BENCH_URL=https://<modal-url> CONCURRENCY=16`

macOS notes
- Apple Silicon GPUs are usable via PyTorch MPS; performance is decent for small batches.
- For best throughput (large batches), prefer a remote GPU like Modal (H100/A10G).

Modal Deployment
This repo includes a Modal app (`modal_app.py`) aligned with `docs/vllm_modal.md` patterns:
- Uses `@modal.web_server` to launch vLLM once per container and serve OpenAI-compatible endpoints.
- Mounts two volumes for caches: `/root/.cache/huggingface` and `/root/.cache/vllm` (faster cold starts).
- Provides a `FAST_BOOT` toggle to trade cold-start speed vs max throughput (`--enforce-eager`). Default is `FAST_BOOT=false` for better steady-state performance; set `FAST_BOOT=true` in the environment for quicker dev spins.
- Sets `--task score` and the required `--hf-overrides` for Mixedbread’s reranker.
- Builds the image with `Image.uv_pip_install(...)` (recommended) and pins `vllm==0.11.0`, `torch==2.8.0`, `huggingface_hub[hf_transfer]==0.35.0`, and `flashinfer-python==0.3.1`.
- Sets `TORCH_CUDA_ARCH_LIST=89` in the container to target NVIDIA L4 GPUs and speed compilation.
- Pre-downloads weights during image build (via `Image.run_function(preload_model, volumes=...)`) into the HF cache volume to minimize first-cold-start latency.

Deploy
- Install CLI (choose one)
  - `uv tool install modal`  (recommended; adds `modal` to PATH)
  - or `pipx install modal`
  - or `pip install modal` (ensure your Python scripts dir is on PATH)
- Dev (hot-reload): `modal serve modal_app.py`  
  - Prints a temporary URL; restart on code changes is automatic. Stop with Ctrl+C.
- Production: `modal deploy modal_app.py`

No CLI? Use uvx
- Dev (hot-reload): `uvx modal serve modal_app.py`
- Production: `uvx modal deploy modal_app.py`
- Toggle cold-start vs throughput: `FAST_BOOT=true uvx modal serve modal_app.py`

Call the deployed endpoint
- Use the printed URL with the client:  
  `uv run python src/client_vllm_rerank.py --url https://<your-app>.modal.run --query "what is vLLM?" --docs-file data/example_docs.json --top-k 3 --model-id mixedbread-ai/mxbai-rerank-base-v2`

Tweaks
- GPU type via `gpu=f"L4:1"` (update if you prefer another SKU, e.g. `H100:1`).
- Concurrency via `@modal.concurrent(max_inputs=...)` (start 16–64 depending on batch size & latency goals).
- Set `FAST_BOOT=false` (default) to enable Torch compilation/CUDA graphs once warm; use `FAST_BOOT=true` for faster cold starts during dev.
- Optionally pre-download weights into a Modal Volume or during image build (advanced; not required for this public model).
- If you change vLLM or CUDA, keep dependencies pinned; the Modal image uses `uv` to install packages.
- Switch model size: edit `MODEL_ID` in `modal_app.py` to `mixedbread-ai/mxbai-rerank-large-v2`, or set `MODEL_ID` env and rebuild.

Preloading details
- The build step calls `huggingface_hub.snapshot_download(MODEL_ID)` while mounting the HF cache Volume at `/root/.cache/huggingface`.
- If you want to skip preloading (to shrink image build time), comment out `.run_function(preload_model, ...)` in `modal_app.py`; weights will then download on first start.

Collecting Web Pages as Markdown
Use `trafilatura` via uv to fetch a page and save Markdown for reranking input:
`uv run --with trafilatura trafilatura --markdown -u "<url>" > output.md`

Optimization Tips
- Batch candidates: larger batches improve GPU utilization; adjust `max_inputs` and client-side chunking.
- Dtype: `--dtype auto` selects bf16/fp16 on GPU for a solid speed/quality balance.
- Packing: tune `--max-num-seqs` and `--gpu-memory-utilization` to keep the device busy.
- Prewarm: keep at least one warm replica or set `FAST_BOOT=False` to allow compiler optimizations.
- Text length: rerank on trimmed snippets first; refine on top-K full passages if needed.
- CPU fallback: the local Python wrapper is simplest; ONNX int8 is possible but out of scope here.

Files
- `src/run_local.py`: Local Python reranker using `mxbai_rerank`.
- `src/client_vllm_rerank.py`: Simple HTTP client for `/v1/rerank`.
- `modal_app.py`: Modal web server launching vLLM with caches and FAST_BOOT.
- `src/run_batch.py`: Batch runner for multiple test cases.
- `data/example_docs.json`: Small example document list.
- `data/hf_harper_lee.json`: HF Harper Lee example passages.
- `data/rerank_samples.json`: Multilingual, code, and long-context cases.
- `bench/`: Utilities for generating benchmark datasets (`build_beir_subset.py`), async throughput tester (`bench_rerank_async.py`), docs.
