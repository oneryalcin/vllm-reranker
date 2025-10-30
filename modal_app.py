import os
import modal

# Modalized vLLM server for Mixedbread reranker.
# Aligns with docs/vllm_modal.md (volumes, image build, web_server pattern).

MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
N_GPU = 1
VLLM_PORT = 8000
# Default to False for better steady-state throughput (enables compilation + cudagraphs).
FAST_BOOT = os.environ.get("FAST_BOOT", "false").lower() == "true"

# Caches to speed up cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Build-step: download weights into HF cache volume
def preload_model() -> None:
    import os
    from huggingface_hub import snapshot_download

    model_id = os.environ.get("MODEL_ID", MODEL_ID)
    # Download the entire repo snapshot into the standard HF cache path.
    snapshot_download(repo_id=model_id)

# Build image similar to docs example; pin versions for reproducibility
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODEL_ID": MODEL_ID,
        "TORCH_CUDA_ARCH_LIST": "89",
    })
    .run_function(
        preload_model,
        volumes={"/root/.cache/huggingface": hf_cache_vol},
    )
)

app = modal.App("mxbai-rerank-vllm")


@app.function(
    image=image,
    gpu=f"L4:{N_GPU}",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess
    import os

    model_id = os.environ.get("MODEL_ID", MODEL_ID)
    import json as _json

    overrides = _json.dumps({
        "architectures": ["Qwen2ForSequenceClassification"],
        "classifier_from_token": ["0", "1"],
        "method": "from_2_way_softmax",
    })

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_id,
        "--served-model-name",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--task",
        "score",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "512",
        "--hf-overrides",
        overrides,
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Avoid shell=True so JSON in --hf-overrides is passed intact.
    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd)
