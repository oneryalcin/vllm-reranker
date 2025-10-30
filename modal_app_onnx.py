from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
ONNX_DIR = Path("/root/onnx")
ONNX_MODEL = "model-int8.onnx"

onnx_volume = modal.Volume.from_name("mxbai-rerank-onnx", create_if_missing=True)


def build_onnx() -> None:
    from scripts.export_onnx import export
    from scripts.quantize_onnx import quantize

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model_fp = ONNX_DIR / "model.onnx"
    quant_fp = ONNX_DIR / ONNX_MODEL

    if quant_fp.exists():
        return

    export(MODEL_ID, ONNX_DIR)
    quantize(model_fp, quant_fp)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "onnxruntime",
        "transformers",
        "optimum[onnxruntime]",
        "torch",
    )
    .add_local_dir("scripts", "/root/app/scripts")
    .add_local_dir("src", "/root/app/src")
    .env({"PYTHONPATH": "/root/app"})
    .run_function(build_onnx, volumes={str(ONNX_DIR): onnx_volume})
)


app = modal.App("mxbai-rerank-onnx-cpu")


_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is None:
        from src.onnx_reranker import OnnxReranker

        _reranker = OnnxReranker(ONNX_DIR, ONNX_MODEL)
    return _reranker


@modal.web_endpoint(method="POST")
def _rerank(request):
    from src.onnx_reranker import RerankRequest

    data = request.json
    reranker = _load_reranker()
    payload = RerankRequest(
        query=data["query"],
        documents=data["documents"],
        top_n=data.get("top_n", 5),
        instruction=data.get("instruction"),
    )
    return {"results": reranker.rerank(payload)}


@app.function(
    image=image,
    cpu=modal.cpu.CPU(4),
    volumes={str(ONNX_DIR): onnx_volume},
    concurrency_limit=8,
)
@modal.web_endpoint(method="POST")
def rerank(request):
    return _rerank(request)


@app.function(image=image, cpu=modal.cpu.CPU(1))
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok"}
