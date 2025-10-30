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


@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    volumes={str(ONNX_DIR): onnx_volume},
    max_containers=20,
    min_containers=0,
    buffer_containers=1,
    scaledown_window=300,
    concurrency_limit=8,
)
class OnnxRerankService:
    @modal.enter()
    def setup(self):
        from src.onnx_reranker import OnnxReranker

        self.reranker = OnnxReranker(ONNX_DIR, ONNX_MODEL)

    @modal.fastapi_endpoint(method="POST")
    def rerank(self, request):
        from src.onnx_reranker import RerankRequest

        data = request.json
        payload = RerankRequest(
            query=data["query"],
            documents=data["documents"],
            top_n=data.get("top_n", 5),
            instruction=data.get("instruction"),
        )
        return {"results": self.reranker.rerank(payload)}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok"}
