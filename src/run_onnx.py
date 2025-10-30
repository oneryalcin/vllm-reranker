import argparse
import json
import pathlib
from typing import List

import math

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer


def load_docs(docs_file: pathlib.Path) -> List[str]:
    payload = json.loads(docs_file.read_text())
    if isinstance(payload, dict) and "documents" in payload:
        return payload["documents"]
    raise ValueError("Expected JSON file with a 'documents' list")


def ensure_multiple_of_8(length: int, max_value: int) -> int:
    if length % 8 == 0:
        return min(length, max_value)
    padded = (length // 8 + 1) * 8
    return min(padded, max_value)


class OnnxReranker:
    sep = "\n"
    instruction_prompt = "instruction: {instruction}"
    query_prompt = "query: {query}"
    doc_prompt = "document: {document}"
    task_prompt = (
        "You are a search relevance expert who evaluates how well documents match search queries. "
        "For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary "
        "relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"
    )
    chat_template = {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    }

    def __init__(self, model_dir: pathlib.Path, model_file: str, max_length: int = 8192):
        self.model_dir = model_dir
        self.session = ort.InferenceSession(str(model_dir / model_file), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, padding_side="left")
        self.config = AutoConfig.from_pretrained(model_dir)
        self.model_max_length = min(self.config.max_position_embeddings, int(1e9))
        self.max_length = min(max_length, self.model_max_length)

        self._prepare_precomputed_inputs()

    def _ids(self, text: str) -> List[int]:
        return self.tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]

    def _prepare_precomputed_inputs(self):
        self.yes_loc = self._ids("1")[0]
        self.no_loc = self._ids("0")[0]

        self.task_prompt_ids = self._ids(self.task_prompt)
        self.sep_ids = self._ids(self.sep)
        self.prefix_ids = self._ids(self.chat_template["prefix"])
        self.suffix_ids = self._ids(self.chat_template["suffix"])

        self.predefined_length = (
            len(self.prefix_ids)
            + len(self.task_prompt_ids)
            + len(self.suffix_ids)
            + len(self.sep_ids)
        )

        if self.max_length + self.predefined_length > self.model_max_length:
            self.max_length = self.model_max_length - self.predefined_length

        self.max_length_padding = ensure_multiple_of_8(
            max(self.model_max_length, self.max_length + self.predefined_length), self.model_max_length
        )

    def _concat_prompt(self, input_ids: List[int]) -> List[int]:
        return self.prefix_ids + input_ids + self.sep_ids + self.task_prompt_ids + self.suffix_ids

    def prepare_inputs(self, queries: List[str], documents: List[str], instruction: str | None = None) -> dict:
        inputs = []
        instruction_prompt = self.instruction_prompt.format(instruction=instruction) if instruction else None

        for query, document in zip(queries, documents):
            query_prompt = self.query_prompt.format(query=query)
            if instruction_prompt:
                query_prompt = instruction_prompt + self.sep + query_prompt

            query_inputs = self.tokenizer(
                query_prompt,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,
                truncation=True,
            )

            available = self.model_max_length - len(query_inputs["input_ids"]) - self.predefined_length
            doc_maxlen = min(available, self.max_length)

            document_inputs = self.tokenizer(
                self.doc_prompt.format(document=document),
                return_tensors=None,
                add_special_tokens=False,
                max_length=doc_maxlen,
                truncation=True,
            )

            item = self.tokenizer.prepare_for_model(
                query_inputs["input_ids"],
                self.sep_ids + document_inputs["input_ids"],
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )

            ids = self._concat_prompt(item["input_ids"])
            inputs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

        padded = self.tokenizer.pad(
            inputs,
            padding="longest",
            max_length=self.max_length_padding,
            pad_to_multiple_of=8,
            return_tensors="np",
        )

        data = {k: v.astype(np.int64) for k, v in padded.items()}
        if "position_ids" not in data:
            mask = data["attention_mask"]
            position_ids = np.cumsum(mask, axis=1) - 1
            position_ids = np.clip(position_ids, a_min=0, a_max=None).astype(np.int64)
            data["position_ids"] = position_ids
        return data

    def score(self, query: str, documents: List[str]) -> List[float]:
        model_inputs = self.prepare_inputs([query] * len(documents), documents)
        logits = self.session.run(None, model_inputs)[0]
        if logits.ndim == 3:
            yes_logits = logits[:, -1, self.yes_loc]
            no_logits = logits[:, -1, self.no_loc]
        else:
            yes_logits = logits[:, self.yes_loc]
            no_logits = logits[:, self.no_loc]
        scores = yes_logits - no_logits
        return scores.tolist()


def rerank(query: str, documents: List[str], model_dir: pathlib.Path, model_file: str, top_k: int) -> List[dict]:
    reranker = OnnxReranker(model_dir, model_file)
    scores = reranker.score(query, documents)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"document": doc, "score": float(score)} for doc, score in ranked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--docs-file", required=True)
    ap.add_argument("--model-dir", default="onnx/mxbai-base")
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    docs = load_docs(pathlib.Path(args.docs_file))
    results = rerank(args.query, docs, pathlib.Path(args.model_dir), args.model_file, args.top_k)

    print(json.dumps({"query": args.query, "results": results}, indent=2))


if __name__ == "__main__":
    main()
