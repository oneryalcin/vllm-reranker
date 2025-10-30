# Benchmarking

Utilities for collecting rerank workloads and measuring throughput/latency against the vLLM service.

## 1. Build a BEIR subset

```bash
uv run \
  --with beir \
  --with datasets \
  --with tqdm \
  python bench/build_beir_subset.py \
  --dataset scifact \
  --split test \
  --limit 100 \
  --top-k 20 \
  --negatives 20 \
  --out data/bench/scifact_rerank.json
```

This produces a JSON payload with `cases`, each containing a query and mixed positive/negative documents. Adjust `--dataset` (e.g. `nq`, `trec-covid`, `fiqa`) and the limits as needed. For quick smoke tests, reduce `--limit` or `--top-k`.

## 2. Measure throughput against the server

```bash
uv run --with httpx python bench/bench_rerank_async.py \
  --cases-file data/bench/scifact_rerank.json \
  --url https://<modal-or-local-url> \
  --concurrency 16 \
  --batch-size 1
```

The script prints aggregate latency statistics and requests-per-second. Increase `--concurrency` to probe server capacity; increase `--batch-size` to reuse connections for multiple requests per coroutine.

## 3. Compare local vs server

Run the local wrapper for a CPU/MPS baseline:

```bash
make local-run-mps QUERY="<your query>" DOCS=data/bench/scifact_rerank.json
```

Then hit the vLLM endpoint with the benchmark client to contrast latency.

## Notes

- Dependencies are installed ad-hoc via `uv run ... --with <package>` to avoid pin drift.
- The BEIR loader downloads once into the Hugging Face cache. Consider pruning negative samples if you only need positives.
- Modal logs report GPU utilization; combine those with the async benchmark output for a fuller performance picture.
