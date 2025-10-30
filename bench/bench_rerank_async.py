import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import httpx


async def rerank_once(client: httpx.AsyncClient, url: str, case: Dict[str, Any]) -> float:
    payload = {
        "model": case.get("model_id", "mixedbread-ai/mxbai-rerank-base-v2"),
        "query": case["query"],
        "documents": case["documents"],
        "top_n": min(len(case["documents"]), case.get("top_n", 20)),
    }
    start = time.perf_counter()
    resp = await client.post(f"{url.rstrip('/')}/v1/rerank", json=payload)
    resp.raise_for_status()
    return time.perf_counter() - start


async def worker(queue: asyncio.Queue, client: httpx.AsyncClient, url: str, latencies: list[float]):
    while True:
        case = await queue.get()
        if case is None:
            queue.task_done()
            break
        try:
            latency = await rerank_once(client, url, case)
            latencies.append(latency)
        finally:
            queue.task_done()


async def run_benchmark(cases, url: str, concurrency: int, batch_size: int) -> Dict[str, Any]:
    queue: asyncio.Queue = asyncio.Queue()
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=60.0, limits=httpx.Limits(max_connections=concurrency)) as client:
        tasks = [asyncio.create_task(worker(queue, client, url, latencies)) for _ in range(concurrency)]

        for i in range(0, len(cases), batch_size):
            for case in cases[i : i + batch_size]:
                await queue.put(case)

        for _ in tasks:
            await queue.put(None)

        await queue.join()

        for task in tasks:
            await task

    total = sum(latencies)
    latencies_sorted = sorted(latencies)
    stats = {
        "requests": len(latencies),
        "concurrency": concurrency,
        "avg_latency_s": total / len(latencies) if latencies else 0.0,
        "p95_latency_s": latencies_sorted[int(0.95 * len(latencies))] if latencies else 0.0,
        "throughput_rps": len(latencies) / total if total else 0.0,
        "median_latency_s": statistics.median(latencies) if latencies else 0.0,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous rerank throughput benchmark")
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help="Requests queued per worker")
    args = parser.parse_args()

    cases_path = Path(args.cases_file)
    payload = json.loads(cases_path.read_text())
    cases = payload.get("cases") or payload.get("queries")
    if not cases:
        raise ValueError("No cases found in payload")

    print(f"Loaded {len(cases)} cases from {cases_path} -> benchmarking against {args.url}")

    stats = asyncio.run(run_benchmark(cases, args.url, args.concurrency, args.batch_size))

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

