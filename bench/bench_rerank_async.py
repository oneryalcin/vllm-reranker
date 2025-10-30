import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


async def sample_metrics(host: str, interval: float, stop_event: asyncio.Event) -> Dict[str, Any]:
    samples: List[Dict[str, float]] = []
    metrics_url = f"{host.rstrip('/')}/metrics"
    pattern_util = re.compile(r"nvml_gpu_utilization\{[^}]*\}\s+(\d+(?:\.\d+)?)")
    pattern_mem = re.compile(r"nvml_memory_used_bytes\{[^}]*\}\s+(\d+(?:\.\d+)?)")

    last_error: Optional[str] = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        while not stop_event.is_set():
            try:
                resp = await client.get(metrics_url)
                resp.raise_for_status()
                text = resp.text
                utils = [float(m.group(1)) for m in pattern_util.finditer(text)]
                mems = [float(m.group(1)) for m in pattern_mem.finditer(text)]
                if utils or mems:
                    samples.append({
                        "util": sum(utils) / len(utils) if utils else None,
                        "mem": sum(mems) / len(mems) if mems else None,
                    })
                last_error = None
            except Exception as exc:
                last_error = str(exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    if not samples:
        return {"error": last_error or "No metrics samples collected"}

    util_values = [s["util"] for s in samples if s.get("util") is not None]
    mem_values = [s["mem"] for s in samples if s.get("mem") is not None]

    summary: Dict[str, float] = {}
    if util_values:
        summary["gpu_util_avg_pct"] = sum(util_values) / len(util_values)
        summary["gpu_util_peak_pct"] = max(util_values)
    if mem_values:
        # Convert bytes to GiB for readability
        summary["gpu_mem_avg_gib"] = (sum(mem_values) / len(mem_values)) / (1024 ** 3)
        summary["gpu_mem_peak_gib"] = max(mem_values) / (1024 ** 3)
    summary["samples"] = len(samples)
    return summary


async def run_benchmark(
    cases,
    url: str,
    concurrency: int,
    batch_size: int,
    metrics_host: Optional[str],
    metrics_interval: float,
) -> Dict[str, Any]:
    queue: asyncio.Queue = asyncio.Queue()
    latencies: list[float] = []

    stop_event = asyncio.Event()
    metrics_task: Optional[asyncio.Task] = None
    metrics_result: Dict[str, Any] = {}

    if metrics_host:
        metrics_task = asyncio.create_task(sample_metrics(metrics_host, metrics_interval, stop_event))

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

    stop_event.set()
    if metrics_task is not None:
        metrics_result = await metrics_task

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

    if metrics_result:
        stats["metrics"] = metrics_result

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous rerank throughput benchmark")
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help="Requests queued per worker")
    parser.add_argument("--metrics", action="store_true", help="Poll /metrics and summarize GPU usage")
    parser.add_argument("--metrics-url", default=None, help="Base URL for metrics (defaults to --url)")
    parser.add_argument("--metrics-interval", type=float, default=1.0, help="Seconds between metric polls")
    args = parser.parse_args()

    cases_path = Path(args.cases_file)
    payload = json.loads(cases_path.read_text())
    cases = payload.get("cases") or payload.get("queries")
    if not cases:
        raise ValueError("No cases found in payload")

    print(f"Loaded {len(cases)} cases from {cases_path} -> benchmarking against {args.url}")

    metrics_host = args.metrics_url if args.metrics_url else args.url if args.metrics else None

    stats = asyncio.run(
        run_benchmark(
            cases,
            args.url,
            args.concurrency,
            args.batch_size,
            metrics_host,
            args.metrics_interval,
        )
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
