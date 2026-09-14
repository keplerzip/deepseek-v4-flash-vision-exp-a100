"""Scored image load ramp and measured-token long-context checks."""

import argparse
import concurrent.futures
import copy
import datetime
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request
from multimodal_suite import (
    content_for,
    differences,
    parse_answer,
    run_case,
    validate_assets,
)

MODEL = 'DeepSeek-V4-Flash'


def request(base, path, payload=None, timeout=7200):
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    req = urllib.request.Request(
        base.rstrip("/") + path,
        headers=headers,
        data=None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def protocol_checks(base, root, case):
    content = content_for(root, case)
    uri = content[1]["image_url"]["url"]
    expected = case["expected"]
    records = []
    for path, payload in [
        (
            "/v1/responses",
            {
                "model": MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": case["prompt"]},
                            {"type": "input_image", "image_url": uri, "detail": "auto"},
                        ],
                    }
                ],
                "max_output_tokens": 4096,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            },
        ),
        (
            "/v1/messages",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": case["prompt"]},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": uri.split(",", 1)[1],
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0,
                "thinking": {"type": "disabled"},
            },
        ),
    ]:
        start = time.monotonic()
        try:
            data = request(base, path, payload)
            if path.endswith("responses"):
                text = "".join(
                    c.get("text", "")
                    for msg in data.get("output", [])
                    if msg.get("type") == "message"
                    for c in msg.get("content", [])
                    if c.get("type") == "output_text"
                )
                assert data.get("status") == "completed", data
            else:
                text = "".join(
                    c.get("text", "")
                    for c in data.get("content", [])
                    if c.get("type") == "text"
                )
                assert data.get("stop_reason") == "end_turn", data
            assert data.get("model") == MODEL, data
            actual = parse_answer(text)
            errors = differences(expected, actual)
            records.append(
                {
                    "id": path,
                    "status": "FAIL" if errors else "PASS",
                    "actual": actual,
                    "expected": expected,
                    "errors": errors,
                    "response": data,
                    "seconds": time.monotonic() - start,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "id": path,
                    "status": "FAIL",
                    "error": str(exc)[:800],
                    "seconds": time.monotonic() - start,
                }
            )
    return records


def long_case(base, root, case, target):
    # The server's multimodal tokenizer measures the exact final prompt,
    # including image sentinels, before the same messages go to inference.
    unit = "这段文字用于上下文长度测试，与图片问题无关。\n"

    def payload(repeats):
        content = content_for(root, case)
        content[0]["text"] = (
            unit * repeats + "\n请忽略上述填充文字，仅根据图片回答：" + case["prompt"]
        )
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"thinking": False},
        }

    def count(n):
        return request(base, "/tokenize", payload(n), timeout=300)["count"]

    baseline = count(0)
    per_unit = max(1, (count(100) - baseline) / 100)
    n = max(0, int((target - baseline) / per_unit))
    actual = count(n)
    for _ in range(8):
        if target - 64 <= actual <= target:
            break
        n = max(
            0, n + int((target - actual) / per_unit) - (1 if actual > target else 0)
        )
        actual = count(n)
    if not target - 64 <= actual <= target:
        raise ValueError(f"cannot construct measured prompt near {target}: {actual}")
    p = payload(n)
    p.update(max_tokens=min(4096, 262144 - actual), temperature=0, seed=42)
    start = time.monotonic()
    data = request(base, "/v1/chat/completions", p)
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("long-context answer truncated")
    answer = parse_answer(choice["message"].get("content"))
    errors = differences(case["expected"], answer)
    usage = data.get("usage", {})
    if usage.get("prompt_tokens") != actual:
        errors.append("measured/inference prompt token mismatch")
    return {
        "id": f"long-image-{target}",
        "status": "FAIL" if errors else "PASS",
        "measured_prompt_tokens": actual,
        "max_output_tokens": p["max_tokens"],
        "total_limit": 262144,
        "seconds": time.monotonic() - start,
        "expected": case["expected"],
        "actual": answer,
        "errors": errors,
        "usage": usage,
        "scope": "one long image request; not 32 simultaneous full-length requests",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--long", action="store_true")
    p.add_argument("--saturation", action="store_true")
    a = p.parse_args()
    root = Path(a.assets)
    m = json.loads((root / "cases.json").read_text())
    validate_assets(root, m)
    cases = {c["id"]: c for c in m["cases"]}
    records = []
    out = Path(a.output)

    def save(record):
        records.append(record)
        with out.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    in [
                        "id",
                        "status",
                        "seconds",
                        "concurrency",
                        "passed",
                        "failed",
                        "measured_prompt_tokens",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if out.exists():
        raise FileExistsError(out)
    models = request(a.base, "/v1/models")["data"]
    assert (
        len(models) == 1
        and {m["id"] for m in models} == {MODEL}
        and all(m["max_model_len"] == 262144 for m in models)
    ), models
    for rec in protocol_checks(a.base, root, cases["image_stream"]):
        save(rec)
    for concurrency in [1, 4, 8, 16, 32]:
        barrier = threading.Barrier(concurrency)

        def job(i):
            case = copy.deepcopy(
                cases["image_multiturn" if i % 2 == 0 else "image_changed_control"]
            )
            case.pop("followup", None)
            case.pop("followup_expected", None)
            # Distinct text prevents accidental identical-request response reuse.
            case["prompt"] = f"请求编号 {i}，无需返回编号。" + case["prompt"]
            barrier.wait(timeout=60)
            return run_case(root, case, a.base, 1800)

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(job, range(concurrency)))
        elapsed = time.monotonic() - start
        failed = sum(r["status"] != "PASS" for r in results)
        latencies = sorted(r["seconds"] for r in results)
        generated = sum(
            (t["response"].get("usage") or {}).get("completion_tokens", 0)
            for r in results
            for t in r["trace"]
        )
        save(
            {
                "id": f"concurrency-{concurrency}",
                "status": "FAIL" if failed else "PASS",
                "concurrency": concurrency,
                "passed": concurrency - failed,
                "failed": failed,
                "seconds": elapsed,
                "client_average_seconds": statistics.mean(latencies),
                "client_p95_seconds": latencies[
                    min(len(latencies) - 1, int(0.95 * len(latencies)))
                ],
                "completion_tokens_per_wall_second": generated / elapsed,
                "results": results,
            }
        )
        if failed:
            break
    if a.long:
        for target in [32768, 65536, 131072, 258048]:
            try:
                save(long_case(a.base, root, cases["image_stream"], target))
            except Exception as exc:
                save(
                    {
                        "id": f"long-image-{target}",
                        "status": "FAIL",
                        "error": str(exc)[:1000],
                    }
                )
                break
    if a.saturation:
        # Explicit full-window C32 capacity stress; may take hours. No reduction
        # in context/concurrency or silent retry under smaller settings.
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(long_case, a.base, root, cases["image_stream"], 258048)
                for _ in range(32)
            ]
            for i, future in enumerate(futures):
                try:
                    r = future.result()
                    r["id"] = f"full-window-c32-{i}"
                    save(r)
                except Exception as exc:
                    save(
                        {
                            "id": f"full-window-c32-{i}",
                            "status": "FAIL",
                            "error": str(exc)[:1000],
                        }
                    )
        save(
            {
                "id": "full-window-c32-duration",
                "status": "PASS",
                "seconds": time.monotonic() - start,
                "scope": "timing only; see all 32 per-request results",
            }
        )
    failed = [r["id"] for r in records if r["status"] != "PASS"]
    summary = {
        "status": "FAIL" if failed else "PASS",
        "model": MODEL,
        "release": "R3.3",
        "failed": failed,
        "long_context_executed": a.long,
        "full_window_c32_executed": a.saturation,
        "records": len(records),
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "inference_requests_executed": True,
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
