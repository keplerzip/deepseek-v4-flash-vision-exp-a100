#!/usr/bin/env python3
"""Offline visual fixtures, scored HTTP checks, and raw reproducible evidence."""

import argparse
import base64
import datetime
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
import unicodedata
import urllib.error
import urllib.request


def parse_answer(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("answer must be a JSON object")
    return result


def differences(expected, actual, prefix=""):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{prefix}: expected object"]
        for key, value in expected.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                errors.append(path + ": missing")
            else:
                errors.extend(differences(value, actual[key], path))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{prefix}: array shape/order mismatch"]
        for index, (left, right) in enumerate(zip(expected, actual)):
            errors.extend(differences(left, right, f"{prefix}[{index}]"))
    elif isinstance(expected, str):
        normalize = lambda value: "".join(
            unicodedata.normalize("NFKC", value).casefold().split()
        )
        if not isinstance(actual, str) or normalize(expected) != normalize(actual):
            errors.append(prefix + ": text mismatch")
    elif expected is None:
        if actual is not None:
            errors.append(prefix + ": expected null")
    elif (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or actual != expected
    ):
        errors.append(prefix + ": numeric mismatch")
    return errors


def validate_assets(root, manifest):
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("unsafe fixture path")
        if (
            path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("fixture hash mismatch: " + item["path"])
    known = {item["path"] for item in manifest["files"]}
    for case in manifest["cases"]:
        for name in case.get("images", []) + (
            [case["video"]] if "video" in case else []
        ):
            if name not in known:
                raise ValueError("unlisted fixture: " + name)


def content_for(root, case):
    content = [{"type": "text", "text": case["prompt"]}]
    for name in case.get("images", []) + ([case["video"]] if "video" in case else []):
        mime = mimetypes.guess_type(name)[0]
        if not mime or not mime.startswith(("image/", "video/")):
            raise ValueError("unsupported fixture media type")
        uri = (
            "data:"
            + mime
            + ";base64,"
            + base64.b64encode((root / name).read_bytes()).decode()
        )
        kind = "video_url" if mime.startswith("video/") else "image_url"
        content.append({"type": kind, kind: {"url": uri}})
    return content


def api_call(base, payload, timeout, trace):
    headers = {"Content-Type": "application/json"}
    # Traces retain prompts and response data, while representing large media
    # by the fixture manifest instead of repeating base64 in each result.
    started = time.monotonic()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode(),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not payload.get("stream"):
            result = json.load(response)
        else:
            events = []
            text = ""
            reasoning = ""
            finish = None
            usage = None
            ended = False
            first = None
            for raw in response:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    ended = True
                    break
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                events.append(event)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    value = delta.get("content") or ""
                    thought = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if (value or thought) and first is None:
                        first = time.monotonic() - started
                    text += value
                    reasoning += thought
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
            if not ended:
                raise ValueError("SSE stream missing [DONE]")
            result = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "reasoning_content": reasoning,
                        },
                        "finish_reason": finish,
                    }
                ],
                "usage": usage,
                "stream_event_count": len(events),
                "ttft_seconds": first,
                "last_events": events[-2:],
            }
    trace.append({"seconds": round(time.monotonic() - started, 3), "response": result})
    if not result.get("choices"):
        raise ValueError("response choices missing")
    if result["choices"][0].get("finish_reason") == "length":
        raise ValueError("output token budget exhausted")
    return result["choices"][0]["message"]


def run_case(root, case, base, timeout):
    trace = []
    start = time.monotonic()
    record = {
        "id": case["id"],
        "category": case["category"],
        "expected": case["expected"],
        "prompt": case["prompt"],
        "images": case.get("images", []),
        "video": case.get("video"),
        "trace": trace,
    }
    try:
        messages = [{"role": "user", "content": content_for(root, case)}]
        payload = {
            "model": 'DeepSeek-V4-Flash',
            "messages": messages,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 4096,
            "chat_template_kwargs": {"thinking": False},
        }
        if case.get("stream"):
            payload.update(stream=True, stream_options={"include_usage": True})
        if case.get("tool"):
            tool = {
                "type": "function",
                "function": {
                    "name": "record_port",
                    "description": "提交图片中读取的服务端口",
                    "parameters": {
                        "type": "object",
                        "properties": {"port": {"type": "integer"}},
                        "required": ["port"],
                    },
                },
            }
            payload.update(tools=[tool], tool_choice="auto")
        message = api_call(base, payload, timeout, trace)
        if case.get("tool"):
            calls = message.get("tool_calls") or []
            if len(calls) != 1 or calls[0]["function"]["name"] != "record_port":
                raise ValueError("expected exactly one record_port tool call")
            actual = json.loads(calls[0]["function"]["arguments"])
            errors = differences(case["expected"], actual)
            if errors:
                raise ValueError("; ".join(errors))
            messages += [
                message,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "端口登记成功：8006。请只返回 JSON，字段 port，值为已登记端口。",
                },
            ]
            message = api_call(base, payload, timeout, trace)
        actual = parse_answer(message.get("content"))
        errors = differences(case["expected"], actual)
        record.update(actual=actual, errors=errors)
        if case.get("followup"):
            messages += [message, {"role": "user", "content": case["followup"]}]
            reply = api_call(base, payload, timeout, trace)
            answer = parse_answer(reply.get("content"))
            errors += differences(case["followup_expected"], answer, "followup")
            record.update(
                followup=case["followup"],
                followup_actual=answer,
                followup_expected=case["followup_expected"],
            )
        record["status"] = "FAIL" if errors else "PASS"
    except urllib.error.HTTPError as error:
        detail = error.read(16384).decode(errors="replace")
        key = os.environ.get("R3_API_KEY")
        if key:
            detail = detail.replace(key, "[REDACTED]")
        record.update(
            status="FAIL", error_type="HTTPError", http_status=error.code, error=detail
        )
    except Exception as error:
        record.update(
            status="FAIL", error_type=type(error).__name__, error=str(error)[:600]
        )
    record["seconds"] = round(time.monotonic() - start, 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be 1–10")
    root = Path(args.assets)
    manifest = json.loads((root / "cases.json").read_text())
    validate_assets(root, manifest)
    cases = [
        case
        for case in manifest["cases"]
        if args.mode == "full" or case["id"] in manifest["smoke_cases"]
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("x") as stream:
        for repeat in range(args.repeat):
            for case in cases:
                record = run_case(root, case, args.base, args.timeout)
                record.update(repeat=repeat + 1, release="R3.9")
                results.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ["id", "status", "seconds", "repeat"]
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    passed = sum(r["status"] == "PASS" for r in results)
    summary = {
        "status": "PASS" if passed == len(results) else "FAIL",
        "model": 'DeepSeek-V4-Flash',
        "release": "R3.9",
        "mode": args.mode,
        "cases_executed": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "fixture_manifest_sha256": hashlib.sha256(
            (root / "cases.json").read_bytes()
        ).hexdigest(),
        "failed_cases": [r["id"] for r in results if r["status"] != "PASS"],
        "scope": manifest["scope"],
        "inference_requests_executed": True,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return passed != len(results)


if __name__ == "__main__":
    raise SystemExit(main())
