"""Real requests: single model, true cache reads, JSON/SSE, changed-image control."""
import base64
import datetime
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://host.docker.internal:8005"
KEY = ''
NAMES = ["DeepSeek-V4-Flash"]
TRACE = Path("/results") / ("cache-r33-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
UNIT = "This line is filler for prefix reuse; it contains no answer.\n"

def emit(record):
    line = json.dumps(record, ensure_ascii=False)
    if KEY:
        line = line.replace(KEY, "[REDACTED]")
    with TRACE.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)

LAST_RESPONSE_HEADERS = {}

def request(path, body=None):
    global LAST_RESPONSE_HEADERS
    LAST_RESPONSE_HEADERS = {}
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if KEY:
        headers.update(Authorization="Bearer " + KEY, **{"x-api-key": KEY})
    req = urllib.request.Request(BASE + path, headers=headers, data=None if body is None else json.dumps(body).encode())
    try:
        response = HTTP.open(req, timeout=300)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: " + e.read(2000).decode(errors="replace")) from None
    with response:
        LAST_RESPONSE_HEADERS = {k: v for k, v in response.headers.items() if k.lower() in ("x-request-id", "x-oneapi-request-id", "x-newapi-request-id")}
        if body is None or not body.get("stream"):
            return json.load(response)
        assert "text/event-stream" in response.headers.get("Content-Type", "")
        events, data, event = [], [], ""
        for raw in response:
            line = raw.decode().rstrip("\r\n")
            if not line:
                if data:
                    text = "\n".join(data)
                    if text == "[DONE]":
                        events.append({"type": "done"})
                    else:
                        item = json.loads(text)
                        assert not event or item.get("type") == event, "SSE event/type mismatch"
                        assert not item.get("error") and item.get("type") not in ("error", "response.failed", "response.incomplete"), item
                        events.append(item)
                data, event = [], ""
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line.startswith("event:"):
                event = line[6:].lstrip(" ")
        assert not data, "Truncated SSE event"
        return events

def normalize(path, payload, raw):
    model = payload["model"]
    # This vLLM revision returns its one fixed model name for every protocol.
    if path.endswith("chat/completions"):
        if payload["stream"]:
            assert raw and raw[-1].get("type") == "done", "Missing [DONE]"
            chunks = [r for r in raw if "choices" in r]
            assert chunks and all(r.get("model") == NAMES[0] for r in chunks)
            assert any(c.get("finish_reason") == "stop" for r in chunks for c in r["choices"]), "No successful finish"
            text = "".join(c.get("delta", {}).get("content") or "" for r in chunks for c in r["choices"])
            usage = next(r["usage"] for r in reversed(chunks) if r.get("usage"))
        else:
            assert raw["model"] == NAMES[0] and raw["choices"][0]["finish_reason"] == "stop", raw
            text, usage = raw["choices"][0]["message"]["content"], raw["usage"]
        total, cached = usage["prompt_tokens"], usage["prompt_tokens_details"]["cached_tokens"]
    elif path.endswith("responses"):
        if payload["stream"]:
            complete = [r["response"] for r in raw if r.get("type") == "response.completed"]
            assert len(complete) == 1, "Missing response.completed"
            raw = complete[0]
        assert raw["model"] == NAMES[0] and raw["status"] == "completed", raw
        text = "".join(c.get("text", "") for m in raw["output"] if m.get("type") == "message" for c in m.get("content", []) if c.get("type") == "output_text")
        usage = raw["usage"]
        total, cached = usage["input_tokens"], usage["input_tokens_details"]["cached_tokens"]
    else:
        if payload["stream"]:
            assert raw[-1].get("type") == "message_stop", "Missing message_stop"
            start = next(r["message"] for r in raw if r.get("type") == "message_start")
            assert start["model"] == NAMES[0]
            usage = dict(start.get("usage", {}))
            for r in raw:
                if r.get("type") == "message_delta":
                    usage.update(r.get("usage") or {})
            assert any(r.get("delta", {}).get("stop_reason") == "end_turn" for r in raw)
            text = "".join(r["delta"].get("text", "") for r in raw if r.get("type") == "content_block_delta")
        else:
            assert raw["model"] == NAMES[0] and raw["stop_reason"] == "end_turn", raw
            usage = raw["usage"]
            text = "".join(c.get("text", "") for c in raw["content"] if c["type"] == "text")
        cached = usage["cache_read_input_tokens"]
        total = usage["input_tokens"] + cached + usage.get("cache_creation_input_tokens", 0)
    assert type(total) is int and type(cached) is int and 0 <= cached <= total
    return json.loads(text), usage, total, cached

def payload_for(path, model, tag, image=None, repeats=512):
    question = ('Read the chart image. Return only a JSON object with integer field b_value for group B. No Markdown.'
                if image else 'Return only this JSON object: {"check": 73019}. No Markdown.')
    content = [{"type": "text", "text": "Independent cache audit " + tag + ".\n"}]
    if image:
        data = base64.b64encode((Path("/deploy/tests/multimodal") / image).read_bytes()).decode()
        if path.endswith("messages"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        elif path.endswith("responses"):
            content.append({"type": "input_image", "image_url": "data:image/png;base64," + data, "detail": "auto"})
        else:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
    content.append({"type": "text", "text": UNIT * repeats + "\nIgnore filler. " + question})
    body = {"model": model, "temperature": 0, "stream": False}
    if path.endswith("responses"):
        for c in content:
            if c["type"] == "text":
                c["type"] = "input_text"
        body.update(input=[{"role": "user", "content": content}], max_output_tokens=256, store=False, chat_template_kwargs={"thinking": False})
    else:
        body.update(messages=[{"role": "user", "content": content}], max_tokens=256)
        if path.endswith("messages"):
            body["thinking"] = {"type": "disabled"}
        else:
            body["chat_template_kwargs"] = {"thinking": False}
    return body

def run_one(path, body, expected, phase, require_hit):
    start = time.monotonic()
    raw = request(path, body)
    actual, usage, total, cached = normalize(path, body, raw)
    # Preserve real usage even if answer/caching checks fail.
    errors = []
    if actual != expected or any(type(actual.get(k)) is not type(v) for k, v in expected.items()):
        errors.append("answer mismatch")
    if require_hit and cached <= 0:
        errors.append("no actual prefix-cache hit")
    if phase == "changed-image" and cached >= total // 2:
        errors.append("unexpected reuse after image changed near prompt start")
    reply_ids = []
    for obj in (raw if isinstance(raw, list) else [raw]):
        for item in (obj, obj.get("response", {}), obj.get("message", {})):
            if item.get("id") and item["id"] not in reply_ids:
                reply_ids.append(item["id"])
    record = {"response_ids": reply_ids, "request_headers": LAST_RESPONSE_HEADERS, "id": path, "model": body["model"], "phase": phase, "stream": body["stream"],
              "status": "FAIL" if errors else "PASS", "actual": actual, "expected": expected,
              "input_total": total, "cached_tokens": cached, "raw_usage": usage,
              "seconds": round(time.monotonic() - start, 3), "errors": errors}
    emit(record)
    assert not errors, errors

def main():
    models = request("/v1/models")["data"]
    assert len(models) == 1 and {m["id"] for m in models} == set(NAMES), models
    assert all(m.get("max_model_len") == 262144 for m in models), models
    failed = []
    # Every protocol gets cold JSON, warm JSON, and warm SSE.
    for path, model in [("/v1/chat/completions", NAMES[0]), ("/v1/responses", NAMES[0]), ("/v1/messages", NAMES[0])]:
        for kind in ("text", "image"):
            try:
                tag = uuid.uuid4().hex
                body = payload_for(path, model, tag, "chart_before.png" if kind == "image" else None)
                expected = {"b_value": 65} if kind == "image" else {"check": 73019}
                run_one(path, body, expected, "cold-" + kind, False)
                run_one(path, body, expected, "warm-json-" + kind, True)
                body["stream"] = True
                if path.endswith("chat/completions"):
                    body["stream_options"] = {"include_usage": True}
                run_one(path, body, expected, "warm-sse-" + kind, True)
                if path.endswith("chat/completions"):
                    body.pop("stream_options", None)
                    run_one(path, body, expected, "warm-sse-no-usage-option-" + kind, True)
                if kind == "image":
                    changed = payload_for(path, model, tag, "chart_after.png")
                    run_one(path, changed, {"b_value": 60}, "changed-image", False)
                    run_one(path, changed, {"b_value": 60}, "warm-changed-image", True)
            except Exception as exc:
                failed.append(path + ":" + kind)
                emit({"status": "FAIL", "id": failed[-1], "error": str(exc)[:1800]})
    # Large prefix + tiny uncached suffix exercises sparse attention after cache hit.
    try:
        path = "/v1/chat/completions"
        body = payload_for(path, NAMES[0], uuid.uuid4().hex, "chart_before.png", repeats=6144)
        run_one(path, body, {"b_value": 65}, "cold-long-image", False)
        run_one(path, body, {"b_value": 65}, "warm-long-image", True)
    except Exception as exc:
        failed.append("long-image-cache")
        emit({"status": "FAIL", "id": failed[-1], "error": str(exc)[:1800]})
    emit({"CACHE_CHANNELS_TEST": "FAIL" if failed else "PASS", "failed": failed,
          "result_file": str(TRACE), "scope": "Direct vLLM usage and synthetic answers; New API billing end-to-end is not tested."})
    return bool(failed)

if __name__ == "__main__":
    raise SystemExit(main())
