#!/usr/bin/env python3
"""CPU contract tests for the client/grader, explicitly not model evaluation."""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mm_suite", ROOT / "scripts/multimodal_suite.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, payload, stream=False):
        self.payload = payload
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()

    def __iter__(self):
        text = json.dumps(
            self.payload["choices"][0]["message"]["content"], ensure_ascii=False
        )
        text = json.loads(text)
        for chunk in [text[:5], text[5:]]:
            yield (
                "data: "
                + json.dumps(
                    {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
                )
                + "\n"
            ).encode()
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":12}}\n'
        yield b"data: [DONE]\n"


def main():
    assets = ROOT / "tests/multimodal"
    manifest = json.loads((assets / "cases.json").read_text())
    module.validate_assets(assets, manifest)
    assert module.parse_answer('```json\n{"port":8006}\n```') == {"port": 8006}
    assert module.differences({"port": 8006}, {"port": 8005})
    assert module.differences({"port": 8006}, {"port": True})
    assert module.differences({"numbers": [1, 2, 3]}, {"numbers": [3, 2, 1]})
    assert module.differences({"device_serial": None}, {"device_serial": "invented"})
    try:
        module.parse_answer("not json")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid JSON accepted")
    results = []
    for case in manifest["cases"]:
        step = [0]

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data)
            assert payload["model"] == "DeepSeek-V4-Flash"
            content = payload["messages"][0]["content"]
            assert any(c["type"] in ["image_url", "video_url"] for c in content)
            for part in content[1:]:
                assert part[part["type"]]["url"].startswith("data:")
            message = {
                "role": "assistant",
                "content": json.dumps(case["expected"], ensure_ascii=False),
            }
            finish = "stop"
            if case.get("tool") and step[0] == 0:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "test-call",
                            "type": "function",
                            "function": {
                                "name": "record_port",
                                "arguments": '{"port":8006}',
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            elif case.get("tool"):
                assert payload["messages"][-1]["role"] == "tool"
                assert payload["messages"][-1]["tool_call_id"] == "test-call"
            elif case.get("followup") and step[0] > 0:
                assert payload["messages"][-1]["content"] == case["followup"]
                message["content"] = json.dumps(case["followup_expected"])
            step[0] += 1
            return Response(
                {
                    "choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"completion_tokens": 12},
                },
                payload.get("stream", False),
            )

        with patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = module.run_case(assets, case, "http://harness.invalid", 5)
        assert result["status"] == "PASS", result
        results.append(
            {"case": case["id"], "client_paths": "PASS", "requests": step[0]}
        )
    bad = lambda request, timeout: Response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"port":8005}'},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    with patch.object(module.urllib.request, "urlopen", side_effect=bad):
        result = module.run_case(
            assets, manifest["cases"][0], "http://harness.invalid", 5
        )
    assert result["status"] == "FAIL"
    report = {
        "status": "PASS",
        "scope": "CPU test harness with simulated HTTP responses; NOT model inference",
        "cases": results,
        "wrong_answer_rejected": True,
        "invalid_json_rejected": True,
        "video_order_error_rejected": True,
        "invented_missing_field_rejected": True,
        "real_inference_requests_executed": False,
        "gpu_inference_tested": False,
    }
    # Deployment sources are mounted read-only; report output is optional.
    if output := os.environ.get("R38_HARNESS_REPORT"):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "harness_paths": len(results),
                "gpu_inference_tested": False,
            }
        )
    )


if __name__ == "__main__":
    main()
