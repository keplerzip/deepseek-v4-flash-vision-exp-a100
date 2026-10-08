"""Real frozen dependency checks for R3.7 tool-history and DSpark selection changes."""
import copy
import json
import pickle
from types import SimpleNamespace as NS

import pytest
import torch
from pydantic import ValidationError

from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest


def history():
    return [
        {"role": "user", "content": "read the port"},
        {"role": "assistant", "content": None, "reasoning_content": "read image",
         "tool_calls": [{"id": "call_port", "type": "function", "function": {
             "name": "record_port", "arguments": '{"port":8006}'}}]},
        {"role": "tool", "tool_call_id": "call_port", "content": "saved"},
    ]


@pytest.mark.parametrize("iterable", ["list", "tuple", "generator"])
def test_tool_history_is_serializable_and_keeps_payload(iterable):
    messages = history()
    expected = copy.deepcopy(messages)
    if iterable == "tuple":
        messages[1]["tool_calls"] = tuple(messages[1]["tool_calls"])
    elif iterable == "generator":
        messages[1]["tool_calls"] = iter(messages[1]["tool_calls"])
    request = TokenizeChatRequest(messages=messages)
    assert isinstance(request.messages[1]["tool_calls"], list)
    assert request.messages[1]["tool_calls"] == expected[1]["tool_calls"]
    assert request.messages[1]["reasoning"] == "read image"
    assert pickle.loads(pickle.dumps(request.messages)) == request.messages
    assert copy.deepcopy(request.messages) == request.messages
    assert json.loads(request.model_dump_json())["messages"][1]["tool_calls"]
    if iterable == "list":
        assert messages == expected


def test_explicit_reasoning_wins_over_alias():
    messages = history()
    messages[1]["reasoning"] = "canonical"
    request = TokenizeChatRequest(messages=messages)
    assert request.messages[1]["reasoning"] == "canonical"
    assert messages[1]["reasoning_content"] == "read image"


@pytest.mark.parametrize("calls", [12, "invalid", [{}], [{"type": "wrong"}]])
def test_invalid_tool_calls_fail_validation_before_renderer(calls):
    messages = history()
    messages[1]["tool_calls"] = calls
    with pytest.raises(ValidationError):
        TokenizeChatRequest(messages=messages)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_argmax_cpu_fallback_keeps_ties_nans_and_indices(dtype):
    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
    x = torch.randn(4, 4097, dtype=dtype)
    x[0].zero_()
    x[1, 5] = x[1, 4096] = float("nan")
    x[2].fill_(float("-inf"))
    x[3].zero_()
    x[3, -1] = 1
    out = greedy_argmax(x)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, x.argmax(-1), rtol=0, atol=0)


def test_frozen_k6_greedy_chain_uses_selection_at_every_step():
    from test_r34_runtime import fn
    from contextlib import nullcontext
    calls = []
    rows, steps, vocab = 3, 6, 17
    base = torch.randn(rows * steps, vocab)
    anchors = torch.tensor([1, 2, 3])
    bias = torch.randn(vocab, vocab)
    def select(x):
        calls.append(x.clone())
        return x.argmax(-1)
    spec = NS(num_speculative_steps=steps, sample_indices=torch.arange(rows * steps),
              use_local_argmax_reduction=False, _fused_markov=None, draft_logits=None,
              input_buffers=NS(input_ids=anchors), _anchor_idx=torch.arange(rows),
              sample_idx_mapping=torch.arange(rows * steps),
              sample_pos=torch.arange(rows * steps),
              draft_tokens=torch.zeros(rows, steps, dtype=torch.int64),
              model=NS(compute_draft_logits=lambda x: base, markov_embed=lambda x: x,
                       markov_bias=lambda x: bias[x], map_draft_to_target=lambda x: x))
    run = fn('v1/worker/gpu/spec_decode/dspark/speculator.py', '_sample_sequential',
             'DSparkSpeculator', greedy_argmax=select,
             record_function_or_nullcontext=lambda *a: nullcontext())
    run(spec, rows, torch.zeros(rows * steps, 1))
    previous = anchors
    reference = []
    for i in range(steps):
        previous = (base.view(rows, steps, vocab)[:, i] + bias[previous]).argmax(-1)
        reference.append(previous)
    assert len(calls) == steps
    torch.testing.assert_close(spec.draft_tokens, torch.stack(reference, 1), rtol=0, atol=0)
