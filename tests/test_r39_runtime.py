"""R3.9 frozen-runtime regressions for the 1M contract and protocol fixes."""

import json
from pathlib import Path

import pytest

from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.deepseek_v4 import _join_tool_text_parts
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.deepseek_v4_encoding import encode_messages


ROOT = Path(__file__).resolve().parents[1]


def test_one_million_context_is_fixed_for_target_and_drafter():
    cfg = json.loads((ROOT / "config/service.json").read_text())
    model = json.loads((ROOT / "source/model-metadata/config.json").read_text())
    assert cfg["release"] == "R3.9"
    assert cfg["served_model_name"] == "DeepSeek-V4-Flash"
    assert cfg["max_model_len"] == cfg["speculative_config"]["max_model_len"] == 1048576
    assert model["max_position_embeddings"] == 1048576
    assert cfg["max_num_seqs"] == 32
    assert cfg["limit_mm_per_prompt"] == {"image": 999}


def test_generation_prompt_flags_are_honored():
    user = [{"role": "user", "content": "Hello"}]
    default = encode_messages(user, "chat")
    omitted = encode_messages(user, "chat", add_generation_prompt=False)
    assert default.endswith("<｜Assistant｜></think>")
    assert not omitted.endswith("<｜Assistant｜></think>")
    assert omitted.endswith("Hello")

    assistant = user + [{"role": "assistant", "content": "Partial answer"}]
    fresh = encode_messages(assistant, "chat", add_generation_prompt=True)
    continuation = encode_messages(assistant, "chat", continue_final_message=True)
    assert fresh.endswith("Partial answer<｜end▁of▁sentence｜><｜Assistant｜></think>")
    assert continuation.endswith("Partial answer")
    with pytest.raises(ValueError):
        encode_messages(user, "chat", continue_final_message=True)


@pytest.mark.parametrize(
    "partial,expected",
    [
        ('{"path": "abc', '"}'),
        ('{"items": [', ']}'),
        ('{"count": ', 'null}'),
        ('{"a": 1, ', ''),
        (r'{"text": "ab\u12', ''),
    ],
)
def test_streamed_tool_json_can_be_closed_without_inventing_arguments(partial, expected):
    suffix = ParserEngine._json_prefix_terminator(partial)
    assert suffix == expected
    if suffix:
        json.loads(partial + suffix)


def test_truncated_tool_string_flushes_valid_json():
    parser = object.__new__(ParserEngine)
    parser._arg_converter = lambda args, partial: None
    slot = ToolCallSlot()
    slot.streamed_json = '{"path": "abc'
    parser._tool_slots = [slot]
    suffix = parser._flush_arg_converter(0)
    assert suffix == '"}'
    assert json.loads(slot.streamed_json) == {"path": "abc"}


def test_unfinished_nonstring_field_is_not_streamed_with_a_dangling_key():
    prefix = ParserEngine._safe_arg_prefix('{"path": "README.md", "count": "1', {"path"})
    assert prefix == '{"path": "README.md"'


class _FakeTokenizer:
    def get_added_vocab(self):
        return {}

    def get_vocab(self):
        return {"<think>": 50, "</think>": 51}

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_request_tools_follow_existing_instruction(role):
    tokenizer = get_deepseek_v4_tokenizer(_FakeTokenizer())
    messages = [
        {"role": role, "content": "INSTRUCTION_SENTINEL"},
        {"role": "user", "content": "hello"},
    ]
    tools = [
        {"type": "function", "function": {
            "name": "probe", "parameters": {"type": "object", "properties": {}},
        }}
    ]
    original = json.dumps(messages, ensure_ascii=False)
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    assert prompt.index("INSTRUCTION_SENTINEL") < prompt.index("## Tools")
    assert prompt.count("## Tools") == 1
    assert json.dumps(messages, ensure_ascii=False) == original


def test_multipart_tool_text_keeps_encoder_blank_line_spacing():
    messages = [
        {"role": "user", "content": "weather"},
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ]},
    ]
    joined = _join_tool_text_parts(messages)
    assert joined[-1]["content"] == "line 1\n\nline 2"
    assert isinstance(messages[-1]["content"], list)
    mixed = [{**messages[-1], "content": [
        {"type": "text", "text": "line 1"}, {"type": "tool_reference", "id": "x"},
    ]}]
    assert _join_tool_text_parts(mixed) == mixed


def _tool_request():
    tool = {"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
        }},
    }}
    return ChatCompletionRequest(model="DeepSeek-V4-Flash", messages=[], tools=[tool])


def _tool_block():
    return (
        "<｜DSML｜tool_calls>"
        '<｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Let me check.\n\n", "Let me check."),
        ("Let me check.\n\n\n", "Let me check.\n"),
        ("Let me check.\n", "Let me check.\n"),
        ("Let me check.", "Let me check."),
    ],
)
def test_tool_separator_removed_once_when_client_echoes_turn(content, expected):
    parser = DeepSeekV4Parser(_FakeTokenizer(), tools=_tool_request().tools)
    result = parser.extract_tool_calls(content + _tool_block(), _tool_request())
    assert result.tools_called
    assert result.content == expected
    assert result.tool_calls[0].function.name == "get_weather"


def test_streamed_content_separator_is_held_until_tool_or_more_text():
    request = _tool_request()
    parser = DeepSeekV4Parser(_FakeTokenizer(), tools=request.tools)
    chunks = ["Let me check.", "\n", "\n", _tool_block()]
    previous = ""
    emitted = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        if delta and delta.content:
            emitted.append(delta.content)
        previous = current
    final = parser.finish_streaming()
    if final and final.content:
        emitted.append(final.content)
    assert "".join(emitted) == "Let me check."

    parser = DeepSeekV4Parser(_FakeTokenizer(), tools=request.tools)
    previous = ""
    emitted = []
    for chunk in ["Let me check.", "\n", "\n", "Then more."]:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        if delta and delta.content:
            emitted.append(delta.content)
        previous = current
    final = parser.finish_streaming()
    if final and final.content:
        emitted.append(final.content)
    assert "".join(emitted) == "Let me check.\n\nThen more."
