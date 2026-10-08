# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from typing import cast

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

_TOOL_TEXT_SEPARATOR = "\n\n"


def _order_tool_messages_before_media(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Apply the encoder's tool-result order before collecting image data.

    DeepSeek merges adjacent user/tool messages and sorts tool-result slots
    by the preceding assistant's call order. Sorting only after media parsing
    would reorder placeholders while leaving the image payloads in wire order.
    Preserve user slots, group boundaries and stable ordering of unknown IDs.
    """
    out = list(messages)
    call_order: dict[str, int] = {}
    tool_slots: list[int] = []
    group_active = False
    group_task = None

    def flush() -> None:
        ordered = sorted(
            (out[i] for i in tool_slots),
            key=lambda m: call_order.get(m.get("tool_call_id", ""), 0),
        )
        for index, message in zip(tool_slots, ordered):
            out[index] = message
        tool_slots.clear()

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            if not group_active:
                group_active, group_task = True, None
            tool_slots.append(index)
        elif role == "user":
            if not group_active or group_task is not None:
                flush()
                group_active, group_task = True, message.get("task")
        else:
            flush()
            group_active, group_task = False, None
            if role == "assistant" and message.get("tool_calls"):
                call_order = {}
                for position, call in enumerate(message["tool_calls"]):
                    call_id = call.get("id") or call.get("function", {}).get("id", "")
                    if call_id:
                        call_order[call_id] = position
    flush()
    return out


def _join_tool_text_parts(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Preserve DeepSeek V4 encoder spacing for multipart tool results."""
    out: list[ChatCompletionMessageParam] = []
    for message in messages:
        content = message.get("content")
        if (
            message.get("role") == "tool"
            and isinstance(content, list)
            and content
            and all(
                isinstance(part, dict) and part.get("type") == "text"
                for part in content
            )
        ):
            joined = _TOOL_TEXT_SEPARATOR.join(
                cast(dict, part).get("text", "") for part in content
            )
            message = cast(ChatCompletionMessageParam, {**message, "content": joined})
        out.append(message)
    return out


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        # R3_TOKENIZER_POOL_HF6: same protection used by HfRenderer.
        # Keep the cached original intact; renderer and MM context share this copy.
        tokenizer = copy.copy(tokenizer)
        super().__init__(config, tokenizer)
        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        messages = _join_tool_text_parts(_order_tool_messages_before_media(messages))
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        messages = _join_tool_text_parts(_order_tool_messages_before_media(messages))
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
