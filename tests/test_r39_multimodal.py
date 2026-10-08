"""Real API validation, media decoding, tokenizer and CPU image preprocessing.

Only model-class discovery is replaced to avoid loading CUDA inference modules.
These tests do not perform vision-encoder or language-model GPU inference.
"""
import asyncio
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.multimodal.processing import InputProcessingContext
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.deepseek_v4_encoding import encode_messages, IMAGE_PLACEHOLDER
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


def image_part(number, detail=True):
    image = Image.new('RGB', (48 + number * 16, 64), (number * 40, 10, 20))
    data = io.BytesIO(); image.save(data, format='PNG')
    part = {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}
    if detail:
        part['detail'] = 'auto'
    return part


def parts(numbers, detail=True):
    result = []
    for i, number in enumerate(numbers):
        result.extend([{'type': 'input_text', 'text': f'POSITION_{i}'}, image_part(number, detail)])
    return result


def call(call_id):
    return {'type': 'function_call', 'name': 'view_image', 'call_id': call_id, 'arguments': '{}'}


def request_items(kind, numbers, detail=True):
    if kind == 'user':
        return [{'role': 'user', 'content': parts(numbers, detail)}]
    if kind == 'split_user':
        return [{'role': 'user', 'content': parts([n], detail)} for n in numbers]
    custom = kind == 'custom'
    tool = ({'type': 'custom_tool_call', 'name': 'view_image', 'call_id': 'a', 'input': 'inspect'} if custom else call('a'))
    return [
        {'role': 'user', 'content': 'Inspect the supplied images.'}, tool,
        {'type': 'custom_tool_call_output' if custom else 'function_call_output', 'call_id': 'a', 'output': parts(numbers, detail)},
        {'role': 'user', 'content': 'Report the image order.'},
    ]


@pytest.fixture(scope='module')
def processing_context():
    root = Path(os.environ.get('R39_TEST_MODEL_DIR', '/model'))
    assert (root / 'tokenizer.json').is_file(), 'Real Vision-Exp tokenizer required'
    source = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/mm_preprocess.py')
    spec = importlib.util.spec_from_file_location('r39_cpu_deepseek_mm', source)
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    cfg = NS(model=str(root), hf_config=DeepseekV4Config(**json.loads((root / 'config.json').read_text())),
             multimodal_config=MultiModalConfig(limit_per_prompt={'image': 999}, mm_processor_cache_gb=0),
             is_multimodal_model=True, dtype=torch.bfloat16, encoder_config={}, hf_image_processor_config={},
             max_model_len=1048576, allowed_local_media_path='', allowed_media_domains=None, enable_prompt_embeds=False)
    cfg.get_multimodal_config = lambda: cfg.multimodal_config
    tokenizer = get_tokenizer(str(root), tokenizer_mode='deepseek_v4')
    info = mm.DeepseekV4VLProcessingInfo(InputProcessingContext(cfg, tokenizer))
    processor = mm.DeepseekV4VLMultiModalProcessor(info, mm.DeepseekV4VLDummyInputsBuilder(info), cache=None)
    return cfg, tokenizer, processor


@pytest.fixture
def renderer(processing_context, monkeypatch):
    cfg, tokenizer, processor = processing_context
    monkeypatch.setattr(BaseMultiModalItemTracker, 'mm_processor', property(lambda self: processor))
    monkeypatch.setattr(BaseMultiModalItemTracker, 'model_cls', property(lambda self: NS(get_placeholder_str=lambda modality, index: IMAGE_PLACEHOLDER)))
    obj = object.__new__(DeepseekV4Renderer)
    obj.model_config, obj.tokenizer = cfg, tokenizer
    async def apply(*args, **kwargs):
        return obj._apply_chat_template(*args, **kwargs)
    obj._apply_chat_template_async = apply
    return obj, processor


def render_and_check(renderer, items, expected, asynchronous=False):
    original = copy.deepcopy(items)
    req = ResponsesRequest(model='DeepSeek-V4-Flash', input=items, store=False)
    messages = construct_input_messages(request_input=req.input)
    rendered = check_rendered_images(renderer, messages, expected, asynchronous)
    assert items == original
    return rendered


def check_rendered_images(renderer, messages, expected, asynchronous=False, fixture_images=False):
    obj, processor = renderer
    params = ChatParams(chat_template_kwargs={'thinking': False})
    conversation, prompt = (asyncio.run(obj.render_messages_async(messages, params)) if asynchronous else obj.render_messages(messages, params))
    mm_data = prompt['multi_modal_data']
    images = [getattr(image, 'media', image) for image in mm_data['image']]
    if fixture_images:
        reference = [Image.open(Path('/deploy/tests/multimodal/video_frames') / f'{n:02}.png').convert('RGB') for n in expected]
        signature = lambda img: (img.size, hashlib.sha256(img.convert('RGB').tobytes()).hexdigest())
        assert [signature(img) for img in images] == [signature(img) for img in reference]
    else:
        assert [image.getpixel((0, 0))[0] // 40 for image in images] == expected
    placeholder_id = obj.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    assert prompt['prompt_token_ids'].count(placeholder_id) == len(expected)
    result = processor(prompt['prompt_token_ids'], processor.data_parser.parse_mm_data(mm_data))
    placeholders = result['mm_placeholders']['image']
    assert len(placeholders) == len(expected)
    assert len(result['mm_kwargs']['image']) == len(expected)
    assert all(p.length > 0 for p in placeholders)
    assert all(a.offset + a.length <= b.offset for a, b in zip(placeholders, placeholders[1:]))
    assert placeholder_id not in result['prompt_token_ids']
    return conversation, prompt, result


@pytest.mark.parametrize('kind', ['user', 'split_user', 'tool', 'custom'])
@pytest.mark.parametrize('count', [2, 8])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_responses_images_survive_real_tokenization_and_processor(renderer, kind, count, asynchronous):
    numbers = ([1, 3, 2, 1, 2, 3, 3, 1])[:count]
    render_and_check(renderer, request_items(kind, numbers), numbers, asynchronous)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_reversed_tool_outputs_reorder_image_payloads_with_prompt(renderer, asynchronous):
    items = [
        {'role': 'user', 'content': 'Read A then B.'}, call('a'), call('b'),
        {'type': 'function_call_output', 'call_id': 'b', 'output': parts([3, 1])},
        {'role': 'user', 'content': parts([2])},
        {'type': 'function_call_output', 'call_id': 'a', 'output': parts([1, 2])},
        {'role': 'user', 'content': 'Report all images in order.'},
    ]
    conversation, prompt, result = render_and_check(renderer, items, [1, 2, 2, 3, 1], asynchronous)
    tool_ids = [m['tool_call_id'] for m in conversation if m['role'] == 'tool']
    assert tool_ids == ['a', 'b']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_multiple_turns_preserve_all_historical_and_new_images(renderer, asynchronous):
    items = request_items('tool', [1, 2]) + [
        {'type': 'message', 'id': 'msg_history', 'status': 'completed', 'role': 'assistant',
         'content': [{'type': 'output_text', 'text': 'Read the first set.', 'annotations': []}]},
        {'role': 'user', 'content': parts([3])}, call('b'),
        {'type': 'function_call_output', 'call_id': 'b', 'output': parts([2, 1])},
    ]
    render_and_check(renderer, items, [1, 2, 3, 2, 1], asynchronous)


@pytest.mark.parametrize('kind', ['user', 'tool', 'custom'])
def test_missing_detail_defaults_without_losing_tool_images(renderer, kind):
    render_and_check(renderer, request_items(kind, [1, 2], detail=False), [1, 2])


@pytest.mark.parametrize('kind', ['user', 'tool', 'custom'])
def test_bad_image_detail_remains_invalid(kind):
    items = request_items(kind, [1, 2])
    target = items[0]['content'] if kind == 'user' else items[2]['output']
    target[1]['detail'] = 'not-a-supported-detail'
    with pytest.raises(ValidationError):
        ResponsesRequest(model='DeepSeek-V4-Flash', input=items)


def test_999_tool_image_placeholders_are_not_silently_capped():
    messages = [{'role': 'tool', 'tool_call_id': 'a', 'content': [{'type': 'image'}] * 999}]
    text = encode_messages(messages, 'chat')
    assert text.count(IMAGE_PLACEHOLDER) == 999
    assert '[Unsupported image]' not in text


def test_unsupported_tool_media_is_rejected_instead_of_silently_dropped():
    with pytest.raises(ValueError, match='Unsupported content block type'):
        encode_messages([{'role': 'tool', 'content': [{'type': 'video'}]}], 'chat')


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from codex_image_probe import build_cases


@pytest.mark.parametrize('case', build_cases(), ids=lambda case: case[0])
@pytest.mark.parametrize('stream', [False, True])
def test_all_live_probe_payloads_validate_and_preserve_the_exact_images(renderer, case, stream):
    ident, endpoint, body, expected = copy.deepcopy(case)
    original = copy.deepcopy(body)
    body['stream'] = stream
    if endpoint == '/v1/responses':
        req = ResponsesRequest(**body)
        messages = construct_input_messages(request_input=req.input)
    elif endpoint == '/v1/messages':
        req = AnthropicMessagesRequest(**body)
        messages = AnthropicServingMessages._convert_anthropic_to_openai_request(req).messages
    else:
        messages = ChatCompletionRequest(**body).messages
    conversation, _, _ = check_rendered_images(renderer, messages, expected, stream, fixture_images=True)
    if endpoint == '/v1/messages':
        # An image must remain inside its tool result, not an unrelated user turn.
        assert sum(part['type'] == 'image' for msg in conversation if msg['role'] == 'tool'
                   for part in msg['content']) == len(expected)
    body.pop('stream')
    assert body == original


def test_tool_presort_preserves_text_prompt_and_input_objects():
    import itertools
    from vllm.renderers.deepseek_v4 import _order_tool_messages_before_media
    assistant = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': ident, 'type': 'function', 'function': {'name': 'view_image', 'arguments': '{}'}}
        for ident in ('a', 'b', 'c')]}
    for order in itertools.permutations(('a', 'b', 'c', 'unknown')):
        for task in (None, 'analysis'):
            messages = [{'role': 'user', 'content': 'start'}, assistant]
            for ident in order:
                messages += [{'role': 'tool', 'tool_call_id': ident, 'content': f'RESULT_{ident}'},
                             {'role': 'user', 'content': f'AFTER_{ident}', 'task': task}]
            messages += [{'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'next'}]
            original = copy.deepcopy(messages)
            assert encode_messages(messages, 'chat') == encode_messages(_order_tool_messages_before_media(messages), 'chat')
            assert messages == original
