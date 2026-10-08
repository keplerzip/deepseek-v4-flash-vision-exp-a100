"""Native Responses/Chat/Messages image-tool history; actual answer checks."""
import base64
import copy
import datetime
import json
from pathlib import Path
import time
import cache_probe as p

p.TRACE = Path('/results') / ('codex-images-r39-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
PROMPT = ('Read the number printed at the top of EACH image, in the order the images appear in the tool results. '
          'Ignore previous assistant text. Return only a JSON object with the field numbers, an integer array. '
          'Preserve repetitions. No Markdown.')

def content(numbers, protocol='responses'):
    out = []
    for position, number in enumerate(numbers):
        data = base64.b64encode((Path('/deploy/tests/multimodal/video_frames') / f'{number:02}.png').read_bytes()).decode()
        out.append({'type': 'input_text' if protocol == 'responses' else 'text', 'text': f'Image position {position+1}:'})
        if protocol == 'responses':
            # Omit detail deliberately: this is a supported API default.
            out.append({'type': 'input_image', 'image_url': 'data:image/png;base64,' + data})
        elif protocol == 'messages':
            out.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}})
        else:
            out.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}})
    return out


def response_case(numbers, custom=False, reverse=False, history=False):
    make_call = lambda ident: ({'type': 'custom_tool_call', 'name': 'view_image', 'call_id': ident, 'input': 'read local images'}
                              if custom else {'type': 'function_call', 'name': 'view_image', 'call_id': ident, 'arguments': '{}'})
    make_result = lambda ident, sequence: {'type': 'custom_tool_call_output' if custom else 'function_call_output', 'call_id': ident, 'output': content(sequence)}
    tools = [{'type': 'custom', 'name': 'view_image'}] if custom else [{'type': 'function', 'name': 'view_image', 'parameters': {'type': 'object', 'properties': {}}}]
    items = [{'role': 'user', 'content': 'Use the local images to answer my next request.'}]
    if reverse:
        items += [make_call('call_a'), make_call('call_b'), make_result('call_b', numbers[2:]), make_result('call_a', numbers[:2])]
    else:
        items += [make_call('call_a'), make_result('call_a', numbers)]
    if history:
        items += [{'type': 'message', 'id': 'msg_image_history', 'status': 'completed', 'role': 'assistant',
                   'content': [{'type': 'output_text', 'text': 'The local images are available.', 'annotations': []}]}]
    items += [{'role': 'user', 'content': PROMPT}]
    return {'model': 'DeepSeek-V4-Flash', 'input': items, 'tools': tools, 'tool_choice': 'none',
            'store': False, 'max_output_tokens': 512, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}}


def native_case(numbers, protocol, reverse=False):
    if protocol == 'chat':
        messages = [{'role': 'user', 'content': 'Inspect the local images.'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'id': 'call_a', 'function': {'name': 'view_image', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'call_a', 'content': content(numbers, protocol)},
                    {'role': 'user', 'content': PROMPT}]
        if reverse:
            messages[1]['tool_calls'].append({'type': 'function', 'id': 'call_b', 'function': {'name': 'view_image', 'arguments': '{}'}})
            messages[2:3] = [{'role': 'tool', 'tool_call_id': ident, 'content': content(sequence, protocol)}
                             for ident, sequence in [('call_b', numbers[2:]), ('call_a', numbers[:2])]]
        return {'model': 'DeepSeek-V4-Flash', 'messages': messages, 'max_tokens': 512, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}}
    messages = [{'role': 'user', 'content': 'Inspect the local images.'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call_a', 'name': 'view_image', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_a', 'content': content(numbers, protocol)}, {'type': 'text', 'text': PROMPT}]}]
    if reverse:
        messages[1]['content'].append({'type': 'tool_use', 'id': 'call_b', 'name': 'view_image', 'input': {}})
        messages[2]['content'] = [
            {'type': 'tool_result', 'tool_use_id': ident, 'content': content(sequence, protocol)}
            for ident, sequence in [('call_b', numbers[2:]), ('call_a', numbers[:2])]
        ] + [{'type': 'text', 'text': PROMPT}]
    return {'model': 'DeepSeek-V4-Flash', 'messages': messages, 'max_tokens': 512, 'temperature': 0, 'thinking': {'type': 'disabled'}}


def build_cases():
    cases = []
    for count in (2, 8):
        numbers = [1, 3, 2, 1, 2, 3, 3, 1][:count]
        cases.append((f'responses-function-{count}', '/v1/responses', response_case(numbers), numbers))
        cases.append((f'responses-custom-{count}', '/v1/responses', response_case(numbers, custom=True), numbers))
    numbers = [2, 1, 3, 2]
    cases.append(('responses-reversed-tool-results', '/v1/responses', response_case(numbers, reverse=True), numbers))
    cases.append(('responses-image-history', '/v1/responses', response_case([3, 2], history=True), [3, 2]))
    cases.append(('chat-tool-images', '/v1/chat/completions', native_case([1, 2], 'chat'), [1, 2]))
    cases.append(('messages-tool-images', '/v1/messages', native_case([3, 1], 'messages'), [3, 1]))
    # Codex view_image returns one input_image per call, without a text block.
    body = response_case([2, 1], reverse=True)
    body['input'][3]['output'] = [content([1])[1]]
    body['input'][4]['output'] = [content([2])[1]]
    for item in body['input'][3:5]:
        item['output'][0]['detail'] = 'high'
        item['output'][0]['image_url'] = item['output'][0]['image_url'].replace('image/png', 'application/octet-stream')
    cases.append(('responses-codex-view-image-calls', '/v1/responses', body, [2, 1]))
    cases.append(('chat-reversed-tool-images', '/v1/chat/completions', native_case(numbers, 'chat', reverse=True), numbers))
    cases.append(('messages-reversed-tool-images', '/v1/messages', native_case(numbers, 'messages', reverse=True), numbers))
    return cases


def main():
    cases = build_cases()
    failed = []
    for ident, path, original, expected in cases:
        for stream in (False, True):
            body = copy.deepcopy(original); body['stream'] = stream
            if stream and path.endswith('chat/completions'):
                body['stream_options'] = {'include_usage': True}
            row = {'id': ident, 'protocol': path, 'stream': stream, 'expected': {'numbers': expected}}
            started = time.monotonic()
            try:
                actual, usage, total, cached = p.normalize(path, body, p.request(path, body))
                assert actual == {'numbers': expected} and all(type(x) is int for x in actual['numbers']), actual
                row.update(status='PASS', actual=actual, raw_usage=usage)
            except Exception as exc:
                row.update(status='FAIL', error=str(exc)[:2500]); failed.append([ident, stream])
            row['seconds'] = round(time.monotonic()-started, 3)
            row['request_ids'] = dict(p.LAST_RESPONSE_HEADERS)
            p.emit(row)
    p.emit({'R39_CODEX_IMAGE_TEST': 'FAIL' if failed else 'PASS', 'requests': len(cases)*2,
            'failed': failed, 'report': str(p.TRACE), 'scope': 'Synthetic ordered image answers in tool history; Backend via codex_image_test.sh; gateway via newapi_test.sh. Not a 999-image or gateway billing benchmark.'})
    return bool(failed)

if __name__ == '__main__':
    raise SystemExit(main())
