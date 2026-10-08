"""Real HTTP checks for tool-history tokenization."""
import datetime
import json
from pathlib import Path

from cache_probe import request, NAMES

TRACE = Path('/results') / ('protocol-r39-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')


def emit(record):
    line = json.dumps(record, ensure_ascii=False)
    with TRACE.open('a') as out:
        out.write(line + '\n')
    print(line, flush=True)


def tokenize_history():
    messages = [
        {'role': 'user', 'content': 'Register the service port 8006.'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'Read the service port.',
         'tool_calls': [{'id': 'call_r39_port', 'type': 'function', 'function': {
             'name': 'record_port', 'arguments': '{"port":8006}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_r39_port', 'content': 'saved'},
    ]
    tools = [{'type': 'function', 'function': {
        'name': 'record_port', 'description': 'Register the service port.',
        'parameters': {'type': 'object', 'properties': {'port': {'type': 'integer'}},
                       'required': ['port'], 'additionalProperties': False}}}]
    result = request('/tokenize', {'model': NAMES[0], 'messages': messages,
                                   'tools': tools})
    assert type(result['count']) is int and result['count'] == len(result['tokens']) > 0, result
    assert all(type(t) is int for t in result['tokens']), result
    decoded = request('/detokenize', {'model': NAMES[0], 'tokens': result['tokens']})
    assert 'record_port' in decoded['prompt'] and '8006' in decoded['prompt'], decoded
    emit({'check': 'tokenize-tool-history', 'status': 'PASS', 'count': result['count']})


failed = []
try:
    tokenize_history()
except Exception as exc:
    failed.append('tokenize-tool-history')
    emit({'check': failed[-1], 'status': 'FAIL', 'error': str(exc)})
emit({'R39_PROTOCOL': 'FAIL' if failed else 'PASS', 'failed': failed,
      'result_file': str(TRACE), 'real_http_requests_attempted': True,
      'inference_requests_executed': False})
raise SystemExit(bool(failed))
