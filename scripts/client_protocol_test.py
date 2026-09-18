"""Real HTTP/SSE image + function roundtrips; no external tools are executed."""
import base64
import datetime
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

BASE = 'http://host.docker.internal:8005'
MODEL = 'DeepSeek-V4-Flash'
KEY = ''
IMAGE = base64.b64encode(Path('/deploy/tests/multimodal/receipt.png').read_bytes()).decode()
PROMPT = '读取图片中的服务端口，必须调用 record_port 工具登记。收到工具结果后，仅返回 JSON 对象，包含 port 和 receipt 两字段；receipt 必须原样使用工具返回值，不要输出解释或 Markdown。'
SCHEMA = {'type':'object','properties':{'port':{'type':'integer'}},'required':['port'],'additionalProperties':False}
STAMP = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
TRACE = Path('/results') / ('client-protocol-'+STAMP+'.jsonl')

def clean(s):
    return s.replace(KEY,'[REDACTED]') if KEY else s

def wire(path, payload):
    headers = {'Content-Type':'application/json','anthropic-version':'2023-06-01'}
    if KEY:
        headers['Authorization'] = 'Bearer '+KEY
        headers['x-api-key'] = KEY
    req = urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=180) as response:
        if not payload.get('stream'):
            result = json.load(response)
        else:
            assert 'text/event-stream' in response.headers.get('Content-Type','')
            result, data, event = [], [], ''
            for raw in response:
                line = raw.decode().rstrip('\r\n')
                if not line:
                    if data:
                        body = '\n'.join(data)
                        if body != '[DONE]':
                            obj = json.loads(body)
                            assert not event or obj.get('type') == event, (event,obj)
                            if obj.get('type') in ('error','response.failed','response.incomplete'):
                                raise RuntimeError(obj)
                            result.append(obj)
                    data, event = [], ''
                elif line.startswith('data:'):
                    data.append(line[5:].lstrip(' '))
                elif line.startswith('event:'):
                    event = line[6:].lstrip(' ')
            assert not data, 'Unterminated SSE event'
    with TRACE.open('a') as out:
        out.write(clean(json.dumps({'path':path,'response':result},ensure_ascii=False))+'\n')
    return result

def tool_result():
    return {'port':8006,'receipt':'R34_'+uuid.uuid4().hex[:10]}

def responses():
    user = {'role':'user','content':[{'type':'input_text','text':PROMPT},{'type':'input_image','image_url':'data:image/png;base64,'+IMAGE,'detail':'auto'}]}
    body = {'model':MODEL,'input':[user],'tools':[{'type':'function','name':'record_port','description':'Record the image service port and return a receipt.','parameters':SCHEMA}],'tool_choice':'auto','max_output_tokens':1024,'stream':True,'store':False,'temperature':0,'chat_template_kwargs':{'thinking':False}}
    first = wire('/v1/responses',body)
    kinds = [x['type'] for x in first]
    assert 'response.created' in kinds and 'response.function_call_arguments.delta' in kinds
    response = next(x['response'] for x in first if x['type']=='response.completed')
    assert response['model'] == MODEL
    calls = [x for x in response['output'] if x['type']=='function_call']
    assert len(calls)==1 and calls[0]['name']=='record_port', response
    call = calls[0]
    args = json.loads(call['arguments'])
    assert type(args.get('port')) is int and args=={'port':8006}, args
    assert call['call_id']
    expected = tool_result()
    body['input'] = [user,*response['output'],{'type':'function_call_output','call_id':call['call_id'],'output':json.dumps(expected)}]
    body['tool_choice'] = 'none'
    second = wire('/v1/responses',body)
    assert any(x['type']=='response.output_text.delta' for x in second)
    final = next(x['response'] for x in second if x['type']=='response.completed')
    assert final['model'] == MODEL
    actual = json.loads(''.join(c['text'] for x in final['output'] if x['type']=='message' for c in x['content'] if c['type']=='output_text'))
    assert actual == expected, (actual,expected)
    return {'actual':actual,'streamed_function_arguments':True,'tool_result_id_matched':True}

def anthropic_message(events):
    kinds = [x['type'] for x in events]
    assert next(x['message'] for x in events if x['type']=='message_start')['model'] == MODEL
    assert 'message_start' in kinds and kinds[-1]=='message_stop', kinds
    blocks, arguments = {}, {}
    stop = None
    for event in events:
        kind = event['type']
        if kind=='content_block_start':
            blocks[event['index']] = dict(event['content_block'])
        elif kind=='content_block_delta':
            block = blocks[event['index']]
            delta = event['delta']
            if delta['type']=='text_delta':
                block['text'] = block.get('text','') + delta['text']
            elif delta['type']=='input_json_delta':
                i = event['index']
                arguments[i] = arguments.get(i,'') + delta['partial_json']
            elif delta['type']=='thinking_delta':
                block['thinking'] = block.get('thinking','') + delta['thinking']
            elif delta['type']=='signature_delta':
                block['signature'] = block.get('signature','') + delta['signature']
        elif kind=='message_delta':
            stop = event['delta'].get('stop_reason') or stop
    for i, value in arguments.items():
        blocks[i]['input'] = json.loads(value)
    return [blocks[i] for i in sorted(blocks)], stop, arguments

def messages():
    user = {'role':'user','content':[{'type':'text','text':PROMPT},{'type':'image','source':{'type':'base64','media_type':'image/png','data':IMAGE}}]}
    body = {'model':MODEL,'messages':[user],'tools':[{'name':'record_port','description':'Record the image service port and return a receipt.','input_schema':SCHEMA}],'max_tokens':1024,'stream':True,'temperature':0,'chat_template_kwargs':{'thinking':False}}
    first = wire('/v1/messages',body)
    content, stop, deltas = anthropic_message(first)
    calls = [x for x in content if x['type']=='tool_use']
    assert stop=='tool_use' and len(calls)==1 and deltas, (content,stop)
    call = calls[0]
    assert call['name']=='record_port' and call['id']
    assert type(call['input'].get('port')) is int and call['input']=={'port':8006}, call
    expected = tool_result()
    body['messages'] += [{'role':'assistant','content':content},{'role':'user','content':[{'type':'tool_result','tool_use_id':call['id'],'content':json.dumps(expected)}]}]
    body['tool_choice'] = {'type':'none'}
    second = wire('/v1/messages',body)
    final, stop, _ = anthropic_message(second)
    assert stop=='end_turn', (stop,final)
    actual = json.loads(''.join(x['text'] for x in final if x['type']=='text'))
    assert actual==expected, (actual,expected)
    return {'actual':actual,'streamed_tool_arguments':True,'tool_result_id_matched':True}

def count_tokens():
    data = wire('/v1/messages/count_tokens',{'model':MODEL,'messages':[{'role':'user','content':[{'type':'text','text':'Read the image.'},{'type':'image','source':{'type':'base64','media_type':'image/png','data':IMAGE}}]}]})
    assert type(data['input_tokens']) is int and data['input_tokens']>0, data
    return data

def completions():
    # Load the image's pure-Python DeepSeek message encoder without importing GPU modules.
    import runpy
    encoding = runpy.run_path('/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py')
    prompt = encoding['encode_messages']([{'role':'user','content':'Compute 7 + 9. Return only a JSON object with integer field sum. No Markdown.'}], thinking_mode='chat')
    data = wire('/v1/completions', {'model':MODEL, 'prompt':prompt, 'max_tokens':128, 'temperature':0})
    assert data['model'] == MODEL and data['choices'][0]['finish_reason'] == 'stop', data
    actual = json.loads(data['choices'][0]['text'])
    assert actual == {'sum':16} and type(actual['sum']) is int, actual
    return {'actual':actual, 'usage':data['usage']}

def main():
    failed = []
    for name, check in [('completions',completions),('responses-image-stream-tool-roundtrip',responses),('messages-image-stream-tool-roundtrip',messages),('messages-count-tokens',count_tokens)]:
        start = time.monotonic()
        try:
            row = {'id':name,'status':'PASS','details':check()}
        except Exception as exc:
            error = exc.read().decode(errors='replace') if isinstance(exc,urllib.error.HTTPError) else str(exc)
            row = {'id':name,'status':'FAIL','error':error[:6000]}
            failed.append(name)
        row['seconds'] = round(time.monotonic()-start,3)
        print(clean(json.dumps(row,ensure_ascii=True)),flush=True)
        with TRACE.open('a') as out:
            out.write(clean(json.dumps(row,ensure_ascii=False))+'\n')
    print(json.dumps({'CLIENT_PROTOCOL_TEST':'PASS' if not failed else 'FAIL','failed':failed,'trace_file':str(TRACE),'scope':'Actual HTTP/SSE image and ordinary function tools; not full Codex/Claude Code client certification'}),flush=True)
    return bool(failed)

if __name__=='__main__':
    raise SystemExit(main())
