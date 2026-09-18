"""Optional requests through the operator's New API; never accesses admin APIs."""
import datetime
import json
import os
from pathlib import Path
import urllib.parse
import uuid
import cache_probe as p
import client_protocol_test as c
import multi_image_probe as m

def main():
    base = os.environ['R34_NEWAPI_URL'].rstrip('/')
    if base.endswith('/v1'): base = base[:-3]
    url = urllib.parse.urlsplit(base)
    assert url.scheme in ('http','https') and url.netloc and not url.username and not url.password and not url.query and not url.fragment, 'Invalid New API root URL'
    key = Path('/newapi-token').read_text().strip()
    assert key and '\n' not in key, 'Empty/invalid New API token file'
    p.BASE = c.BASE = base
    p.KEY = c.KEY = key
    p.TRACE = c.TRACE = Path('/results')/('newapi-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.jsonl')
    failed = []
    for path in ('/v1/chat/completions','/v1/responses','/v1/messages'):
        try:
            body = p.payload_for(path,'DeepSeek-V4-Flash',uuid.uuid4().hex,'chart_before.png')
            p.run_one(path,body,{'b_value':65},'newapi-cold',False)
            p.run_one(path,body,{'b_value':65},'newapi-hot-json',True)
            body['stream'] = True
            if path.endswith('chat/completions'): body['stream_options'] = {'include_usage':True}
            p.run_one(path,body,{'b_value':65},'newapi-hot-sse',True)
            body, expected = m.body_for(path,5,False)
            actual, usage, _, _ = p.normalize(path,body,p.request(path,body))
            assert actual == expected and all(type(x) is int for x in actual['numbers']), (actual,expected)
            p.emit({'id':path,'phase':'newapi-five-images','status':'PASS','actual':actual,'raw_usage':usage})
        except Exception as exc:
            failed.append(path)
            p.emit({'id':path,'status':'FAIL','error':str(exc)[:3000]})
    for name, check in (('responses-stream-tools',c.responses),('messages-stream-tools',c.messages),('messages-count-tokens',c.count_tokens)):
        try: p.emit({'id':name,'status':'PASS','details':check()})
        except Exception as exc:
            failed.append(name)
            p.emit({'id':name,'status':'FAIL','error':str(exc)[:3000]})
    p.emit({'NEWAPI_WIRE_TEST':'FAIL' if failed else 'PASS','failed':failed,'result_file':str(p.TRACE),'billing_ledger_checked':False,'scope':'Actual gateway JSON/SSE cache usage, five images and tool roundtrips. Compare these request IDs and cache counts with New API usage logs; no admin/database changes.'})
    return bool(failed)

if __name__=='__main__':
    raise SystemExit(main())
