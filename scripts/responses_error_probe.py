"""Native HTTP status and replay probes; real server, no mocked gateway."""
import json
import urllib.error
import urllib.request
from cache_probe import BASE, HTTP, request, NAMES

for stream in (False,True):
    body={'model':NAMES[0],'input':[{'type':'item_reference','id':'r39_unknown_item'}],
          'max_output_tokens':16,'stream':stream}
    req=urllib.request.Request(BASE+'/v1/responses',data=json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    try:
        with HTTP.open(req,timeout=60) as response:
            raise AssertionError(f'Unsupported item unexpectedly accepted: HTTP {response.status}')
    except urllib.error.HTTPError as exc:
        data=json.load(exc)
        assert exc.code==400 and 'Unsupported' in data['error']['message'],(exc.code,data)
        print(json.dumps({'check':'responses-unsupported-item','stream':stream,'status':'PASS','http':400}),flush=True)
# Empty assistant messages can appear in saved conversation histories.
body={'model':NAMES[0],'input':[
    {'id':'msg_r39_empty','type':'message','role':'assistant','status':'completed','content':[]},
    {'role':'user','content':'Reply with exactly the integer 34.'}],
    'max_output_tokens':256,'temperature':0,'chat_template_kwargs':{'thinking':False}}
r=request('/v1/responses',body)
text=''.join(part.get('text','') for item in r['output'] if item.get('type')=='message' for part in item.get('content',[]) if part.get('type')=='output_text')
assert r['status']=='completed' and text.strip()=='34',r
print(json.dumps({'check':'responses-empty-assistant-replay','status':'PASS'}),flush=True)
