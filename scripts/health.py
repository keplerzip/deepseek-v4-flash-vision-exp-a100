import json, os, sys, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8005'
headers = {}
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open(urllib.request.Request(base.rstrip('/') + '/v1/models', headers=headers), timeout=15) as r:
    data = json.load(r)['data']
assert len(data) == 1 and data[0]['id'] == 'DeepSeek-V4-Flash' and data[0].get('max_model_len') == 262144, data
print('HEALTH=PASS model=DeepSeek-V4-Flash context=262144 release=R3.4')
