"""Runs inside the service; only selected configuration is printed, never keys."""
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
phase = 'R3.4'
cfg = json.loads(Path('/deploy/config/service.json').read_text())
argv = Path('/proc/1/cmdline').read_bytes().decode().rstrip('\x00').split('\x00')

def flag(name):
    if name not in argv:
        return None
    start = argv.index(name) + 1
    out = []
    for x in argv[start:]:
        if x.startswith('--'):
            break
        out.append(x)
    return out
fields = ('served_model_name', 'gpu_memory_utilization', 'max_model_len', 'max_num_seqs', 'tensor_parallel_size', 'enable_prefix_caching', 'enforce_eager', 'speculative_config', 'limit_mm_per_prompt')
actual = {k: flag('--' + k.replace('_', '-')) for k in fields if k not in ('enable_prefix_caching', 'enforce_eager')}
actual['prefix_enabled'] = '--enable-prefix-caching' in argv
actual['enforce_eager'] = '--enforce-eager' in argv
env = {k: os.environ.get(k) for k in ('VLLM_USE_V2_MODEL_RUNNER', 'VLLM_USE_BREAKABLE_CUDAGRAPH', 'TOKENIZERS_PARALLELISM')}
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/renderers/deepseek_v4.py')
sha = hashlib.sha256(path.read_bytes()).hexdigest()
headers = {}
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open(urllib.request.Request('http://127.0.0.1:8005/v1/models', headers=headers), timeout=30) as r:
    models = [{k: m.get(k) for k in ('id', 'max_model_len')} for m in json.load(r)['data']]
print(json.dumps({'phase': phase, 'disk_config': {k: cfg.get(k) for k in fields}, 'running_argv': actual, 'effective_env': env, 'models': models, 'renderer_sha256': sha}), flush=True)
assert '--api-key' not in argv and not os.environ.get('VLLM_API_KEY') and not os.environ.get('R3_API_KEY')
names = ['DeepSeek-V4-Flash']
assert cfg['served_model_name'] == names[0] and actual['served_model_name'] == names
assert {m['id'] for m in models} == set(names) and len(models) == 1
assert all((m['max_model_len'] == 262144 for m in models))
assert cfg['gpu_memory_utilization'] == 0.92 and actual['gpu_memory_utilization'] == ['0.92']
for k, v in (('max_model_len', 262144), ('max_num_seqs', 32), ('tensor_parallel_size', 8)):
    assert cfg[k] == v and actual[k] == [str(v)]
assert cfg['enable_prefix_caching'] is True and actual['prefix_enabled']
assert cfg['enforce_eager'] is False and (not actual['enforce_eager'])
assert json.loads(actual['limit_mm_per_prompt'][0]) == {'image': 999}
assert json.loads(actual['speculative_config'][0]) == cfg['speculative_config']
assert '--enable-prompt-tokens-details' in argv and '--enable-force-include-usage' in argv
assert env['VLLM_USE_V2_MODEL_RUNNER'] == env['VLLM_USE_BREAKABLE_CUDAGRAPH'] == '1'
assert sha == '531b7a867bf2391d25be37c49ff34172b6186bd8f4b16f295e8f9532c1633d4a'
print('EFFECTIVE_STATE_R34=PASS MEMORY=0.92 CONTEXT=262144 CONCURRENCY=32 MULTIMODAL=ON IMAGE_LIMIT=999 USAGE_DETAILS=ON AUTH=DISABLED', flush=True)

overlay = json.loads(Path('/deploy/source/overlay-manifest.json').read_text())
for relative, expected in overlay.items():
    path = Path('/usr/local/lib/python3.12/dist-packages') / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
print('R34_RUNTIME_OVERLAY=PASS AUX_REUSE=ON STACKED_WKV=ON DSPARK_K=6', flush=True)
