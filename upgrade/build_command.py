"""Reproducible small offline updater from this source tree; no external downloads."""
import base64, gzip, hashlib, json, textwrap
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parent
images=json.loads((here/'runtime-images.json').read_text())
files={}
# Same deployed scripts in source and incremental installations, no divergent variants.
for directory in (root,root/'scripts',root/'config',root/'clients',root/'tests',root/'source'):
    paths=directory.iterdir() if directory in (root,root/'source') else directory.rglob('*')
    for p in paths:
        if not p.is_file() or '__pycache__' in p.parts: continue
        if p.name in ('FILES.sha256','deployment.env') or p.suffix in ('.pyc','.cmd','.gz','.tar'):continue
        if directory==root and p.suffix not in ('.sh','.md','.json') and p.name not in ('VERSION','LICENSE','NOTICE','THIRD_PARTY.md'):continue
        if directory==root/'source' and p.name not in ('overlay-files.txt','overlay-manifest.json','deepseek_v4_renderer.py'):continue
        name=p.relative_to(root).as_posix();data=p.read_bytes()
        files[name]={'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'mode':0o755 if p.suffix=='.sh' else 0o644}
for directory in (root/'source/overlay',root/'source/model-metadata'):
    for p in directory.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        name=p.relative_to(root).as_posix();data=p.read_bytes()
        files[name]={'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'mode':0o644}
ids='\n'.join(i for v in images.values() for i in v['ids'])+'\n'
files['images/accepted-image-ids.txt']={'data':base64.b64encode(ids.encode()).decode(),'sha256':hashlib.sha256(ids.encode()).hexdigest(),'mode':0o644}
payload={'version':'R3.4','revision':'20260916-k6-overlay1','files':files,'runtime_images':images}
(here/'payload.json').write_text(json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n')
assets=''
for name in ('installer.py','payload.json'):
    body=(here/name).read_text();end='R34_'+name.replace('.','_').upper()+'_END'
    assert end not in body
    assets+='cat > "$assets/'+name+'" <<\''+end+'\'\n'+body+'\n'+end+'\n'
script=(here/'controller.sh.in').read_text().replace('@@PAYLOAD_FILES@@',assets)
(here/'upgrade-r34.sh').write_text(script)
digest=hashlib.sha256(script.encode()).hexdigest()
encoded=base64.b64encode(gzip.compress(script.encode(),mtime=0)).decode()
# A heredoc avoids Linux's 128 KiB single-argument limit. Verify the WHOLE
# decoded script before execution, including when delivery was truncated.
command='''#!/usr/bin/env bash
set -euo pipefail
umask 077
r34_tmp=$(mktemp)
trap 'rm -f -- "$r34_tmp"' EXIT
base64 -d <<'R34_BASE64_END' | gzip -dc > "$r34_tmp"
'''+textwrap.fill(encoded,76)+'''\nR34_BASE64_END
[[ $(sha256sum "$r34_tmp" | cut -d ' ' -f 1) == '''+digest+''' ]] || { printf 'R34_CHECKSUM=FAIL; no service was changed\\n' >&2; exit 1; }
bash "$r34_tmp"
'''
(here/'base64-r3.4.cmd').write_text(command)
(here/'delivery.json').write_text(json.dumps({'release':'R3.4','revision':payload['revision'],
    'script_sha256':digest,'command_sha256':hashlib.sha256(command.encode()).hexdigest(),
    'command_bytes':len(command.encode()),'image_transfer_required':False,'new_image_build_required':False,
    'weights_changed':False,'api_authentication':'disabled'},indent=2)+'\n')
print((here/'delivery.json').read_text())
