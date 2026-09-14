import base64
import gzip
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
template=(here/'controller.sh.in').read_text()
assets=''
for name in ('installer.py','payload.json'):
    body=(here/name).read_text()
    end='R33_'+name.replace('.','_').upper()+'_END'
    assert end not in body
    assets+='cat > "$assets/'+name+'" <<\''+end+'\'\n'+body+'\n'+end+'\n'
script=template.replace('@@PAYLOAD_FILES@@',assets)
(here/'upgrade-r33.sh').write_text(script)
digest=hashlib.sha256(script.encode()).hexdigest()
data=base64.b64encode(gzip.compress(script.encode(),mtime=0)).decode()
command='bash -c \'set -euo pipefail; t=$(mktemp); trap "rm -f -- \\"$t\\"" EXIT; printf %s '+data+' | base64 -d | gzip -dc > "$t"; test "$(sha256sum "$t" | cut -d " " -f 1)" = '+digest+'; bash "$t"\'\n'
# Fully decode and checksum before execution; never execute a truncated pipe.
(here/'base64-r3.3.cmd').write_text(command)
(here/'delivery.json').write_text(json.dumps({'release':'R3.3','revision':'20260914-r3.3-final-noauth','script_sha256':digest,'command_sha256':hashlib.sha256(command.encode()).hexdigest(),'command_bytes':len(command.encode()),'image_transfer_required':False,'new_image_build_required':False,'api_authentication':'disabled'},indent=2)+'\n')
print((here/'delivery.json').read_text())
