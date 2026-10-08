"""Prepare a verified, flat sibling deployment without modifying the old one."""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import os


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or str(path) != name
            or any(c in name for c in ('\n', '\r', '\x00', ','))):
        raise ValueError('Invalid package path')
    return path


def prepare(old, new, payload):
    old, new = Path(old).resolve(), Path(new).resolve()
    if payload['release'] != 'R3.9' or payload['baseline'] != 'R3.8':
        raise ValueError('Wrong payload version')
    if new == old or old in new.parents or new in old.parents:
        raise ValueError('R3.9 must be in a separate sibling directory')
    cfg = json.loads((old / 'config/service.json').read_text())
    if cfg.get('release') != 'R3.8':
        raise ValueError('The previous container must use an R3.8 deployment')
    previous = {}
    for line in (old / 'FILES.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative_path(name)
        data = (old / name).read_bytes()
        if sha(data) != digest:
            raise ValueError('Previous package checksum mismatch: ' + name)
        previous[name] = data
    for relative, digest in payload['baseline_overlay'].items():
        if sha(previous['source/overlay/' + relative]) != digest:
            raise ValueError('R3.8 overlay changed: ' + relative)
    records = {}
    for name, entry in payload['files'].items():
        relative_path(name)
        if entry['mode'] not in (0o644, 0o755):
            raise ValueError('Invalid file mode: ' + name)
        data = base64.b64decode(entry['data'], validate=True) if 'data' in entry else previous[name]
        if sha(data) != entry['sha256']:
            raise ValueError('R3.9 content mismatch: ' + name)
        records[name] = data
    config = json.loads(records['config/service.json'])
    if not (config['release'] == 'R3.9' and config['served_model_name'] == 'DeepSeek-V4-Flash'
            and config['max_model_len'] == 1048576 and config['max_num_seqs'] == 32
            and config['gpu_memory_utilization'] == 0.92 and config['limit_mm_per_prompt'] == {'image': 999}
            and not config['enforce_eager'] and config['speculative_config']['num_speculative_tokens'] == 6):
        raise ValueError('Fixed service contract mismatch')
    manifest = ''.join(sha(data) + '  ' + name + '\n' for name, data in sorted(records.items()))
    if new.exists():
        if (new / 'FILES.sha256').read_text() != manifest:
            raise ValueError('Destination already contains a different package')
        for name, data in records.items():
            if (new / name).read_bytes() != data:
                raise ValueError('Existing R3.9 file changed: ' + name)
        print('R39_PREPARE=ALREADY_PREPARED', flush=True)
        return
    # All old/new hashes are checked before creating any candidate files.
    staged = Path(tempfile.mkdtemp(prefix='.' + new.name + '.prepare-', dir=new.parent))
    try:
        for name, data in records.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(payload['files'][name]['mode'])
        (staged / 'FILES.sha256').write_text(manifest)
        if new.exists() or new.is_symlink():
            raise ValueError('Destination appeared during preparation')
        os.rename(staged, new)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    print('R39_PREPARE=PASS FILES=' + str(len(records)) + ' INSTALL_DIR=' + str(new), flush=True)
