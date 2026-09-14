"""Install into a fresh directory; validate the entire payload before any write."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

def install(old, new, payload_path, model, cache, engine_image):
    payload = json.loads(payload_path.read_text())
    assert payload['version'] == 'R3.3' and payload['revision'] == '20260914-final-noauth'
    assert new.is_dir() and not any(new.iterdir()), 'Candidate directory must be empty'
    records = {}
    for name, item in payload['files'].items():
        data = base64.b64decode(item['data'], validate=True)
        assert hashlib.sha256(data).hexdigest() == item['sha256'], 'Embedded hash mismatch: '+name
        assert item['mode'] in (0o644, 0o755)
        records[name] = (data, item['mode'])
    for name, item in payload['copy_from_old'].items():
        path = old/name
        assert path.is_file() and path.resolve().is_relative_to(old.resolve()), 'Missing/unsafe source: '+name
        assert path.stat().st_size == item['bytes'], 'Original file size mismatch: '+name
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], 'Original file hash mismatch: '+name
        assert name not in records
        records[name] = (data, 0o644)
    for name in records:
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and str(p) == name and '\n' not in name
    matches = [info for tag, info in payload['runtime_images'].items()
               if engine_image == tag or engine_image in info['ids']]
    assert len(matches) == 1, 'Unrecognized frozen runtime image'
    image_info = matches[0]
    lib, mode = records['scripts/lib.sh']
    lib = lib.decode()
    old_line = 'IMAGE=' + payload['base_image']
    assert lib.count(old_line + '\n') == 1
    lib = lib.replace(old_line + '\n', 'IMAGE=' + engine_image + '\n')
    lib = lib.replace('== 2026.09.10-r3-vision-rc1', '== ' + image_info['release'])
    lib = lib.replace('R3.3 增量版需要原始 R3 镜像', 'R3.3 增量运行镜像标签不匹配')
    records['scripts/lib.sh'] = (lib.encode(), mode)
    records['images/accepted-image-ids.txt'] = (('\n'.join(image_info['ids']) + '\n').encode(), 0o644)
    cfg = json.loads(records['config/service.json'][0])
    assert cfg['served_model_name'] == 'DeepSeek-V4-Flash' and cfg['limit_mm_per_prompt'] == {'image':999}
    assert cfg['gpu_memory_utilization'] == .92 and cfg['max_model_len'] == 262144 and cfg['max_num_seqs'] == 32
    assert cfg['decoding'] == 'dspark' and cfg['speculative_config']['num_speculative_tokens'] == 6 and not cfg['enforce_eager']
    assert Path(model).is_absolute() and Path(cache).is_absolute()
    for name, (data, mode) in records.items():
        path = new/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
    env = 'MODEL_DIR='+shlex.quote(model)+'\nR3_RUNTIME_DIR=\n'
    (new/'deployment.env').write_text(env)
    (new/'deployment.env').chmod(0o600)
    runtime = new/'runtime'
    runtime.mkdir(mode=0o700)
    (runtime/'cache').symlink_to(cache, target_is_directory=True)
    manifest = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name,(data,_) in sorted(records.items()))
    (new/'FILES.sha256').write_text(manifest)
    print(json.dumps({'INSTALL_PAYLOAD':'PASS','files':len(records),'model':'DeepSeek-V4-Flash','image_limit':999,'memory':.92,'decoding':'dspark-k6-v2-graph','authentication':'disabled','model_weights_copied':False,'base_image_reused':True}),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('old',type=Path); p.add_argument('new',type=Path); p.add_argument('payload',type=Path)
    p.add_argument('model'); p.add_argument('cache'); p.add_argument('engine_image')
    a=p.parse_args()
    install(a.old,a.new,a.payload,a.model,a.cache,a.engine_image)
