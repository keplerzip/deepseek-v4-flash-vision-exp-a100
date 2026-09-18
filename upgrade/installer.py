"""Validate offline content before installing into a new directory."""
import argparse, base64, hashlib, json, shlex
from pathlib import Path, PurePosixPath

def install(new, payload_path, model, cache, image):
    p=json.loads(payload_path.read_text())
    assert p['version']=='R3.4' and p['revision']=='20260916-k6-overlay1'
    assert new.is_dir() and not any(new.iterdir()), 'Candidate must be empty'
    assert image in {i for v in p['runtime_images'].values() for i in v['ids']}, 'Unknown base image'
    assert Path(model).is_absolute() and Path(cache).is_absolute()
    records={}
    for name,item in p['files'].items():
        path=PurePosixPath(name)
        assert name and not path.is_absolute() and '..' not in path.parts and str(path)==name
        assert not any(c in name for c in ('\n','\r','\x00',','))
        data=base64.b64decode(item['data'],validate=True)
        assert hashlib.sha256(data).hexdigest()==item['sha256'],name
        assert item['mode'] in (0o644,0o755)
        records[name]=(data,item['mode'])
    cfg=json.loads(records['config/service.json'][0])
    assert cfg['served_model_name']=='DeepSeek-V4-Flash'
    assert cfg['max_model_len']==262144 and cfg['max_num_seqs']==32
    assert cfg['gpu_memory_utilization']==.92 and cfg['tensor_parallel_size']==8
    assert cfg['limit_mm_per_prompt']=={'image':999} and cfg['enable_prefix_caching']
    assert cfg['speculative_config']['num_speculative_tokens']==6 and not cfg['enforce_eager']
    lib,mode=records['scripts/lib.sh'];old='IMAGE=dsv4-flash-a100:20260914-r3.3\n'
    assert lib.decode().count(old)==1
    records['scripts/lib.sh']=(lib.decode().replace(old,'IMAGE='+image+'\n').encode(),mode)
    records['images/accepted-image-ids.txt']=((image+'\n').encode(),0o644)
    for name,(data,mode) in records.items():
        dest=new/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data);dest.chmod(mode)
    (new/'deployment.env').write_text('MODEL_DIR='+shlex.quote(model)+'\nR3_RUNTIME_DIR=\n')
    (new/'deployment.env').chmod(0o600)
    (new/'runtime').mkdir(mode=0o700)
    (new/'runtime/cache').symlink_to(cache,target_is_directory=True)
    (new/'FILES.sha256').write_text(''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name,(data,_) in sorted(records.items())))
    print(json.dumps({'INSTALL':'PASS','version':'R3.4','files':len(records),'image_reused':True,'weights_copied':False,'api_key_required':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('new',type=Path);p.add_argument('payload',type=Path)
    p.add_argument('model');p.add_argument('cache');p.add_argument('image');a=p.parse_args()
    install(a.new,a.payload,a.model,a.cache,a.image)
