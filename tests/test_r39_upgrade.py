"""Run the real installer against a simulated Docker API; no GPU claims."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:64cc83da4129dbddb6d040ebc5a9a9038071257d82d62b831c3c782e1b097e49'
DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p = Path(os.environ['R39_FAKE_STATE'])
s = json.loads(p.read_text()); a = sys.argv[1:]
def end(value='', code=0):
    p.write_text(json.dumps(s))
    if value: print(value)
    sys.exit(code)
if s.get('sudo_only') and os.environ.get('R39_FAKE_SUDO') != '1':
    s['denied'] = s.get('denied', 0) + 1
    end('permission denied', 77)
s['commands'].append(a)
if a == ['info']: end()
if a[:2] == ['container', 'inspect']:
    end(code=0 if a[2] in s['containers'] else 1)
if a[0] == 'inspect':
    name = a[-1]
    if name not in s['containers']: end(code=1)
    c=s['containers'][name]; f=a[a.index('--format')+1]
    if 'com.deepseek.owner' in f: end(c['owner'])
    if '.Mounts' in f:
        dest=next(x for x in ('/deploy','/model','/runtime-cache') if x in f)
        end(c['mounts'].get(dest,''))
    if f == '{{.Id}}': end(c['id'])
    if f == '{{.Image}}': end(s['image'])
    if f == '{{.State.Running}}': end('true' if c['running'] else 'false')
    if '.State.Status' in f: end(('running' if c['running'] else 'exited')+' 0')
if a[:2] == ['image','inspect']:
    if '--format' in a:
        f=a[a.index('--format')+1]
        end(s['image'] if '.Id' in f else s['image_release'])
    end('[]')
if a[:2] == ['network','inspect']: end('172.17.0.1')
if a[0] == 'run':
    joined=' '.join(a)
    if '--name' in a:
        name=a[a.index('--name')+1]
        owner=next(x.split('=',1)[1] for x in a if x.startswith('com.deepseek.owner='))
        s['containers'][name]={'id':'candidate-id','owner':owner,'running':s['scenario']!='startup_fail',
            'mounts':{'/deploy':s['candidate'],'/model':s['model'],'/runtime-cache':s['cache']}}
        end('candidate-id')
    if 'pytest' in a:
        if s['scenario']=='state_changed': s['containers']['dsv4-flash-r38']['running']=False
        end('SIMULATED_CPU_CHECK', code=5 if s['scenario']=='cpu_fail' else 0)
    if '--query-gpu=name,memory.total,driver_version' in a:
        end('\n'.join(['NVIDIA A100-SXM4-80GB, 81920, 580.126.20']*8))
    end()
if a[0]=='exec':
    if 'simulated-verify' in a:
        s['verify_calls']=s.get('verify_calls',0)+1
        end('SIMULATED_VERIFY',code=8 if s['scenario']=='verify_fail' else 0)
    end('HEALTH=PASS')
if a[0]=='logs': end('simulated log')
if a[0] in ('stop','start','rm'):
    name=a[-1]
    if a[0]=='rm': s['containers'].pop(name,None)
    else: s['containers'][name]['running']=(a[0]=='start')
    end(name)
end('Unsupported simulated Docker command: '+str(a),91)
'''


def manifest(root):
    entries = []
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root).as_posix()
        if path.is_file() and not relative.startswith('runtime/') and relative not in ('FILES.sha256', 'deployment.env'):
            entries.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+relative+'\n')
    (root/'FILES.sha256').write_text(''.join(entries))


class InstallerFailureTests(unittest.TestCase):
    def scenario(self, name, old_running=True, sudo_only=False, corrupt=False, bad_image=False, already=False):
        with tempfile.TemporaryDirectory(prefix='r39-controller-test-') as tmp:
            root=Path(tmp); new=root/'r39'; old=root/'r38'; model=root/'model'; cache=old/'runtime/cache'
            shutil.copytree(ROOT,new,ignore=shutil.ignore_patterns('runtime','__pycache__','deployment.env','.pytest_cache'))
            (old/'runtime/control').mkdir(parents=True); cache.mkdir(); model.mkdir()
            # Keep real install/start/preflight and image/ownership checks. Only
            # verification workload is substituted; model inference is unavailable.
            (new/'verify.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource "$(dirname -- "$0")/scripts/lib.sh"\ncheck_image\ndc exec "$CONTAINER" simulated-verify\n')
            manifest(new)
            if corrupt: (new/'VERSION').write_text('corrupted')
            bin_dir=root/'bin'; bin_dir.mkdir()
            (bin_dir/'docker').write_text(DOCKER)
            (bin_dir/'sudo').write_text('#!/usr/bin/env bash\n[[ $1 != -n ]] || shift\nexport R39_FAKE_SUDO=1\nexec "$@"\n')
            (bin_dir/'ss').write_text('#!/usr/bin/env bash\nexit 0\n')
            for path in bin_dir.iterdir(): path.chmod(0o755)
            old_state={'id':'previous-id','owner':'deepseek-v4-flash-r3.8-20260923','running':old_running,
                       'mounts':{'/deploy':str(old),'/model':str(model),'/runtime-cache':str(cache)}}
            state={'scenario':name,'candidate':str(new),'model':str(model),'cache':str(cache),
                   'containers':{'dsv4-flash-r38':old_state},'commands':[], 'sudo_only':sudo_only,
                   'image':IMAGE,'image_release':'wrong' if bad_image else '2026.09.10-r3-vision-rc1'}
            if already:
                state['containers']['dsv4-flash-r39']={'id':'candidate-id','owner':'deepseek-v4-flash-r3.9-20260929','running':True,'mounts':{'/deploy':str(new)}}
                (new/'runtime').mkdir(); (new/'runtime/base-image-id').write_text(IMAGE+'\n')
                (new/'deployment.env').write_text('MODEL_DIR='+str(model)+'\n')
            state_path=root/'state.json'; state_path.write_text(json.dumps(state))
            env=dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], R39_FAKE_STATE=str(state_path))
            result=subprocess.run(['bash',str(new/'install.sh')],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            result_state=json.loads(state_path.read_text())
            return result.returncode,result.stdout,result_state

    def assert_unchanged(self, result, running=True):
        rc, log, state=result
        self.assertNotEqual(rc,0,log)
        self.assertTrue(state['containers']['dsv4-flash-r38']['running']==running,log)
        self.assertFalse(any(a[0] in ('start','stop','rm') for a in state['commands']),log)

    def test_sudo_only_original_image_reused_successfully(self):
        rc,log,state=self.scenario('success',sudo_only=True)
        self.assertEqual(rc,0,log); self.assertIn('R39_INSTALL=PASS',log)
        self.assertTrue(state['denied']>0)
        self.assertFalse(state['containers']['dsv4-flash-r38']['running'])
        self.assertTrue(state['containers']['dsv4-flash-r39']['running'])
        self.assertEqual(state['verify_calls'],1)
        self.assertFalse(any(a[0] in ('tag','pull','build') for a in state['commands']))

    def test_stopped_previous_does_not_need_starting_first(self):
        rc,log,state=self.scenario('success',old_running=False)
        self.assertEqual(rc,0,log)
        self.assertFalse(any(a[0]=='start' for a in state['commands']))

    def test_checksum_failure_does_not_stop_previous(self):
        self.assert_unchanged(self.scenario('success',corrupt=True))

    def test_unknown_image_release_does_not_stop_previous(self):
        self.assert_unchanged(self.scenario('success',bad_image=True))

    def test_cpu_failure_does_not_stop_previous(self):
        self.assert_unchanged(self.scenario('cpu_fail'))

    def test_startup_failure_restores_running_previous(self):
        rc,log,state=self.scenario('startup_fail')
        self.assertNotEqual(rc,0,log); self.assertIn('RESTORE_RC=0',log)
        self.assertTrue(state['containers']['dsv4-flash-r38']['running'])
        self.assertFalse(state['containers']['dsv4-flash-r39']['running'])

    def test_verification_failure_restores_running_previous(self):
        rc,log,state=self.scenario('verify_fail')
        self.assertNotEqual(rc,0,log); self.assertIn('RESTORE_RC=0',log)
        self.assertTrue(state['containers']['dsv4-flash-r38']['running'])
        self.assertFalse(state['containers']['dsv4-flash-r39']['running'])

    def test_verification_failure_preserves_previous_stopped_state(self):
        rc,log,state=self.scenario('verify_fail',old_running=False)
        self.assertNotEqual(rc,0,log); self.assertIn('RESTORE_RC=0',log)
        self.assertFalse(state['containers']['dsv4-flash-r38']['running'])
        self.assertFalse(state['containers']['dsv4-flash-r39']['running'])
        self.assertFalse(any(a[0]=='start' for a in state['commands']))

    def test_running_unverified_candidate_cannot_skip_verification(self):
        rc,log,state=self.scenario('verify_fail',already=True)
        self.assertNotEqual(rc,0,log)
        self.assertNotIn('ALREADY_CURRENT',log); self.assertEqual(state['verify_calls'],1)

    def test_external_state_change_blocks_service_switch(self):
        result=self.scenario('state_changed')
        self.assert_unchanged(result,running=False)
        self.assertIn('state changed',result[1])


if __name__=='__main__': unittest.main(verbosity=2)
