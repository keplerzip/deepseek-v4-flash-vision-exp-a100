"""Fault injection for the real controller with simulated Docker/GPU phases."""
import base64, gzip, hashlib, json, os, pathlib, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:3a93108b2925da6d5498f2a7f7c442159b7a14fc0d6e64da3bbcc5b0a3a338e9'
R3IMAGE='sha256:64cc83da4129dbddb6d040ebc5a9a9038071257d82d62b831c3c782e1b097e49'
DOCKER=r'''#!/usr/bin/env python3
import sys,os,json,re,subprocess,importlib.util
from pathlib import Path
p=Path(os.environ['R34_TEST_STATE']);s=json.loads(p.read_text());args=sys.argv[1:]
s['calls'].append(args);p.write_text(json.dumps(s))
def save():p.write_text(json.dumps(s))
def done(text='',rc=0):
 if text:print(text)
 save();sys.exit(rc)
if args==['info']:done()
if args[0]=='container' and args[1]=='inspect':done('{}' if args[-1] in s['containers'] else '',0 if args[-1] in s['containers'] else 1)
if args[:2]==['image','inspect']:
 image=args[-1]
 if image not in (s['image'],s['tag']):done(rc=1)
 fmt=args[args.index('--format')+1] if '--format' in args else ''
 done(s['release'] if 'Labels' in fmt else s['image'])
if args[0]=='inspect':
 c=s['containers'].get(args[-1])
 if c is None:done(rc=1)
 f=args[args.index('--format')+1]
 if f=='{{.Id}}':done(c['id'])
 if f=='{{.Image}}':done(s['image'])
 if 'State.Running' in f:done('true' if c['running'] else 'false')
 if 'State.Status' in f:done('running' if c['running'] else 'exited')
 if 'Labels' in f:done(c['owner'])
 for dst in ('/deploy','/model','/runtime-cache'):
  if '"'+dst+'"' in f:done(c['mounts'][dst])
 done(rc=9)
if args[0]=='exec':done('HEALTH=PASS')
if args[0]=='stop':s['containers'][args[-1]]['running']=False;done()
if args[0]=='start':s['containers'][args[-1]]['running']=True;done()
if args[0]=='rename':s['containers'][args[2]]=s['containers'].pop(args[1]);done()
if args[0]=='logs':done()
if args[0]=='run':
 mounts={}
 for i,a in enumerate(args):
  if a=='--mount':
   kv=dict(part.split('=',1) for part in args[i+1].split(',') if '=' in part)
   mounts[kv['dst']]=kv['src']
 def mapped(a):
  for k,v in mounts.items():
   if a==k or a.startswith(k+'/'):return v+a[len(k):]
  return a
 if '/upgrade/installer.py' in args:
  a=args[args.index('/upgrade/installer.py'):]
  done(rc=subprocess.run([sys.executable]+[mapped(x) for x in a]).returncode)
 if '/deploy/scripts/verify_model.py' in args:done('{}',12 if s['scenario']=='precheck_fail' else 0)
 if '/deploy/tests/test_r34_runtime.py' in args:done('CPU=SIMULATED',13 if s['scenario']=='cpu_fail' else 0)
 if '/deploy/scripts/performance.py' in args:
  for k in ('text','image'):print(json.dumps({'summary':k,'full_1024_samples':3,'median_client_tps':100.}))
  done()
 if '/deploy/scripts/compare_performance.py' in args:
  spec=importlib.util.spec_from_file_location('cmp',mapped('/deploy/scripts/compare_performance.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  rows,failed=m.compare(m.summaries(Path(mapped('/report/performance-before.log'))),m.summaries(Path(mapped('/report/performance-after.log'))))
  done(json.dumps(rows),int(failed))
 done(rc=19)
done(rc=20)
'''
BASH=r'''#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
p=Path(os.environ['R34_TEST_STATE']);s=json.loads(p.read_text());role=Path(sys.argv[1]).name
if role not in ('gpu_kernel_test.sh','start.sh','verify.sh','performance_test.sh'):
 os.execv('/bin/bash',['bash']+sys.argv[1:])
s['phases'].append(role)
if role=='start.sh':
 s['containers']['dsv4-flash-r34']={'id':'new-id','owner':'deepseek-v4-flash-r3.4-20260916','running':True,'mounts':{'/deploy':str(Path(sys.argv[1]).parent),'/model':s['model'],'/runtime-cache':s['cache']}}
p.write_text(json.dumps(s))
if s['scenario']=={'gpu_kernel_test.sh':'gpu_fail','start.sh':'start_fail','verify.sh':'verify_fail'}.get(role):sys.exit(23)
if role=='performance_test.sh':
 for kind in ('text','image'):print(json.dumps({'summary':kind,'full_1024_samples':3,'median_client_tps':60. if s['scenario']=='regression' else 110.}))
print('PHASE_SIMULATED='+role)
'''

def case(scenario,old='r3',running=True,failed_candidate=False):
    with tempfile.TemporaryDirectory(prefix='r34-audit-') as d:
        d=Path(d);old_dir=d/'old';(old_dir/'scripts').mkdir(parents=True)
        (old_dir/'start.sh').write_text('# existing deployment\n');(old_dir/'scripts/engine.py').write_text('# existing engine\n')
        model=d/'model';model.mkdir();(model/'config.json').write_text('{}')
        cache=old_dir/'runtime/cache';cache.mkdir(parents=True)
        (old_dir/'deployment.env').write_text('MODEL_DIR='+str(model)+'\n')
        image=R3IMAGE if old in ('r3','absent') else IMAGE
        name='dsv4-vision-r3-scheme4' if old=='r3' else 'dsv4-flash-r33'
        containers={} if old=='absent' else {name:{'id':'old-id','running':running,'owner':'legacy-r3' if old=='r3' else 'deepseek-v4-flash-r3.3-20260914','mounts':{'/deploy':str(old_dir),'/model':str(model),'/runtime-cache':str(cache)}}}
        if failed_candidate:containers['dsv4-flash-r34']={'id':'failed-id','running':False,'owner':'deepseek-v4-flash-r3.4-20260916','mounts':{'/deploy':str(d/'failed'),'/model':str(model),'/runtime-cache':str(cache)}}
        state={'scenario':scenario,'image':image if scenario!='unknown_image' else 'sha256:unknown','release':'2026.09.10-r3-vision-rc1' if old in ('r3','absent') else '2026.09.14-r3.3',
            'tag':'dsv4-vision-a100:20260910-r3-sm80-rc1' if old in ('r3','absent') else 'dsv4-flash-a100:20260914-r3.3',
            'containers':containers,'calls':[],'phases':[],'model':str(model),'cache':str(cache)}
        statefile=d/'state.json';statefile.write_text(json.dumps(state));bin=d/'bin';bin.mkdir()
        for name2,source in [('docker',DOCKER),('bash',BASH)]:
            p=bin/name2;p.write_text(source);p.chmod(0o755)
        env=dict(os.environ,PATH=str(bin)+':'+os.environ['PATH'],R34_TEST_STATE=str(statefile))
        result=subprocess.run(['/bin/bash',str(ROOT/'upgrade/upgrade-r34.sh')],cwd=old_dir,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        state=json.loads(statefile.read_text())
        success=scenario=='success'
        assert (result.returncode==0)==success,result.stdout[-7000:]
        assert (old_dir/'start.sh').read_text()=='# existing deployment\n'
        if old!='absent':assert state['containers'][name]['running']==(False if success else running),result.stdout[-5000:]
        if success:assert state['containers']['dsv4-flash-r34']['running']
        elif 'dsv4-flash-r34' in state['containers']:assert not state['containers']['dsv4-flash-r34']['running']
        if scenario in ('precheck_fail','cpu_fail','unknown_image'):assert not any(a[0]=='stop' for a in state['calls'])
        if failed_candidate:assert any(a[0]=='rename' for a in state['calls'])
        print(json.dumps({'scenario':scenario,'old':old,'previous_running':running,'prior_failed_candidate':failed_candidate,'status':'PASS','phases':state['phases']}),flush=True)

for args in [('success','r3',True),('success','r3',False),('success','absent',False),('success','r33',True),('precheck_fail','r3',True),('cpu_fail','r3',True),('gpu_fail','r3',True),('start_fail','r3',True),('verify_fail','r33',True),('regression','r33',True),('unknown_image','r3',True)]:case(*args)
case('success',failed_candidate=True)
command=(ROOT/'upgrade/base64-r3.4.cmd').read_text()
encoded=command.split("<<'R34_BASE64_END' | gzip -dc > \"$r34_tmp\"\n",1)[1].split('\nR34_BASE64_END',1)[0]
decoded=gzip.decompress(base64.b64decode(encoded))
assert decoded==(ROOT/'upgrade/upgrade-r34.sh').read_bytes()
assert hashlib.sha256(decoded).hexdigest() in command
with tempfile.TemporaryDirectory(prefix='r34-b64-') as d:
    p=Path(d)/'broken.cmd';p.write_text(command[:len(command)//2])
    result=subprocess.run(['/bin/bash',str(p)],cwd=d,capture_output=True,timeout=10)
    assert result.returncode!=0
print('BASE64_DECODE_AND_TRUNCATION=PASS')
