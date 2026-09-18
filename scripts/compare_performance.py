"""Compare matched C1 medians. A >20% sustained regression blocks promotion."""
import json
from pathlib import Path

def summaries(path):
    rows={}
    if not path.exists(): return rows
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except ValueError: continue
        if row.get('summary') in ('text','image') and row.get('full_1024_samples')==3:
            rows[row['summary']]=row
    return rows

def compare(before, after):
    rows=[]; failed=False
    for kind in ('text','image'):
        if kind not in before or kind not in after:
            rows.append({'kind':kind,'status':'NOT_COMPARABLE','reason':'missing three matched 1024-token samples'})
            continue
        old=before[kind]['median_client_tps']; new=after[kind]['median_client_tps']
        assert old>0 and new>0
        ratio=new/old
        rows.append({'kind':kind,'status':'PASS' if ratio>=.8 else 'REGRESSION',
            'before_client_tps':old,'after_client_tps':new,'ratio':ratio})
        failed |= ratio<.8
    return rows,failed

if __name__=='__main__':
    rows,failed=compare(summaries(Path('/report/performance-before.log')),summaries(Path('/report/performance-after.log')))
    print(json.dumps({'R34_C1_COMPARISON':rows,'regression_threshold':.8,
        'scope':'Same client/prompt and 3x1024 outputs. Not a C32 throughput benchmark or isolated kernel speedup.'}),flush=True)
    raise SystemExit(1 if failed else 0)
