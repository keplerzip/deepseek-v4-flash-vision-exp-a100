"""Ten independent C32 rounds; no retry or downgraded pass criterion."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import threading
import time
import uuid
import cache_probe as p

DEST = Path('/results') / ('concurrency-r33-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
PATHS = ['/v1/chat/completions', '/v1/responses', '/v1/messages']

def make_case(round_id, n):
    path = PATHS[0] if round_id < 5 else PATHS[n % 3]
    model = p.NAMES[0] if path.endswith('messages') else p.NAMES[0]
    # Include text + image requests, both image values, JSON and streaming.
    picture = (['chart_before.png', 'chart_after.png'][n % 2]
               if round_id < 5 or n % 4 != 0 else None)
    expected = {'b_value': 65 if picture == 'chart_before.png' else 60} if picture else {'check': 73019}
    body = p.payload_for(path, model, uuid.uuid4().hex, picture, repeats=8)
    body['stream'] = round_id >= 5 and n % 2 == 1
    if body['stream'] and path.endswith('chat/completions'):
        body['stream_options'] = {'include_usage': True}
    return path, body, expected

def run_round(round_id):
    barrier = threading.Barrier(32)
    def one(n):
        path, body, expected = make_case(round_id, n)
        rec = {'round': round_id + 1, 'request': n, 'path': path, 'model': body['model'],
               'stream': body['stream'], 'expected': expected}
        started = time.monotonic()
        try:
            barrier.wait(timeout=60)
            raw = p.request(path, body)
            actual, usage, total, cached = p.normalize(path, body, raw)
            rec.update(actual=actual, usage=usage)
            assert actual == expected and all(type(actual[k]) is type(v) for k, v in expected.items()), 'Answer mismatch'
            rec['status'] = 'PASS'
        except Exception as exc:
            rec.update(status='FAIL', error_type=type(exc).__name__, error=str(exc)[:1800])
        rec['seconds'] = round(time.monotonic() - started, 3)
        return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        return list(pool.map(one, range(32)))

def main():
    failures = 0
    for i in range(10):
        records = run_round(i)
        failed = [r for r in records if r['status'] != 'PASS']
        failures += len(failed)
        with DEST.open('a') as f:
            for r in records:
                line = json.dumps(r, ensure_ascii=False)
                if p.KEY:
                    line = line.replace(p.KEY, '[REDACTED]')
                f.write(line + '\n')
        print(json.dumps({'id': 'r33-concurrency-32', 'round': i + 1, 'status': 'FAIL' if failed else 'PASS',
                          'passed': 32 - len(failed), 'failed': len(failed),
                          'mix': 'images-chat-json' if i < 5 else 'text-image-three-protocols-json-sse'}), flush=True)
        for r in failed:
            line = json.dumps(r, ensure_ascii=False)
            print(line.replace(p.KEY, '[REDACTED]') if p.KEY else line, flush=True)
    print(json.dumps({'C32_R33': 'FAIL' if failures else 'PASS', 'requests': 320, 'failed': failures,
                      'retries': 0, 'result_file': str(DEST), 'full_window_c32_executed': False}), flush=True)
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
