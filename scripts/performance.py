"""Single-request sustained output measurement, with warmup and raw usage."""
import base64
import datetime
import json
from pathlib import Path
import statistics
import time
from cache_probe import request, NAMES

out = Path('/results') / ('performance-r33-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
prompt = '请详细介绍数据库事务、隔离级别、索引优化和故障恢复，逐项解释并举例，持续展开论述至少四千字。'
image = base64.b64encode(Path('/deploy/tests/multimodal/receipt.png').read_bytes()).decode()
for kind in ('text', 'image'):
    records = []
    for repeat in range(4):
        content = [{'type': 'text', 'text': prompt}]
        if kind == 'image':
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}})
        payload = {'model': NAMES[0], 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 64 if repeat == 0 else 1024, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}}
        print(json.dumps({'starting': kind, 'repeat': repeat}), flush=True)
        start = time.monotonic()
        data = request('/v1/chat/completions', payload)
        elapsed = time.monotonic() - start
        assert data['model'] == NAMES[0]
        count = data['usage']['completion_tokens']
        metrics = data.get('metrics') or {}
        duration = metrics.get('generation_time_ms')
        record = {'kind': kind, 'phase': 'warmup' if repeat == 0 else 'measure', 'repeat': repeat,
                  'prompt_tokens': data['usage']['prompt_tokens'], 'output_tokens': count,
                  'client_total_s': elapsed, 'client_end_to_end_tps': count / elapsed,
                  'decode_tps_after_first': (count - 1) * 1000 / duration if duration and count > 1 else None,
                  'server_metrics': metrics, 'usage': data['usage'], 'finish_reason': data['choices'][0]['finish_reason'],
                  'sample_note': '1024 tokens' if count == 1024 else 'short sample; EOS is respected'}
        with out.open('a') as f: f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if repeat: records.append(record)
    full = [r for r in records if r['output_tokens'] == 1024]
    rates = [r['decode_tps_after_first'] for r in full if r['decode_tps_after_first'] is not None]
    print(json.dumps({'summary': kind, 'full_1024_samples': len(full), 'median_client_tps': statistics.median(r['client_end_to_end_tps'] for r in full) if full else None,
                      'median_decode_tps': statistics.median(rates) if rates else None, 'result_file': str(out),
                      'scope': 'Single request; not C32 throughput or guaranteed production speed.'}), flush=True)
