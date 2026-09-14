"""Five/eight ordered images over native JSON/SSE APIs, using existing fixtures."""
import base64
import json
from pathlib import Path
import time
import cache_probe as p

def body_for(path, count, stream):
    sequence = [1, 3, 2, 1, 2, 3, 3, 1][:count]
    prompt = 'Read the number at the top of EACH image in the supplied order. Return only a JSON object with one field numbers, an array of integers. Preserve repetitions. No Markdown.'
    content = [{'type': 'input_text' if path.endswith('responses') else 'text', 'text': prompt}]
    for number in sequence:
        data = base64.b64encode((Path('/deploy/tests/multimodal/video_frames')/f'{number:02}.png').read_bytes()).decode()
        if path.endswith('messages'):
            image = {'type':'image','source':{'type':'base64','media_type':'image/png','data':data}}
        elif path.endswith('responses'):
            image = {'type':'input_image','image_url':'data:image/png;base64,'+data,'detail':'auto'}
        else:
            image = {'type':'image_url','image_url':{'url':'data:image/png;base64,'+data}}
        content.append(image)
    body = {'model':'DeepSeek-V4-Flash','stream':stream,'temperature':0,'chat_template_kwargs':{'thinking':False}}
    body['input' if path.endswith('responses') else 'messages'] = [{'role':'user','content':content}]
    body['max_output_tokens' if path.endswith('responses') else 'max_tokens'] = 1024
    return body, {'numbers':sequence}

def main():
    failed = []
    for path in ('/v1/chat/completions','/v1/responses','/v1/messages'):
        for count, stream in ((5,False),(8,True)):
            started = time.monotonic()
            row = {'id':path, 'image_count':count, 'stream':stream}
            try:
                body, expected = body_for(path,count,stream)
                actual, usage, total, cached = p.normalize(path,body,p.request(path,body))
                assert actual == expected and all(type(x) is int for x in actual['numbers']), (actual,expected)
                row.update(status='PASS', actual=actual, raw_usage=usage)
            except Exception as exc:
                failed.append([path,count])
                row.update(status='FAIL',error=str(exc)[:3000])
            row['seconds'] = round(time.monotonic()-started,3)
            p.emit(row)
    p.emit({'MULTI_IMAGE_TEST':'FAIL' if failed else 'PASS','failed':failed,'configured_limit':999,'tested_counts':[5,8],'all_999_images_tested':False})
    return bool(failed)

if __name__=='__main__':
    raise SystemExit(main())
