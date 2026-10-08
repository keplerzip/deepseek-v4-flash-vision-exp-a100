"""R3.5 CPU protocol regressions using the real frozen runtime with overlays."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError
from openai.types.responses import CustomTool
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.deepseek_v4_encoding import flatten_content_blocks
from vllm.tool_parsers.utils import collect_tool_names, find_tool_properties, get_json_schema_from_tools


@pytest.mark.parametrize('detail',['MISSING','auto','low','high',None,'invalid'])
def test_responses_image_detail_default_preserves_explicit_validation(detail):
    part={'type':'input_image','image_url':'data:image/png;base64,AAAA'}
    if detail!='MISSING': part['detail']=detail
    request={'model':'DeepSeek-V4-Flash','input':[{'role':'user','content':[part]}]}
    before=copy.deepcopy(request)
    if detail in (None,'invalid'):
        with pytest.raises(ValidationError): ResponsesRequest(**request)
    else:
        actual=ResponsesRequest(**request)
        assert actual.input[0]['content'][0]['detail']==('auto' if detail=='MISSING' else detail)
    assert request==before


@pytest.mark.parametrize('choice',['auto','none','required',{'type':'custom','name':'apply_patch'}])
def test_custom_tools_reach_template_parser_and_forced_schema(choice):
    req=ResponsesRequest(model='DeepSeek-V4-Flash',input='probe',tools=[{'type':'custom','name':'apply_patch'}],tool_choice=choice)
    converted=construct_tool_dicts(req.tools,req.tool_choice)
    assert converted[0]['function']['name']=='apply_patch'
    assert converted[0]['function']['parameters']['properties']['input']['type']=='string'
    assert collect_tool_names(req.tools)=={'apply_patch'}
    assert find_tool_properties(req.tools,'apply_patch')['input']['type']=='string'
    if choice not in ('auto','none'):
        assert 'input' in json.dumps(get_json_schema_from_tools(req.tool_choice,req.tools))


def test_flatten_interleaved_images_preserves_boundaries():
    blocks=[{'type':'text','text':'A'}, {'type':'image_url','image_url':{'url':'a'}},
            {'type':'text','text':'B'}, {'type':'image_url','image_url':{'url':'b'}}, {'type':'text','text':'C'}]
    assert flatten_content_blocks(blocks)=='A\n\n<｜deepseek_image｜>\n\nB\n\n<｜deepseek_image｜>\n\nC'
    assert flatten_content_blocks('A\nB')=='A\nB'


class FakeTokenizer:
    def get_added_vocab(self): return {}
    def get_vocab(self): return {'<think>':50,'</think>':51}
    def encode(self,text,**kwargs): return [ord(c) for c in text]


def test_system_tools_attach_without_mutating_messages():
    tokenizer=get_deepseek_v4_tokenizer(FakeTokenizer())
    messages=[{'role':'system','content':'SYSTEM_SENTINEL'}, {'role':'user','content':'hello'}]
    tools=[{'type':'function','function':{'name':'probe','parameters':{'type':'object','properties':{}}}}]
    before=copy.deepcopy(messages)
    prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False)
    assert 'SYSTEM_SENTINEL' in prompt and 'probe' in prompt and messages==before
    assert tokenizer.supports_inline_system_messages is True


@pytest.mark.parametrize('preserve',[False,True])
def test_anthropic_trailing_system_kept_only_for_capable_tokenizer(preserve):
    request=AnthropicMessagesRequest(model='DeepSeek-V4-Flash',max_tokens=32,system='stable',
        messages=[{'role':'user','content':'hello'}, {'role':'assistant','content':'hi'},
                  {'role':'system','content':'changing reminder'}])
    converted=AnthropicServingMessages._convert_anthropic_to_openai_request(
        request,merge_inline_system=False,preserve_trailing_system=preserve)
    if preserve:
        assert [m['role'] for m in converted.messages]==['system','user','assistant','system']
        assert converted.messages[0]['content']=='stable'
        tokenizer=get_deepseek_v4_tokenizer(FakeTokenizer())
        prompt=tokenizer.apply_chat_template(converted.messages,tokenize=False)
        assert prompt.endswith('<｜Assistant｜></think>')
    else:
        assert [m['role'] for m in converted.messages]==['system','user','assistant']
        assert 'changing reminder' in converted.messages[0]['content']
    assert AnthropicServingMessages._detect_merge_inline_system(None,NS(supports_inline_system_messages=preserve)) is not preserve


@pytest.mark.parametrize('chunk_size',[1,2,7,32,128])
@pytest.mark.parametrize('body',['hello '*600, '中文😀\\line\n"quoted" <literal> end '*30])
def test_tool_string_streams_before_closer_and_preserves_exact_json(chunk_size,body):
    tool={'type':'function','function':{'name':'emit_text','parameters':{'type':'object',
          'properties':{'text':{'type':'string'}},'required':['text']}}}
    request=ChatCompletionRequest(model='DeepSeek-V4-Flash',messages=[],tools=[tool])
    parser=DeepSeekV4Parser(FakeTokenizer(),tools=request.tools)
    header='<｜DSML｜tool_calls>\n<｜DSML｜invoke name="emit_text">\n<｜DSML｜parameter name="text" string="true">'
    tail='</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    chunks=[header]+[body[i:i+chunk_size] for i in range(0,len(body),chunk_size)]+list(tail)
    previous='';args=[];preclose=0
    for i,chunk in enumerate(chunks):
        current=previous+chunk
        delta=parser.extract_tool_calls_streaming(previous_text=previous,current_text=current,delta_text=chunk,
            previous_token_ids=[],current_token_ids=[],delta_token_ids=[1],request=request)
        previous=current
        for call in (delta.tool_calls or []) if delta else []:
            if call.function and call.function.arguments:
                args.append(call.function.arguments)
                preclose += i<1+(len(body)+chunk_size-1)//chunk_size
    assert preclose>2
    assert json.loads(''.join(args))=={'text':body}


def test_broadcast_refinalization_preserves_storage_and_new_weights():
    from test_r34_runtime import fn, MODEL
    layer=NS(hc_mult=4,hidden_size=8,hc_attn_fn=torch.randn(24,32),hc_attn_fn_broadcast=None)
    finalize=fn(MODEL,'finalize_mhc_broadcast_weight','DeepseekV4DecoderLayer')
    finalize(layer);buffer=layer.hc_attn_fn_broadcast
    layer.hc_attn_fn.add_(1);finalize(layer)
    assert layer.hc_attn_fn_broadcast is buffer
    torch.testing.assert_close(buffer,layer.hc_attn_fn.reshape(24,4,8).sum(1))


@pytest.mark.parametrize('custom',[False,True])
def test_final_response_reuses_streamed_tool_identity(custom):
    from unittest.mock import AsyncMock
    from openai.types.responses import ResponseCustomToolCall, ResponseFunctionToolCall, ResponseOutputItemDoneEvent
    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
    from vllm.sampling_params import SamplingParams
    req=ResponsesRequest(model='DeepSeek-V4-Flash',input='probe')
    sampling=SamplingParams(max_tokens=32)
    cls=ResponseCustomToolCall if custom else ResponseFunctionToolCall
    payload={'input':'payload'} if custom else {'arguments':'{"port":8006}'}
    item=cls(type='custom_tool_call' if custom else 'function_call',id='stream_item',call_id='stream_call',name='probe',**payload)
    final_item=item.model_copy(update={'id':'reparsed_item','call_id':'reparsed_call'})
    final=ResponsesResponse.from_request(req,sampling,model_name=req.model,created_time=1,output=[final_item],status='completed',usage=None)
    service=object.__new__(OpenAIServingResponses)
    service.use_harmony=False
    async def events(*args,**kwargs):
        yield ResponseOutputItemDoneEvent(type='response.output_item.done',sequence_number=1,output_index=0,item=item)
    service._process_simple_streaming_events=events
    service.responses_full_generator=AsyncMock(return_value=final)
    async def run():
        return [event async for event in service.responses_stream_generator(req,sampling,None,None,req.model,None,None,created_time=1)]
    result=asyncio.run(run())
    completed=result[-1].response
    assert completed.output[0].id==item.id and completed.output[0].call_id==item.call_id
    assert completed.output[0].model_dump()==item.model_dump()
