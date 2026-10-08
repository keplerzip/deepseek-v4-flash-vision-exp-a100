"""CPU regressions for the exact overlay functions; no GPU import emulation.

AST extraction isolates functions from CUDA-only module imports. Kernel numeric
validation and real HTTP checks are separate target-machine gates.
"""
import ast
from collections.abc import Iterable
from itertools import islice
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace as NS

import pytest
import torch
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call_output_item import ResponseFunctionToolCallOutputItem
from vllm.exceptions import VLLMValidationError, VLLMClientError

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT/'source/overlay/vllm'
MODEL = 'models/deepseek_v4/nvidia/model.py'
DSPARK = 'models/deepseek_v4/nvidia/dspark.py'


def fn(file, name, cls=None, **extra):
    tree=ast.parse((OVERLAY/file).read_text())
    scope=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else tree.body
    node=next(n for n in scope if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    env=dict(torch=torch, re=re, Iterable=Iterable, islice=islice,
             _CONTEXT_WKV_RE=re.compile(r'^mtp\.(\d+)\.attn\.wkv\.(.+)$'))
    env.update(extra)
    isolated=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated),str(OVERLAY/file),'exec'),env)
    return env[name]


def test_weight_and_scale_shards_keep_original_storage():
    duplicate=fn(DSPARK,'_duplicate_context_wkv_weights')
    w=torch.randn(3,5)
    values=list(duplicate([('mtp.0.attn.wkv.weight',w),('mtp.2.attn.wkv.scale',w),
        ('mtp.1.attn.wq_a.weight',w),('mtp.3.attn.wkv.weight',w)],3))
    assert [v[0] for v in values]==['mtp.0.attn.wkv.weight','context_wkv_proj.weight',
        'mtp.2.attn.wkv.scale','context_wkv_proj.scale','mtp.1.attn.wq_a.weight','mtp.3.attn.wkv.weight']
    assert [values[i][1].shard_id for i in (1,3)]==[0,2]
    assert values[1][1].data_ptr()==w.data_ptr() and not hasattr(w,'shard_id')


@pytest.mark.parametrize('slots',[None, [torch.tensor([0,1]),None,torch.tensor([4,5])]])
def test_stacked_projection_keeps_layer_slots_and_normalization(slots):
    torch.manual_seed(34)
    x=torch.randn(2,5); weights=torch.randn(3,4,5); stored=[];calls=[]
    def projection(v):
        calls.append(1)
        return v @ weights.reshape(12,5).T
    layers=[NS(attn=NS(kv_norm=lambda v,i=i:v/(i+1))) for i in range(3)]
    model=NS(_r34_context_validated=True,context_wkv_proj=projection,num_dspark_layers=3,
        config=NS(head_dim=4),layers=layers)
    insert=lambda attn,kv,pos,mapping:stored.append((attn,kv.clone(),mapping))
    run=fn(DSPARK,'precompute_and_store_context_kv','DSparkDeepseekV4Model',_insert_context_kv=insert)
    run(model,x,torch.arange(2),slots)
    assert len(calls)==1
    if slots is None: assert not stored
    else:
        assert len(stored)==2
        for record,i in zip(stored,(0,2)):
            torch.testing.assert_close(record[1],(x@weights[i].T)/(i+1))
            assert record[2] is slots[i]


@pytest.mark.parametrize('capture',[False,True])
def test_aux_captured_from_first_post_and_int8_scales_preserved(capture):
    scale_in=object(); scale_out=object(); seen=[]
    def fused(x,res,post,mix,*args,**kwargs):
        seen.append(kwargs['x_scales'])
        return res+1,post,mix,x
    norm=NS(weight=NS(data=torch.ones(8)),variance_epsilon=1e-6)
    layer=NS(attn_norm=norm,ffn_norm=norm,attn=lambda pos,x,kv:x,ffn=lambda x,ids:x,
        _hoisted_all_reduce=lambda x:(x,scale_out),rms_norm_eps=1e-6,hc_eps=1e-6,
        hc_post_alpha=2.,hc_sinkhorn_iters=2)
    for field in ('hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base'):
        setattr(layer,field,None)
    residual=torch.randn(4,2,8); captured=[] if capture else None
    run=fn(MODEL,'forward','DeepseekV4DecoderLayer',mhc_fused_post_pre_tilelang=fused)
    out=run(layer,torch.randn(4,8),torch.arange(4),None,None,None,residual,scale_in,previous_aux=captured)
    assert len(out)==5 and out[-1] is scale_out and seen==[scale_in,scale_out]
    if capture: torch.testing.assert_close(captured[0],(residual+1).mean(1))


@pytest.mark.parametrize('aux_ids',[(1,3,5),(2,4),()])
def test_aux_layer_order_final_reconstruction_and_mtp_copy(aux_ids):
    def post(x,res,p,m,**kw): return res+x[:,None,:]*p
    def layer(x,pos,ids,p=None,m=None,res=None,scale=None,previous_aux=None):
        res=x[:,None,:].repeat(1,2,1) if res is None else post(x,res,p,m)
        if previous_aux is not None: previous_aux.append(res.mean(1))
        return res.mean(1)*.1,res,.5,None,None
    x=torch.arange(12,dtype=torch.float32).reshape(3,4)
    model=NS(use_mega_moe=False,embed_input_ids=lambda ids:x,layers=[layer]*5,start_layer=0,end_layer=5,
        aux_hidden_state_layers=aux_ids,_mtp_hidden_buffer=torch.full((3,8),-1.),hc_head_fn=None,
        hc_head_scale=None,hc_head_base=None,rms_norm_eps=1e-6,hc_eps=1e-6,norm=lambda v:v)
    run=fn(MODEL,'forward','DeepseekV4Model',get_pp_group=lambda:NS(is_first_rank=True,is_last_rank=True),
        mhc_post_tilelang=post,hc_head_fused_kernel_tilelang=lambda v,*args:v.mean(1))
    out=run(model,torch.arange(3),torch.arange(3),None)
    h=x;r=p=m=scale=None;expected=[]
    for idx in range(5):
        h,r,p,m,scale=layer(h,None,None,p,m,r,scale)
        if idx+1 in aux_ids: expected.append(post(h,r,p,m).mean(1))
    final=post(h,r,p,m)
    if aux_ids:
        torch.testing.assert_close(out[0],final.mean(1))
        assert len(out[1])==len(expected)
        for a,b in zip(out[1],expected): torch.testing.assert_close(a,b)
        assert torch.all(model._mtp_hidden_buffer==-1)
    else:
        torch.testing.assert_close(out,final.mean(1))
        torch.testing.assert_close(model._mtp_hidden_buffer,final.flatten(1))


def responses_fn():
    from vllm.entrypoints.openai.responses.utils import _construct_message_from_response_item
    return _construct_message_from_response_item


@pytest.mark.parametrize('item',[{'type':'item_reference','id':'unknown'},NS(type='computer_call'),{}, {'type':'message','role':'invalid'}])
def test_unsupported_response_item_is_typed_client_error(item):
    with pytest.raises(VLLMClientError) as info: responses_fn()(item)
    assert isinstance(info.value,VLLMValidationError) and info.value.parameter=='input'


@pytest.mark.parametrize('parts',[[],['a'],['a','b']])
def test_response_replay_empty_and_multiple_text_parts(parts):
    item=ResponseOutputMessage(id='msg_1',role='assistant',status='completed',type='message',
        content=[ResponseOutputText(type='output_text',text=s,annotations=[]) for s in parts])
    assert responses_fn()(item)=={'role':'assistant','content':''.join(parts)}


def test_response_function_call_roundtrip_is_preserved():
    call=ResponseFunctionToolCall(type='function_call',call_id='call_1',name='record_port',arguments='{"port":8006}',id='fc_1')
    message=responses_fn()(call)
    assert message['tool_calls'][0]['function']['arguments']=='{"port":8006}'
    result=responses_fn()({'type':'function_call_output','call_id':'call_1','output':'{"ok":true}'})
    assert result=={'role':'tool','tool_call_id':'call_1','content':'{"ok":true}'}


def test_fixed_contract_and_overlay_parse():
    cfg=json.loads((ROOT/'config/service.json').read_text())
    assert cfg['served_model_name']=='DeepSeek-V4-Flash' and cfg['limit_mm_per_prompt']=={'image':999}
    assert cfg['speculative_config']['num_speculative_tokens']==6 and cfg['tensor_parallel_size']==8
    assert (cfg['max_model_len'],cfg['max_num_seqs'],cfg['gpu_memory_utilization'])==(1048576,32,.92)
    for path in OVERLAY.rglob('*.py'): ast.parse(path.read_text(),filename=str(path))
