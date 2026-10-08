"""SM80 numeric and CUDA graph replay gates for DSpark's 2D mHC input."""
import pytest
import torch


@pytest.mark.parametrize('tokens',[1,6,32,224])
def test_mhc_broadcast_matches_repeated_embedding_and_replay(tokens):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang, mhc_pre_broadcast_tilelang
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0)
    h,hc=4096,4
    torch.manual_seed(3506)
    x=torch.randn(tokens,h,device='cuda',dtype=torch.bfloat16)
    fn=torch.randn(hc*(2+hc),hc*h,device='cuda',dtype=torch.float32)*.01
    scale=torch.ones(3,device='cuda')
    base=torch.zeros(hc*(2+hc),device='cuda')
    norm=torch.ones(h,device='cuda',dtype=torch.bfloat16)
    broadcast=fn.view(-1,hc,h).sum(1)
    def actual():
        return mhc_pre_broadcast_tilelang(x,fn,scale,base,1e-6,1e-6,1e-6,2.,20,
            norm_weight=norm,norm_eps=1e-6,fn_broadcast=broadcast)
    def reference():
        expanded=x[:,None,:].repeat(1,hc,1)
        return (expanded,*mhc_pre_tilelang(expanded,fn,scale,base,1e-6,1e-6,1e-6,2.,20,
            norm_weight=norm,norm_eps=1e-6))
    def compare(out):
        expected=reference()
        assert len(out)==len(expected)==4
        for a,b in zip(out,expected):torch.testing.assert_close(a,b,rtol=.02,atol=.02)
    compare(actual())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):actual()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=actual()
    for step in range(3):
        x.copy_(torch.randn_like(x)*(step+1))
        graph.replay();torch.cuda.synchronize();compare(out)
