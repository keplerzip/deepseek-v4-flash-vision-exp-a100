"""Real SM80 mHC reconstruction and graph replay; no model weights needed."""
import pytest
import torch

@pytest.mark.parametrize('tokens',[1,8,64,256])
def test_fused_previous_aux_matches_standalone_post(tokens):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_fused_post_pre_tilelang
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0)
    h,hc=4096,4
    torch.manual_seed(3406)
    x=torch.randn(tokens,h,device='cuda',dtype=torch.bfloat16)
    residual=torch.randn(tokens,hc,h,device='cuda',dtype=torch.bfloat16)
    post=torch.rand(tokens,hc,1,device='cuda',dtype=torch.float32)
    mix=torch.softmax(torch.randn(tokens,hc,hc,device='cuda'),dim=-1)
    fn=torch.randn(hc*(2+hc),hc*h,device='cuda',dtype=torch.float32)*.01
    scale=torch.ones(3,device='cuda')
    base=torch.zeros(hc*(2+hc),device='cuda')
    norm=torch.ones(h,device='cuda',dtype=torch.bfloat16)
    def fused():
        return mhc_fused_post_pre_tilelang(x,residual,post,mix,fn,scale,base,
            1e-6,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-6)[0].mean(1)
    expected=mhc_post_tilelang(x,residual,post,mix).mean(1)
    actual=fused()
    torch.testing.assert_close(actual,expected,atol=1e-2,rtol=1e-2)
    # Warm JITs outside capture, then verify that captured values update.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured=fused()
    x.add_(.125)
    graph.replay()
    torch.cuda.synchronize()
    expected=mhc_post_tilelang(x,residual,post,mix).mean(1)
    torch.testing.assert_close(captured,expected,atol=1e-2,rtol=1e-2)
