"""SM80 greedy selection parity and changed-input graph replay; no weights."""
import pytest
import torch

from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax


@pytest.fixture(scope="module", autouse=True)
def require_sm80():
    assert torch.cuda.is_available(), "R3.7 argmax gate requires a CUDA GPU"
    assert torch.cuda.get_device_capability() == (8, 0), "R3.7 targets SM80"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("vocab", [4095, 4096, 4097, 129280])
def test_edges_and_changed_graph(dtype, vocab):
    logits = torch.randn(8, vocab + 19, device="cuda", dtype=dtype)[:, 1:vocab + 1]
    logits[0].zero_()
    logits[1].fill_(float("-inf"))
    logits[2, 5] = logits[2, -1] = float("inf")
    logits[3, 5] = logits[3, -1] = float("nan")
    logits[4].fill_(float("nan"))
    logits[5].fill_(float("-inf"))
    logits[5, -1] = 1
    logits[6].zero_()
    logits[6, -1] = 1
    if dtype == torch.float64:
        logits[7].fill_(1)
        logits[7, -1] = 1 + 2**-30
    def check(result):
        assert result.dtype == torch.int64
        torch.testing.assert_close(result, logits.argmax(-1), rtol=0, atol=0)
    check(greedy_argmax(logits))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            greedy_argmax(logits)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = greedy_argmax(logits)
    for column in (vocab - 1, 0, vocab // 2):
        logits[7].zero_()
        logits[7, column] = 2
        graph.replay()
        check(result)


@pytest.mark.parametrize("rows", [1, 2, 4, 16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_k6_dependent_chain_graph(rows, dtype):
    """Each step depends on the previously selected token, like DSpark."""
    vocab, steps = 129280, 6
    base = torch.randn(rows, steps, vocab, device="cuda", dtype=dtype)
    bias = torch.randn(31, vocab, device="cuda", dtype=dtype)
    anchors = torch.arange(rows, device="cuda", dtype=torch.int64) % 31
    def chain(select):
        previous = anchors
        selected = []
        for step in range(steps):
            previous = select(base[:, step] + bias[previous % 31])
            selected.append(previous)
        return torch.stack(selected, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            chain(greedy_argmax)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = chain(greedy_argmax)
    for delta in (1, 7, 29):
        anchors.add_(delta)
        base.add_(torch.randn_like(base) * 0.1)
        graph.replay()
        expected = chain(lambda x: x.argmax(-1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_non_unit_stride_and_empty_rows():
    logits = torch.randn(3, 8192, device="cuda")[:, ::2]
    torch.testing.assert_close(greedy_argmax(logits), logits.argmax(-1), rtol=0, atol=0)
    empty = torch.empty(0, 129280, device="cuda")
    result = greedy_argmax(empty)
    assert result.shape == (0,) and result.dtype == torch.int64 and result.is_cuda
