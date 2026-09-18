"""R3.4 single service contract; model path is a read-only mount."""
import json
import os
from pathlib import Path
import sys

EXPECTED = {'decoding': 'dspark', 'enforce_eager': False, 'gpu_memory_utilization': 0.92, 'model': '/model', 'served_model_name': 'DeepSeek-V4-Flash', 'max_model_len': 262144, 'max_num_seqs': 32, 'tensor_parallel_size': 8, 'max_num_batched_tokens': 4096, 'kv_cache_dtype': 'fp8', 'block_size': 256, 'limit_mm_per_prompt': {'image': 999}, 'mm_processor_cache_gb': 0, 'enable_prefix_caching': True, 'disable_chunked_mm_input': True, 'speculative_config': {'method': 'dspark', 'num_speculative_tokens': 6, 'draft_sample_method': 'greedy', 'max_model_len': 262144}, 'dtype': 'bfloat16', 'mm_encoder_attn_backend': 'TORCH_SDPA', 'prefix_caching_hash_algo': 'sha256', 'release': 'R3.4'}

def settings():
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'config/service.json').read_text())
    if cfg != EXPECTED:
        raise ValueError('R3.4 fixed service configuration changed; restore config/service.json')
    # Runner/graph decorators must read these values before vLLM import.
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return cfg

def arguments(cfg):
    args = [
        "vllm",
        "serve",
        cfg["model"],
        "--host",
        "0.0.0.0",
        "--port",
        "8005",
        "--tokenizer-mode",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
        "--reasoning-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--enable-chunked-prefill",
        "--disable-chunked-mm-input",
        "--enable-prefix-caching" if cfg["enable_prefix_caching"] else "--no-enable-prefix-caching",
        "--prefix-caching-hash-algo",
        "sha256",
        "--mm-processor-cache-gb",
        "0",
        "--mm-encoder-tp-mode",
        "weights",
        "--enable-tokenizer-info-endpoint",
        "--enable-prompt-tokens-details",
        "--enable-force-include-usage",
        "--enable-per-request-metrics",
        "--generation-config",
        "vllm",
    ]
    names = cfg["served_model_name"]
    args += ["--served-model-name"] + ([names] if isinstance(names, str) else names)
    for key in [
        "max_model_len",
        "max_num_seqs",
        "tensor_parallel_size",
        "max_num_batched_tokens",
        "dtype",
        "mm_encoder_attn_backend",
        "kv_cache_dtype",
        "block_size",
        "gpu_memory_utilization",
    ]:
        args += ["--" + key.replace("_", "-"), str(cfg[key])]
    args += ["--limit-mm-per-prompt", json.dumps(cfg["limit_mm_per_prompt"])]
    if cfg["enforce_eager"]:
        args += ["--enforce-eager"]
    if "speculative_config" in cfg:
        args += ["--speculative-config", json.dumps(cfg["speculative_config"])]
    # VLLM_API_KEY is read directly by the server; never put a key in argv/logs.
    return args


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--print']):
        raise SystemExit('Usage: engine.py [--print]; R3.4 has one deployment configuration')
    cfg = settings()
    argv = arguments(cfg)
    print(json.dumps({'release': 'R3.4', 'config': cfg, 'argv': argv}, ensure_ascii=False), flush=True)
    if '--print' not in sys.argv:
        os.environ.pop('VLLM_API_KEY', None)
        os.environ.pop('R3_API_KEY', None)
        os.execvp(argv[0], argv)
