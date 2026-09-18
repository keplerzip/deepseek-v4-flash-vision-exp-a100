"""Actual CLI parsing, model-class resolution and image preprocessing; no weights."""
import json
import sys
from pathlib import Path
from engine import arguments, settings
cfg = settings()
import torch
from PIL import Image
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.config import set_current_vllm_config
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers import get_tokenizer
from vllm.v1.worker.gpu_worker import Worker
from vllm.models.deepseek_v4.nvidia.vl_model import DeepseekV4ForConditionalGeneration
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.model_executor.model_loader.utils import get_model_architecture
parser = make_arg_parser(FlexibleArgumentParser())
cli = parser.parse_args(arguments(cfg)[2:])
cli.model = cli.model_tag
engine = EngineArgs.from_cli_args(cli).create_engine_config()
wrapped = {name: hasattr(getattr(DeepseekV4Attention, name), '__wrapped__') for name in ('_prepare_and_attn_eager', '_sparse_indexer_and_attn')}
assert all(wrapped.values()), wrapped
assert engine.use_v2_model_runner is True, engine.use_v2_model_runner
assert engine.model_config.architectures == ['DeepseekV4ForConditionalGeneration']
assert engine.model_config.is_multimodal_model and engine.model_config.is_mm_prefix_lm
assert engine.model_config.max_model_len == 262144 and engine.scheduler_config.max_num_seqs == 32
assert engine.scheduler_config.disable_chunked_mm_input
assert engine.cache_config.enable_prefix_caching is True
assert engine.cache_config.prefix_caching_hash_algo == 'sha256'
assert engine.cache_config.gpu_memory_utilization == 0.92
assert cli.served_model_name == ['DeepSeek-V4-Flash']
assert cli.limit_mm_per_prompt == {'image': 999}
assert cli.enable_prompt_tokens_details and cli.enable_force_include_usage
assert engine.model_config.served_model_name == 'DeepSeek-V4-Flash'
assert engine.model_config.multimodal_config.mm_encoder_attn_backend.name == 'TORCH_SDPA'
assert engine.model_config.dtype == torch.bfloat16
spec = engine.speculative_config
assert spec.use_dspark() and spec.num_speculative_tokens == 6
assert spec.draft_model_config.max_model_len == 262144
assert spec.draft_model_config.architectures == ['DSparkDraftModel'], spec.draft_model_config.architectures
assert get_model_architecture(spec.draft_model_config)[0].__name__ == 'DSparkDeepseekV4ForCausalLM'
with set_current_vllm_config(engine):
    tokenizer = get_tokenizer('/model', tokenizer_mode='deepseek_v4')
    processor = MULTIMODAL_REGISTRY.create_processor(engine.model_config, tokenizer=tokenizer)
    images = [Image.open(Path(__file__).resolve().parents[1] / 'tests/multimodal/chart_before.png')]
    data = processor.data_parser.parse_mm_data({'image': images})
    result = processor('Read this image: <｜deepseek_image｜>', data)
    assert len(result['mm_placeholders']['image']) == 1
    assert result['mm_kwargs']
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
    from vllm.tokenizers.hf import ThreadSafeHFTokenizerMixin
    renderer = DeepseekV4Renderer(engine, tokenizer)
    try:
        assert isinstance(renderer.get_tokenizer(), ThreadSafeHFTokenizerMixin)
        assert renderer.get_mm_processor().info.get_tokenizer() is renderer.get_tokenizer()
    finally:
        renderer.shutdown()
print(json.dumps({'status': 'PASS', 'scope': 'CLI/config, Worker imports, real-image CPU preprocessing; not model inference', 'release': 'R3.4', 'attention_graph_breaks': wrapped, 'model_runner': 'V2' if engine.use_v2_model_runner else 'V1', 'architecture': engine.model_config.architectures, 'dtype': str(engine.model_config.dtype), 'vision_attention': 'TORCH_SDPA', 'context': 262144, 'concurrency': 32, 'speculative_method': spec.method if spec else None, 'raw_image_tokens': len(result['prompt_token_ids']), 'prefix_caching': engine.cache_config.enable_prefix_caching, 'gpu_memory_utilization': engine.cache_config.gpu_memory_utilization, 'served_model_names': cli.served_model_name, 'gpu_inference_tested': False}, ensure_ascii=False))
