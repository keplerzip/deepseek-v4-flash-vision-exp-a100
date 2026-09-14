from torch.utils.cpp_extension import load
from pathlib import Path
load(name='_r3_vision_ops',sources=['/src/csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu','/build/vision_bindings.cpp'],extra_include_paths=['/src/csrc'],extra_cflags=['-O3','-DUSE_CUDA','-DTORCH_TARGET_VERSION=0x020B000000000000ULL'],extra_cuda_cflags=['-O3','--use_fast_math','-DUSE_CUDA','-DTORCH_TARGET_VERSION=0x020B000000000000ULL'],build_directory='/build/output',is_python_module=False,verbose=True)
print('NATIVE_VISION_OPS_BUILD=PASS')
