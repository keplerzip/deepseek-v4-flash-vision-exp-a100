#include "libtorch_stable/moe/moe_ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(r3_vision_ops, m) {
  m.def("topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize, float routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, Tensor? is_padding, Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(r3_vision_ops, CUDA, m) {
  m.impl("topk_softplus_sqrt", TORCH_BOX(&topk_softplus_sqrt));
}
