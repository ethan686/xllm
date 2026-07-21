/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <utility>

#include "core/framework/parallel_state/process_group.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace dit {

// Mixin providing classifier-free guidance (CFG) parallelism.
//
// Usage:
//   class MyPipeline : public torch::nn::Module,
//                      public dit::CFGParallelMixin { ... };
class CFGParallelMixin {
 public:
  // Returns {positive_noise_pred, negative_noise_pred}.
  template <typename ForwardFn>
  std::pair<torch::Tensor, torch::Tensor> forward_cfg(ForwardFn&& forward_fn,
                                                      ProcessGroup* pg,
                                                      bool do_cfg) const {
    if (!do_cfg) {
      return {forward_fn(true), torch::Tensor()};
    }

    if (pg != nullptr && pg->world_size() > 1) {
      int32_t rank = pg->rank();
      torch::Tensor noise_pred = forward_fn(rank == 0);
      torch::Tensor gathered =
          parallel_state::gather(noise_pred, pg, /*dim=*/0);
      auto chunks = torch::chunk(gathered, 2, 0);
      return {chunks[0], chunks[1]};
    }

    return {forward_fn(true), forward_fn(false)};
  }
};

// Mixin for VAE parallelism (to be implemented).
class VaeParallelMixin {};

// Mixin for sequence parallelism (to be implemented).
class SpParallelMixin {};

}  // namespace dit
}  // namespace xllm
