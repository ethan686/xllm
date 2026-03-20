/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "dit_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "linear.h"

namespace xllm {
namespace layer {

Dit_ColumnParallelLinearImpl::Dit_ColumnParallelLinearImpl(
    const ModelContext& context)
    : Dit_ColumnParallelLinearImpl(context.get_model_args().hidden_size(),
                                   context.get_model_args().vocab_size(),
                                   /*bias=*/false,
                                   /*gather_output=*/true,
                                   QuantArgs{},
                                   context.get_parallel_args().tp_group_,
                                   context.get_tensor_options()) {}

// Linear layer with column parallelism.
Dit_ColumnParallelLinearImpl::Dit_ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options)
    // const FusedLinearExtraArgs& linear_extra_args)
    : gather_output_(gather_output),
      quant_args_(quant_args),
      process_group_(process_group),
      device_(options.device()) {
  LOG(INFO) << "in_features: " << in_features;
  if (process_group_ == nullptr) LOG(FATAL) << "process_group_ is null!";
  LOG(INFO) << "process_group_->rank(); " << process_group_->rank();
  rank_ = process_group_->rank();
  LOG(INFO) << "rank: " << rank_;
  world_size_ = process_group_->world_size();
  LOG(INFO) << " world_size: " << process_group_->world_size();
  CHECK(out_features % world_size_ == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size_;
  const int64_t out_features_per_partition = out_features / world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == "smoothquant") {
    LOG(INFO) << "SmoothQuant is enabled";
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({in_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else {
    LOG(INFO) << "SmoothQuant is disabled";
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor Dit_ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (quant_args_.quant_method() == "smoothquant") {
    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    // Quantize input tensor with int8 in 'dynamic_per_token' mode using
    // scaled_quantize
    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.active_coef = 1.0;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    // For SmoothQuant, use scaled_matmul with quantization parameters
    // for now, we only support w8a8 quantization
    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else {
    // For unquantized case, use regular matmul
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void Dit_ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight
  if (quant_args_.quant_method() == "smoothquant") {
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);
    // for input, there is one smooth value
    LOAD_WEIGHT(smooth);
  } else {
    LOAD_SHARDED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void Dit_ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight
  if (quant_args_.quant_method() == "smoothquant") {
    // Find the first available "smooth" tensor in prefixes (e.g.,
    // "gate.smooth", "up_proj.smooth", etc.)
    for (const auto& prefix : prefixes) {
      auto smooth_tensor_candidate = state_dict.get_tensor(prefix + "smooth");
      if (smooth_tensor_candidate.defined()) {
        // Copy the found smooth tensor to the module parameter
        CHECK_EQ(smooth_.sizes(), smooth_tensor_candidate.sizes())
            << "smooth weight size mismatch for " << state_dict.prefix()
            << "smooth";
        smooth_.copy_(smooth_tensor_candidate);
        smooth_is_loaded_ = true;
        break;
      }
    }
    LOAD_FUSED_WEIGHT(qweight, 0);
    LOAD_FUSED_WEIGHT(per_channel_scale, 0);
  } else {
    LOAD_FUSED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

Dit_QKVParallelLinearImpl::Dit_QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, hidden_size}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor Dit_QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;

  auto output = xllm::kernel::matmul(matmul_params);
  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

void Dit_QKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  LOAD_QKV_WEIGHT(weight, 0, num_kv_head_replicas_);
  if (bias_.defined()) {
    LOAD_QKV_WEIGHT(bias, 0, num_kv_head_replicas_);
  }
}

void Dit_QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int32_t shard_tensor_count = 3;
  const int64_t shard_size = num_heads_ * head_size_;
  CHECK_EQ(num_heads_, num_kv_heads_);
  LOAD_MERGED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
}

// Linear layer with row parallelism.
Dit_RowParallelLinearImpl::Dit_RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    // const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options)
    // const FusedLinearExtraArgs& linear_extra_args)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      process_group_(process_group) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (quant_args_.quant_method() == "smoothquant") {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ = register_parameter(
        "per_channel_scale",
        torch::empty({out_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    smooth_ = register_parameter("smooth",
                                 torch::empty({in_features_per_partition},
                                              options.dtype(torch::kFloat32)),
                                 /*requires_grad=*/false);
    // Output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor Dit_RowParallelLinearImpl::forward(torch::Tensor input) {
  if (!input_is_parallelized_) {
    input = xllm::parallel_state::scatter(input, process_group_);
  }

  auto bias = (bias_.defined() && rank_ == 0)
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;
  torch::Tensor output;
  if (quant_args_.quant_method() == "smoothquant") {
    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    // Call scaled_quantize: quantizes input tensor with int8 in
    // 'dynamic_per_token' mode
    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.active_coef = 1.0;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    // for now, we only support w8a8 quantization
    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }
  if (enable_result_reduction_ && world_size_ > 1) {
    output = xllm::parallel_state::reduce(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void Dit_RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;

  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight.
  if (quant_args_.quant_method() == "smoothquant") {
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_WEIGHT(per_channel_scale);
    LOAD_SHARDED_WEIGHT(smooth, 0);
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options) {
  weight_ =
      register_parameter("weight",
                         torch::empty({out_features, in_features}, options),
                         /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;

  auto output = xllm::kernel::matmul(matmul_params);
  return output;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
