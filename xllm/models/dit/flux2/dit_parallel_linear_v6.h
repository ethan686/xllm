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

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {

enum class LinearType { Default, SequenceParallel, TensorParallel };

struct SpOptions {
  int64_t head_num = 0;
  int64_t head_dim = 0;
  int64_t hidden_size = 0;
  bool before_attention = false;
  ProcessGroup* process_group = nullptr;

  SpOptions() = default;

  SpOptions(int64_t head_num,
            int64_t head_dim,
            int64_t hidden_size,
            bool before_attention,
            ProcessGroup* process_group = nullptr)
      : head_num(head_num),
        head_dim(head_dim),
        hidden_size(hidden_size),
        before_attention(before_attention),
        process_group(process_group) {}

  void validate() const {
    CHECK(head_num > 0) << "head_num should be greater than 0, got "
                        << head_num;
    CHECK(head_dim > 0) << "head_dim should be greater than 0, got "
                        << head_dim;
    CHECK(hidden_size > 0) << "hidden_size should be greater than 0, got "
                           << hidden_size;
    CHECK(hidden_size == head_dim * head_num)
        << "hidden_size should equal to head_dim * head_num, got head_dim "
        << head_dim << ", head_num " << head_num << ", hidden_size "
        << hidden_size;
    if (!process_group) {
      LOG(ERROR) << "SpOptions expected an initialized process_group for "
                    "all2all communication, but got nullptr";
    }
  }
};

struct TpOptions {
  bool column_parallel = true;
  int64_t tp_rank = 0;
  int64_t tp_size = 1;
  bool gather_output = false;
  bool need_scatter = false;
  bool is_save = false;
  ProcessGroup* process_group = nullptr;

  TpOptions() = default;

  TpOptions(bool column_parallel,
            int64_t tp_rank,
            int64_t tp_size,
            bool gather_output = false,
            bool need_scatter = false,
            bool is_save = false,
            ProcessGroup* process_group = nullptr)
      : column_parallel(column_parallel),
        tp_rank(tp_rank),
        tp_size(tp_size),
        gather_output(gather_output),
        need_scatter(need_scatter),
        is_save(is_save),
        process_group(process_group) {}

  void validate() const {
    CHECK(tp_size > 0) << "tp_size should be greater than 0, got " << tp_size;
    CHECK(tp_rank >= 0 && tp_rank < tp_size)
        << "tp_rank should be in [0, tp_size), got tp_rank " << tp_rank
        << ", tp_size " << tp_size;
    if (!process_group) {
      LOG(ERROR) << "TpOptions expected an initialized process_group for "
                    "tensor parallel communication, but got nullptr";
    }
  }
};

class DiTParallelLinearImpl : public torch::nn::Module {
 public:
  DiTParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      LinearType linear_type = LinearType::Default,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias),
        options_(options),
        linear_type_(linear_type),
        sp_options_(sp_options),
        tp_options_(tp_options) {
    if (linear_type_ == LinearType::Default) {
      // Use WeightTransposeAddMatmul for default case
      linear_ = register_module("linear",
                                layer::WeightTransposeAddMatmul(
                                    in_features, out_features, bias, options));
    } else if (linear_type_ == LinearType::SequenceParallel) {
      CHECK(sp_options_.has_value())
          << "SpOptions must be provided for SequenceParallel";
      sp_options_.value().validate();
      linear_ = register_module("linear",
                                layer::WeightTransposeAddMatmul(
                                    in_features, out_features, bias, options));
    } else if (linear_type_ == LinearType::TensorParallel) {
      CHECK(tp_options_.has_value())
          << "TpOptions must be provided for TensorParallel";
      tp_options_.value().validate();
      init_tensor_parallel_weights();
    }
  }

  torch::Tensor forward(const torch::Tensor& input, bool if_save = false) {
    switch (linear_type_) {
      case LinearType::Default:
        return forward_default(input);
      case LinearType::SequenceParallel:
        return forward_sequence_parallel(input);
      case LinearType::TensorParallel:
        return forward_tensor_parallel(input, if_save);
      default:
        LOG(FATAL) << "Unknown LinearType: " << static_cast<int>(linear_type_);
        return input;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    if (linear_type_ == LinearType::TensorParallel) {
      load_tensor_parallel_weights(state_dict);
    } else {
      linear_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (linear_type_ == LinearType::TensorParallel) {
      CHECK(weight_is_loaded_)
          << "weight is not loaded for " << prefix + "weight";
      if (has_bias_) {
        CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
      }
    } else {
      linear_->verify_loaded_weights(prefix);
    }
  }

  // Directly set the weight tensor
  torch::Tensor get_weight() const { return weight_; }

  void set_weight(const torch::Tensor& weight) {
    if (linear_type_ == LinearType::TensorParallel) {
      weight_ = weight;
      weight.to(options_);
      weight_is_loaded_ = true;
    } else {
      // For default and sequence parallel, we need to set weight through
      // linear_ This requires access to WeightTransposeAddMatmul's weight,
      // which is protected For now, we'll keep using load_state_dict for these
      // cases
      std::unordered_map<std::string, torch::Tensor> temp_dict;
      temp_dict["weight"] = weight;
      StateDict temp_state_dict(temp_dict);
      linear_->load_state_dict(temp_state_dict);
    }
  }

 private:
  torch::Tensor forward_default(const torch::Tensor& input) {
    return linear_->forward(input);
  }

  torch::Tensor forward_sequence_parallel(const torch::Tensor& input) {
    CHECK(input.sizes().size() == 3)
        << "SP linear input is expected to be a tensor "
        << "with shape {batch, seq_len, hidden_size}";

    const auto& options = sp_options_.value();
    auto group_size = options.process_group->world_size();

    if (options.before_attention) {
      auto linear_output = linear_->forward(input);
      auto all_to_all_func = parallel_state::all_to_all_4D(
          linear_output.view(
              {input.size(0), -1, options.head_num, options.head_dim}),
          2,
          1,
          false,
          options.process_group);
      auto output = all_to_all_func();
      return output.view({input.size(0), -1, options.hidden_size / group_size});
    } else {
      auto all_to_all_func = parallel_state::all_to_all_4D(
          input.view({input.size(0),
                      -1,
                      options.head_num / group_size,
                      options.head_dim}),
          1,
          2,
          false,
          options.process_group);
      auto all_to_all_output = all_to_all_func();
      all_to_all_output =
          all_to_all_output.view({input.size(0), -1, options.hidden_size});
      auto output = linear_->forward(all_to_all_output);
      return output;
    }
  }

  void init_tensor_parallel_weights() {
    const auto& options = tp_options_.value();
    int64_t tp_size = options.tp_size;

    if (options.column_parallel) {
      // Column parallel: split output features
      int64_t out_features_per_partition = out_features_ / tp_size;
      weight_ = register_parameter(
          "weight",
          torch::empty({out_features_per_partition, in_features_}, options_),
          false);
      if (has_bias_) {
        bias_ = register_parameter(
            "bias",
            torch::empty({out_features_per_partition}, options_),
            false);
      }
    } else {
      // Row parallel: split input features
      int64_t in_features_per_partition = in_features_ / tp_size;
      weight_ = register_parameter(
          "weight",
          torch::empty({out_features_, in_features_per_partition}, options_),
          false);
      if (has_bias_) {
        bias_ = register_parameter(
            "bias", torch::empty({out_features_}, options_), false);
      }
    }
  }

  torch::Tensor forward_tensor_parallel(const torch::Tensor& input,
                                        bool if_save = false) {
    CHECK(input.sizes().size() == 3)
        << "TP linear input is expected to be a tensor "
        << "with shape {batch, seq_len, hidden_size}";

    const auto& options = tp_options_.value();
    if (options.tp_size <= 1) {
      return linear_->forward(input);
    }

    if (options.column_parallel) {
      return forward_column_parallel(input, if_save);
    } else {
      return forward_row_parallel(input, if_save);
    }
  }

  torch::Tensor forward_column_parallel(const torch::Tensor& input,
                                        bool if_save = false) {
    const auto& options = tp_options_.value();

    if (!weight_is_loaded_) {
      LOG(INFO) << "weight is not loaded for column parallel";
    }

    // Use direct matmul instead of WeightTransposeAddMatmul
    auto bias = has_bias_ ? std::optional<torch::Tensor>(bias_) : std::nullopt;
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;

    LOG(INFO) << "input; device" << matmul_params.a.device();
    LOG(INFO) << "weight device" << matmul_params.b.device();

    LOG(INFO) << "************************options.tp_rank:" << options.tp_rank;
    if (if_save) {
      LOG(INFO) << "weight shape" << weight_.sizes();
      torch::Tensor weight_cpu = weight_.to(torch::kCPU);
      torch::save(weight_cpu,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/01_single_parallel_weight/rank" +
                      std::to_string(options.tp_rank) +
                      "/01_single_parallel_qkvmlp_weight.pt");
      LOG(INFO) << "save the weight successfully !!!";
    }
    matmul_params.bias = bias;
    auto output = xllm::kernel::matmul(matmul_params);
    // LOG(INFO) << "11111111output shape" << output.sizes();

    if (!options.process_group) {
      LOG(INFO) << "!process_group is true!!!!!";
    }

    if (options.gather_output) {
      // LOG(INFO) << "options.gather_output" << options.gather_output;
      output = parallel_state::gather(output, options.process_group, -1);
      LOG(INFO) << "22222222output shape" << output.sizes();
    }
    LOG(INFO) << "3333333333output shape" << output.sizes();
    return output;
  }

  torch::Tensor forward_row_parallel(const torch::Tensor& input,
                                     bool if_save = false) {
    const auto& options = tp_options_.value();

    auto scattered_input = input;
    // Scatter input if needed
    if (options.need_scatter) {
      scattered_input =
          parallel_state::scatter(input, options.process_group, -1);
    }

    // Use direct matmul instead of WeightTransposeAddMatmul
    auto bias = has_bias_ ? std::optional<torch::Tensor>(bias_) : std::nullopt;
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = scattered_input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    auto output = xllm::kernel::matmul(matmul_params);
    // Reduce output
    // if (tp_options_.value().is_save) {
    //   LOG(INFO) << "save output shape" << output.sizes();
    //   torch::save(output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(tp_options_.value().tp_rank) +
    //   "/05_29_save_hidden_output_0.pt");
    // }
    // torch::npu::synchronize();
    // if (if_save) {
    //   LOG(INFO) << "Before reduce output save shape" << output.sizes();
    //   torch::save(output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(options.tp_rank) + "/05_29_save_hidden_output_0.pt");
    //   // tp_options_.value().is_save = false;
    // }
    // Reduce output

    if (!options.process_group) {
      LOG(INFO) << "!process_group is true!!!!!";
    }

    output = parallel_state::reduce(output, options.process_group);
    // if (if_save) {
    //   LOG(INFO) << "Reduce output save shape" << output.sizes();
    //   torch::save(output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(options.tp_rank) + "/05_30_save_hidden_output_0.pt");
    // }

    return output;
  }

  void load_tensor_parallel_weights(const StateDict& state_dict) {
    const auto& options = tp_options_.value();
    const int64_t rank = options.tp_rank;
    const int64_t world_size = options.tp_size;

    if (options.column_parallel) {
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  0,
                                  rank,
                                  world_size,
                                  weight_,
                                  weight_is_loaded_);
      if (has_bias_) {
        weight::load_sharded_weight(
            state_dict, "bias", 0, rank, world_size, bias_, bias_is_loaded_);
      }
    } else {
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  1,
                                  rank,
                                  world_size,
                                  weight_,
                                  weight_is_loaded_);
      if (has_bias_) {
        weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
      }
    }
  }

  // Common parameters
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
  torch::TensorOptions options_;

  // Module for default and sequence parallel cases
  layer::WeightTransposeAddMatmul linear_{nullptr};
  LinearType linear_type_;
  std::optional<SpOptions> sp_options_;
  std::optional<TpOptions> tp_options_;

  // Tensor parallel specific members
  torch::Tensor weight_;
  torch::Tensor bias_;
  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;
};

TORCH_MODULE(DiTParallelLinear);

class DiTParallelLinearFactory {
 public:
  static DiTParallelLinear create(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      LinearType linear_type = LinearType::Default,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt) {
    return DiTParallelLinear(in_features,
                             out_features,
                             bias,
                             options,
                             linear_type,
                             sp_options,
                             tp_options);
  }
};

}  // namespace xllm
