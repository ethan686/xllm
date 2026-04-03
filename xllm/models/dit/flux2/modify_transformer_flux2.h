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
#include <glog/logging.h>
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "dit_parallel_linear_v6.h"
#include "framework/model_context.h"
#include "models/dit/transformer_flux.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {
class Flux2SwiGLUImpl : public torch::nn::Module {
 public:
  Flux2SwiGLUImpl() { gate_fn_ = torch::nn::SiLU(); }

  torch::Tensor forward(torch::Tensor x) {
    auto chunks = torch::chunk(x, 2, /*dim=*/-1);
    torch::Tensor x1 = chunks[0];
    torch::Tensor x2 = chunks[1];

    torch::Tensor x_out = gate_fn_(x1) * x2;
    return x_out;
  }

 private:
  torch::nn::SiLU gate_fn_;
};
TORCH_MODULE(Flux2SwiGLU);

class Flux2FeedForwardImpl : public torch::nn::Module {
 public:
  explicit Flux2FeedForwardImpl(const ModelContext& context,
                                const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    auto eps = model_args.mlp_ratio();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;
    LinearType linear_type = LinearType::Default;
    std::optional<TpOptions> tp_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      linear_type = LinearType::TensorParallel;
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/false,
          /*is_save=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);

      if (!parallel_args_.dit_tp_group_ || !tp_options.has_value() ||
          !tp_options->process_group) {
        LOG(INFO) << "zhubowei test !process_group is true!!!!!";
      }
    }

    auto linear_in = DiTParallelLinearFactory::create(inner_dim,
                                                      inner_dim * 6,
                                                      false,
                                                      options_,
                                                      linear_type,
                                                      std::nullopt,
                                                      tp_options);
    linear_in_ = register_module("linear_in", linear_in);
    act_fn_ = register_module("act_fn", Flux2SwiGLU());

    LinearType linear_out_type = LinearType::Default;
    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      linear_out_type = LinearType::TensorParallel;
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/true,
          /*is_save=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto linear_out = DiTParallelLinearFactory::create(inner_dim * 3,
                                                       inner_dim,
                                                       false,
                                                       options_,
                                                       linear_out_type,
                                                       std::nullopt,
                                                       tp_out_options);
    linear_out_ = register_module("linear_out", linear_out);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const int flag_dit_timestep,
                        const int i,
                        const bool is_save) {
    LOG(INFO) << "Flux2FeedForwardImpl forward";
    LOG(INFO) << "hidden_states shape" << hidden_states.sizes();
    auto out = linear_in_->forward(hidden_states, false);
    if (flag_dit_timestep == 0 and i == 0 and is_save) {
      torch::Tensor save_linear_in = out.to(torch::kCPU);
      LOG(INFO) << "-------------linear_in shape" << out.sizes();
      torch::save(
          save_linear_in,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/06_cpp_double_inner_FeedForward_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/06_01_save_linear_in_0.pt");
    }
    out = act_fn_->forward(out);
    if (flag_dit_timestep == 0 and i == 0 and is_save) {
      torch::Tensor save_act_fn = out.to(torch::kCPU);
      LOG(INFO) << "-------------act_fn shape" << out.sizes();
      torch::save(
          save_act_fn,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/06_cpp_double_inner_FeedForward_tensor/rank" +
              std::to_string(parallel_args_.rank_) + "/06_02_save_act_fn_0.pt");
    }
    out = linear_out_->forward(out);
    if (flag_dit_timestep == 0 and i == 0 and is_save) {
      torch::Tensor save_linear_out = out.to(torch::kCPU);
      LOG(INFO) << "-------------linear_out shape" << out.sizes();
      torch::save(
          save_linear_out,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/06_cpp_double_inner_FeedForward_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/06_03_save_linear_out_0.pt");
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_in_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("linear_in."));
    linear_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("linear_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_in_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                               "linear_in.");
    linear_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "linear_out.");
  }

 private:
  int64_t dim_;
  int64_t inner_dim_;
  float mult_;
  DiTParallelLinear linear_in_{nullptr};
  Flux2SwiGLU act_fn_{nullptr};
  DiTParallelLinear linear_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2FeedForward);

class Flux2AttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2AttentionImpl(const ModelContext& context,
                              const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    added_kv_proj_dim_ = query_dim_;

    LinearType linear_type = LinearType::Default;
    std::optional<TpOptions> tp_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      linear_type = LinearType::TensorParallel;
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*is_save=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_q = DiTParallelLinearFactory::create(query_dim_,
                                                 out_dim_,
                                                 false,
                                                 options_,
                                                 linear_type,
                                                 std::nullopt,
                                                 tp_options);
    to_q_ = register_module("to_q", to_q);

    auto to_k = DiTParallelLinearFactory::create(query_dim_,
                                                 out_dim_,
                                                 false,
                                                 options_,
                                                 linear_type,
                                                 std::nullopt,
                                                 tp_options);
    to_k_ = register_module("to_k", to_k);

    auto to_v = DiTParallelLinearFactory::create(query_dim_,
                                                 out_dim_,
                                                 false,
                                                 options_,
                                                 linear_type,
                                                 std::nullopt,
                                                 tp_options);
    to_v_ = register_module("to_v", to_v);

    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

    LinearType to_out_type = LinearType::Default;
    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      to_out_type = LinearType::TensorParallel;
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*is_save=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_out = DiTParallelLinearFactory::create(out_dim_,
                                                   query_dim_,
                                                   false,
                                                   options_,
                                                   to_out_type,
                                                   std::nullopt,
                                                   tp_out_options);
    to_out_ = register_module("to_out", to_out);

    if (added_kv_proj_dim_ > 0) {
      LinearType to_add_out_type = LinearType::Default;
      std::optional<TpOptions> tp_add_out_options = std::nullopt;

      if (FLAGS_dit_tp_size > 1) {
        to_add_out_type = LinearType::TensorParallel;
        tp_add_out_options = TpOptions(
            /*column_parallel=*/false,
            /*tp_rank=*/parallel_args_.rank_,
            /*tp_size=*/FLAGS_dit_tp_size,
            /*gather_output=*/false,
            /*need_scatter=*/false,
            /*is_save=*/true,
            /*process_group=*/parallel_args_.dit_tp_group_);
      }

      auto to_add_out = DiTParallelLinearFactory::create(out_dim_,
                                                         added_kv_proj_dim_,
                                                         false,
                                                         options_,
                                                         to_add_out_type,
                                                         std::nullopt,
                                                         tp_add_out_options);
      to_add_out_ = register_module("to_add_out", to_add_out);
      norm_added_q_ = register_module(
          "norm_added_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
      norm_added_k_ = register_module(
          "norm_added_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

      auto to_add_q = DiTParallelLinearFactory::create(added_kv_proj_dim_,
                                                       out_dim_,
                                                       false,
                                                       options_,
                                                       linear_type,
                                                       std::nullopt,
                                                       tp_options);
      to_add_q_ = register_module("to_add_q", to_add_q);

      auto to_add_k = DiTParallelLinearFactory::create(added_kv_proj_dim_,
                                                       out_dim_,
                                                       false,
                                                       options_,
                                                       linear_type,
                                                       std::nullopt,
                                                       tp_options);
      to_add_k_ = register_module("to_add_k", to_add_k);

      auto to_add_v = DiTParallelLinearFactory::create(added_kv_proj_dim_,
                                                       out_dim_,
                                                       false,
                                                       options_,
                                                       linear_type,
                                                       std::nullopt,
                                                       tp_options);
      to_add_v_ = register_module("to_add_v", to_add_v);
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& image_rotary_emb,
      const int flag_dit_timestep,
      const int i) {
    int64_t input_ndim = hidden_states.dim();
    // LOG(INFO) << "hidden_states shape" << hidden_states.sizes();
    torch::Tensor hidden_states_reshaped = hidden_states;
    if (input_ndim == 4) {
      auto shape = hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      hidden_states_reshaped =
          hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    LOG(INFO) << "hidden_states_reshaped shape"
              << hidden_states_reshaped.sizes();
    int64_t context_input_ndim = encoder_hidden_states.dim();
    torch::Tensor encoder_hidden_states_reshaped = encoder_hidden_states;
    if (context_input_ndim == 4) {
      auto shape = encoder_hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      encoder_hidden_states_reshaped =
          encoder_hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_encoder_hidden_states =
    //   encoder_hidden_states.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/05_01_save_encoder_hidden_states_0.pt");
    //   torch::Tensor save_encoder_hidden_states_reshaped =
    //   encoder_hidden_states_reshaped.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_reshaped,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/05_02_save_encoder_hidden_states_reshaped_0.pt");

    // }
    LOG(INFO) << "encoder_hidden_states shape" << encoder_hidden_states.sizes();
    LOG(INFO) << "encoder_hidden_states_reshaped shape"
              << encoder_hidden_states_reshaped.sizes();

    int64_t batch_size = encoder_hidden_states_reshaped.size(0);
    LOG(INFO) << "zhubowei checkpoint4";

    hidden_states_reshaped = hidden_states_reshaped.squeeze(1).squeeze(1);
    encoder_hidden_states_reshaped =
        encoder_hidden_states_reshaped.squeeze(1).squeeze(1);
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_hidden_states_reshaped =
    //   hidden_states_reshaped.to(torch::kCPU);
    //   torch::save(save_hidden_states_reshaped,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/05_01_save_hidden_states_reshaped_0.pt");
    //   torch::Tensor save_encoder_hidden_states_reshaped =
    //   encoder_hidden_states_reshaped.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_reshaped,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/05_02_save_encoder_hidden_states_reshaped_0.pt");

    // }
    auto query = to_q_->forward(hidden_states_reshaped, false);
    auto key = to_k_->forward(hidden_states_reshaped, false);
    auto value = to_v_->forward(hidden_states_reshaped, false);
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_query = query.to(torch::kCPU);
    //   // LOG(INFO) << "-------------parallel_args_.rank_:" <<
    //   parallel_args_.rank_; LOG(INFO) << "-------------query shape" <<
    //   query.sizes(); torch::save(save_query,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_03_save_query_0.pt");
    //   torch::Tensor save_key = key.to(torch::kCPU);
    //   LOG(INFO) << "-------------key shape" << key.sizes();
    //   torch::save(save_key,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_04_save_key_0.pt");
    //   torch::Tensor save_value = value.to(torch::kCPU);
    //   LOG(INFO) << "value shape" << value.sizes();
    //   torch::save(save_value,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_05_save_value_0.pt");
    // }
    LOG(INFO) << "zhubowei checkpoint5";
    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = inner_dim / head_dim_;

    int64_t head_dim = head_dim_;
    query = query.view({batch_size, -1, attn_heads, head_dim});
    key = key.view({batch_size, -1, attn_heads, head_dim});
    value = value.view({batch_size, -1, attn_heads, head_dim});
    if (norm_q_) query = std::get<0>(norm_q_(query));
    if (norm_k_) key = std::get<0>(norm_k_(key));
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_query = query.to(torch::kCPU);
    //   torch::save(save_query,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_06_save_query_0.pt");
    //   torch::Tensor save_key = key.to(torch::kCPU);
    //   torch::save(save_key,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_07_save_key_0.pt");
    //   torch::Tensor save_value = value.to(torch::kCPU);
    //   torch::save(save_value,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_08_save_value_0.pt");
    // }
    LOG(INFO) << "query shape" << query.sizes();
    LOG(INFO) << "key shape" << key.sizes();
    LOG(INFO) << "value shape" << value.sizes();
    LOG(INFO) << "image_rotary_emb shape" << image_rotary_emb.sizes();

    auto encoder_hidden_states_query_proj =
        to_add_q_->forward(encoder_hidden_states_reshaped, false);
    auto encoder_hidden_states_key_proj =
        to_add_k_->forward(encoder_hidden_states_reshaped, false);
    auto encoder_hidden_states_value_proj =
        to_add_v_->forward(encoder_hidden_states_reshaped, false);
    LOG(INFO) << "zhubowei checkpoint 6";
    // if (flag_dit_timestep == 0 and i == 0){
    //   LOG(INFO) << "encoder_hidden_states_query_proj shape" <<
    //   encoder_hidden_states_query_proj.sizes(); LOG(INFO) <<
    //   "encoder_hidden_states_key_proj shape" <<
    //   encoder_hidden_states_key_proj.sizes(); LOG(INFO) <<
    //   "encoder_hidden_states_value_proj shape" <<
    //   encoder_hidden_states_value_proj.sizes(); torch::Tensor
    //   save_encoder_hidden_states_query_proj =
    //   encoder_hidden_states_query_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_query_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_09_save_encoder_hidden_states_query_proj_0.pt"); torch::Tensor
    //   save_encoder_hidden_states_key_proj =
    //   encoder_hidden_states_key_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_key_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_10_save_encoder_hidden_states_key_proj_0.pt"); torch::Tensor
    //   save_encoder_hidden_states_value_proj =
    //   encoder_hidden_states_value_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_value_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_11_save_encoder_hidden_states_value_proj_0.pt");
    // }
    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          std::get<0>(norm_added_q_(encoder_hidden_states_query_proj));

    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          std::get<0>(norm_added_k_(encoder_hidden_states_key_proj));
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_encoder_hidden_states_query_proj =
    //   encoder_hidden_states_query_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_query_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_12_save_encoder_hidden_states_query_proj_0.pt"); torch::Tensor
    //   save_encoder_hidden_states_key_proj =
    //   encoder_hidden_states_key_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_key_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_13_save_encoder_hidden_states_key_proj_0.pt"); torch::Tensor
    //   save_encoder_hidden_states_value_proj =
    //   encoder_hidden_states_value_proj.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_value_proj,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_14_save_encoder_hidden_states_value_proj_0.pt");
    // }
    // LOG(INFO) << "encoder_hidden_states_query_proj shape"
    //           << encoder_hidden_states_query_proj.sizes();
    // LOG(INFO) << "encoder_hidden_states_key_proj shape"
    //           << encoder_hidden_states_key_proj.sizes();
    // LOG(INFO) << "encoder_hidden_states_value_proj shape"
    //           << encoder_hidden_states_value_proj.sizes();

    // TODO some are right some are wrong query1& key1.
    // encoder_hidden_states_query_proj
    auto query1 = torch::cat({encoder_hidden_states_query_proj, query}, 1);
    auto key1 = torch::cat({encoder_hidden_states_key_proj, key}, 1);
    auto value1 = torch::cat({encoder_hidden_states_value_proj, value}, 1);
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_query1 = query1.to(torch::kCPU);
    //   torch::save(save_query1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_15_save_query1_0.pt");
    //   torch::Tensor save_key1 = key1.to(torch::kCPU);
    //   torch::save(save_key1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_16_save_key1_0.pt");
    //   torch::Tensor save_value1 = value1.to(torch::kCPU);
    //   torch::save(save_value1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_17_save_value1_0.pt");
    // }
    if (image_rotary_emb.defined()) {
      LOG(INFO) << "zhubowei checkpoint if has rotary";
      query1 = apply_rotary_emb(query1, image_rotary_emb, false);
      key1 = apply_rotary_emb(key1, image_rotary_emb, false);
    }
    // if (flag_dit_timestep == 0 and i == 0){
    //   int64_t encoder_dim = encoder_hidden_states_query_proj.size(1);
    //   auto encoder_hidden_states_query_proj_rotated_1 = query1.slice(1, 0,
    //   encoder_dim); auto encoder_hidden_states_query_proj_rotated_2 =
    //   query1.slice(1, encoder_dim, torch::size_t(-1)); torch::Tensor
    //   save_encoder_hidden_states_query_proj_rotated_1 =
    //   encoder_hidden_states_query_proj_rotated_1.to(torch::kCPU);
    //   torch::save(save_encoder_hidden_states_query_proj_rotated_1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_18_save_encoder_hidden_states_query_proj_rotated_1_0.pt");
    //   torch::Tensor save_encoder_hidden_states_query_proj_rotated_2 =
    //   encoder_hidden_states_query_proj_rotated_2.to(torch::kCPU);
    //   torch::save(encoder_hidden_states_query_proj_rotated_2,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_19_save_encoder_hidden_states_query_proj_rotated_2_0.pt");
    // }
    LOG(INFO) << "zhubowei checkpoint 6";
    // LOG(INFO) << "queri1 shape" <<
    // encoder_hidden_states_query_proj_rotated_1.sizes(); LOG(INFO) << "queri1
    // shape" << encoder_hidden_states_query_proj_rotated_2.sizes();
    LOG(INFO) << "queri1 shape" << query1.sizes();
    LOG(INFO) << "key1 shape" << key1.sizes();
    LOG(INFO) << "value1 shape" << value1.sizes();
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_query1 = query1.to(torch::kCPU);
    //   torch::save(save_query1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_18_save_query1_0.pt");
    //   torch::Tensor save_key1 = key1.to(torch::kCPU);
    //   torch::save(save_key1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_19_save_key1_0.pt");
    //   torch::Tensor save_value1 = value1.to(torch::kCPU);
    //   torch::save(save_value1,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/05_20_save_value1_0.pt");
    // }
#if defined(USE_NPU)
    // torch::Tensor attn_output = torch::scaled_dot_product_attention(
    //     query1, key1, value1, torch::nullopt, 0.0, false);
    int64_t head_num_ = query1.size(2);
    int64_t head_dim_ = query1.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query1,
                                                         key1,
                                                         value1,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);

    attn_output = attn_output.reshape({batch_size, -1, attn_heads * head_dim});

    LOG(INFO) << "attn output shape" << attn_output.sizes();
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_attn_output = attn_output.to(torch::kCPU);
    //   torch::save(save_attn_output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_21_save_attn_output_0.pt");
    // }
#elif defined(USE_CUDA)
    // SDPA expects (B, H, S, D); our query1/key1/value1 are (B, S, H, D).
    // Transpose to match diffusers dispatch_attention_fn (permute 0,2,1,3).
    query1 = query1.transpose(1, 2);
    key1 = key1.transpose(1, 2);
    value1 = value1.transpose(1, 2);
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query1, key1, value1, torch::nullopt, 0.0, false);
    attn_output = attn_output.transpose(1, 2).reshape(
        {batch_size, -1, attn_heads * head_dim});
#else
    NOT_IMPLEMENTED();
#endif

    LOG(INFO) << "check before attn output";
    attn_output = attn_output.to(query.dtype());
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_attn_output = attn_output.to(torch::kCPU);
    //   torch::save(save_attn_output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_22_save_attn_output_0.pt");
    // }
    int64_t encoder_length = encoder_hidden_states_reshaped.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    // if (flag_dit_timestep == 0 and i == 0){
    //   torch::Tensor save_hidden_output = hidden_output.to(torch::kCPU);
    //   torch::save(save_hidden_output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_23_save_hidden_output_0.pt"); torch::Tensor save_encoder_output =
    //   encoder_output.to(torch::kCPU); torch::save(save_encoder_output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) +
    //   "/05_24_save_encoder_output_0.pt");
    // }
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);
    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "parallel rank" << parallel_args_.rank_;
      LOG(INFO) << "hidden_output shape" << hidden_output.sizes();
      LOG(INFO) << "encoder_output shape" << encoder_output.sizes();
      torch::Tensor save_encoder_output = encoder_output.to(torch::kCPU);
      torch::save(
          save_encoder_output,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/05_25_save_encoder_output_0.pt");
      torch::Tensor save_hidden_output = hidden_output.to(torch::kCPU);
      torch::save(
          save_hidden_output,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/05_26_save_hidden_output_0.pt");
    }

    hidden_output = to_out_->forward(hidden_output);

    bool if_save = false;
    if (flag_dit_timestep == 0 and i == 0) {
      if_save = true;
    }

    encoder_output = to_add_out_->forward(encoder_output, if_save);  // 有问题

    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "parallel rank" << parallel_args_.rank_;
      LOG(INFO) << "hidden_output shape" << hidden_output.sizes();
      LOG(INFO) << "encoder_output shape" << encoder_output.sizes();
      torch::Tensor save_hidden_output = hidden_output.to(torch::kCPU);
      torch::save(
          save_hidden_output,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/05_27_save_hidden_output_0.pt");
      torch::Tensor save_encoder_output = encoder_output.to(torch::kCPU);
      torch::save(
          save_encoder_output,
          "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
          "dump_flux2_tensor/05_cpp_double_inner_attention_tensor/rank" +
              std::to_string(parallel_args_.rank_) +
              "/05_28_save_encoder_output_0.pt");
    }

    LOG(INFO) << "zhubowei checkpoint 7";
    return std::make_tuple(hidden_output, encoder_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_q."));
    to_k_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_k."));
    to_v_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_v."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));

    LOG(INFO) << "zhubowei checkpoint 1";
    to_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out.0."));

    if (added_kv_proj_dim_ > 0) {
      norm_added_q_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_q."));
      norm_added_k_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_k."));
      to_add_q_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_q_proj."));
      to_add_k_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_k_proj."));
      to_add_v_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_v_proj."));
      to_add_out_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("to_add_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_q_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_q.");
    to_k_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_k.");
    to_v_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_v.");

    LOG(INFO) << "zhubowei checkpoint2";

    to_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                            "to_out.0.");
    if (added_kv_proj_dim_ > 0) {
      to_add_q_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_q_proj.");
      to_add_k_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_k_proj.");
      to_add_v_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_v_proj.");
      to_add_out_->as<DiTParallelLinear>()->verify_loaded_weights(
          prefix + "to_add_out.");
    }
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  int64_t added_kv_proj_dim_;
  DiTParallelLinear to_q_{nullptr};
  DiTParallelLinear to_k_{nullptr};
  DiTParallelLinear to_v_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  DiTParallelLinear to_out_{nullptr};
  layer::RMSNorm norm_added_q_{nullptr};
  layer::RMSNorm norm_added_k_{nullptr};
  DiTParallelLinear to_add_out_{nullptr};
  DiTParallelLinear to_add_q_{nullptr};
  DiTParallelLinear to_add_k_{nullptr};
  DiTParallelLinear to_add_v_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2Attention);

class Flux2ModulationImpl : public torch::nn::Module {
 public:
  explicit Flux2ModulationImpl(const ModelContext& context,
                               int64_t dim,
                               int64_t mod_param_sets,
                               bool bias = false)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();

    linear_ = register_module(
        "linear",
        layer::AddMatmul(dim, dim * 3 * mod_param_sets, bias, options_));
    act_fn_ = torch::nn::SiLU();
  }

  torch::Tensor forward(const torch::Tensor& temb) {
    auto mod = act_fn_->forward(temb);
    mod = linear_->forward(mod);
    return mod;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

  static std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
  split(const torch::Tensor& mod, int64_t mod_param_sets) {
    torch::Tensor mod_reshaped;
    if (mod.dim() == 2) {
      mod_reshaped = mod.unsqueeze(1);
    } else {
      mod_reshaped = mod;
    }

    auto mod_params = torch::chunk(mod_reshaped, 3 * mod_param_sets, -1);

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    for (int64_t i = 0; i < mod_param_sets; ++i) {
      int64_t start_idx = 3 * i;
      auto param_tuple = std::make_tuple(mod_params[start_idx],
                                         mod_params[start_idx + 1],
                                         mod_params[start_idx + 2]);
      result.push_back(param_tuple);
    }

    return result;
  }

 private:
  layer::AddMatmul linear_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2Modulation);

class Flux2TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  Flux2TimestepEmbeddingImpl(int64_t in_channels,
                             int64_t time_embed_dim,
                             int64_t out_dim = -1,
                             bool sample_proj_bias = true)
      : options_(torch::dtype(torch::kFloat32)) {
    linear_1_ = register_module(
        "linear_1",
        layer::AddMatmul(
            in_channels, time_embed_dim, sample_proj_bias, options_));

    act_ = register_module("act", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim > 0) ? out_dim : time_embed_dim;
    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, options_));
  }

  torch::Tensor forward(const torch::Tensor& sample) {
    torch::Tensor result = sample;

    result = linear_1_->forward(result);

    if (act_) {
      result = act_->forward(result);
    }

    result = linear_2_->forward(result);
    return result;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  torch::TensorOptions options_;
  layer::AddMatmul linear_1_{nullptr};
  torch::nn::SiLU act_{nullptr};
  layer::AddMatmul linear_2_{nullptr};
};
TORCH_MODULE(Flux2TimestepEmbedding);

class Flux2TimestepsImpl : public torch::nn::Module {
 public:
  Flux2TimestepsImpl(int64_t num_channels,
                     bool flip_sin_to_cos = true,
                     float downscale_freq_shift = 0.0,
                     int64_t scale = 1)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  num_channels_,
                                  flip_sin_to_cos_,
                                  downscale_freq_shift_,
                                  scale_);
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
  int64_t scale_;

  torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                       int embedding_dim,
                                       bool flip_sin_to_cos = false,
                                       float downscale_freq_shift = 1.0f,
                                       float scale = 1.0f,
                                       int max_period = 10000) {
    int half_dim = embedding_dim / 2;
    auto exponent = -std::log(static_cast<float>(max_period)) *
                    torch::arange(0,
                                  half_dim,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(timesteps.device()));
    exponent = exponent / (half_dim - downscale_freq_shift);

    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
    emb = scale * emb;
    emb = torch::cat({torch::sin(emb), torch::cos(emb)}, /*dim=*/-1);

    if (flip_sin_to_cos) {
      emb = torch::cat({emb.slice(/*dim=*/-1, /*start=*/half_dim),
                        emb.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)},
                       /*dim=*/-1);
    }

    if (embedding_dim % 2 == 1) {
      emb = torch::nn::functional::pad(
          emb, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
    }

    return emb;
  }
};
TORCH_MODULE(Flux2Timesteps);

class Flux2TimestepGuidanceEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit Flux2TimestepGuidanceEmbeddingsImpl(const ModelContext& context,
                                               int64_t embedding_dim,
                                               bool bias = false,
                                               bool guidance_embeds = true)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    in_channels_ = model_args.timestep_guidance_channels();
    embedding_dim_ = embedding_dim;
    guidance_embeds_ = guidance_embeds;

    time_proj_ =
        register_module("time_proj", Flux2Timesteps(in_channels_, true, 0.0));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        Flux2TimestepEmbedding(in_channels_, embedding_dim_, -1, bias));
    if (guidance_embeds_) {
      guidance_embedder_ = register_module(
          "guidance_embedder",
          Flux2TimestepEmbedding(in_channels_, embedding_dim_, -1, bias));
    }
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj.to(timestep.dtype()));

    if (guidance_embeds_ && guidance.defined()) {
      auto guidance_proj = time_proj_->forward(guidance);
      auto guidance_emb =
          guidance_embedder_->forward(guidance_proj.to(guidance.dtype()));
      return timesteps_emb + guidance_emb;
    } else {
      return timesteps_emb;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    if (guidance_embeds_) {
      guidance_embedder_->load_state_dict(
          state_dict.get_dict_with_prefix("guidance_embedder."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    if (guidance_embeds_) {
      guidance_embedder_->verify_loaded_weights(prefix + "guidance_embedder.");
    }
  }

 private:
  int64_t in_channels_;
  int64_t embedding_dim_;
  bool guidance_embeds_;
  Flux2Timesteps time_proj_{nullptr};
  Flux2TimestepEmbedding timestep_embedder_{nullptr};
  Flux2TimestepEmbedding guidance_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2TimestepGuidanceEmbeddings);

class Flux2TransformerBlockImpl : public torch::nn::Module {
 public:
  explicit Flux2TransformerBlockImpl(const ModelContext& context,
                                     int64_t dim,
                                     const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    auto eps = model_args.eps();
    LOG(INFO) << "-----------eps:" << eps;
    LOG(INFO) << "-----------dim:" << dim;

    norm1_ = register_module(
        "norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    norm1_context_ = register_module(
        "norm1_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    attn_ = register_module("attn", Flux2Attention(context, parallel_args));

    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_ = register_module("ff", Flux2FeedForward(context, parallel_args));

    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_context_ =
        register_module("ff_context", Flux2FeedForward(context, parallel_args));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb_img,
      const torch::Tensor& temb_txt,
      const torch::Tensor& image_rotary_emb,
      const int flag_dit_timestep,
      const int i) {
    auto img_mod_params = Flux2ModulationImpl::split(temb_img, 2);
    auto txt_mod_params = Flux2ModulationImpl::split(temb_txt, 2);

    // LOG(INFO) << "-----------img_mod_params:" << img_mod_params;
    // LOG(INFO) << "-----------txt_mod_params:" << txt_mod_params;
    /*if (flag_dit_timestep == 0){
      torch::Tensor save_img_mod_params = img_mod_params.to(torch::kCPU);
      torch::save(save_img_mod_params,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor/04_10_save_img_mod_params_0.pt");
      torch::Tensor save_txt_mod_params = txt_mod_params.to(torch::kCPU);
      torch::save(save_txt_mod_params,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor/04_11_save_txt_mod_params_0.pt");
    }*/

    auto [shift_msa_img, scale_msa_img, gate_msa_img] = img_mod_params[0];
    auto [shift_mlp_img, scale_mlp_img, gate_mlp_img] = img_mod_params[1];

    auto [shift_msa_txt, scale_msa_txt, gate_msa_txt] = txt_mod_params[0];
    auto [shift_mlp_txt, scale_mlp_txt, gate_mlp_txt] = txt_mod_params[1];

    LOG(INFO) << "scale_msa_img shape" << scale_msa_img.sizes();
    LOG(INFO) << "shift_msa_img shape" << shift_msa_img.sizes();

    // if (flag_dit_timestep == 0 and i == 0){

    //   torch::Tensor save_scale_msa_img = scale_msa_img.to(torch::kCPU);
    //   torch::save(save_scale_msa_img,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor/04_10_save_scale_msa_img_0.pt");
    //   torch::Tensor save_shift_msa_img = shift_msa_img.to(torch::kCPU);
    //   torch::save(save_shift_msa_img,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor/04_11_save_shift_msa_img_0.pt");
    // }
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_pre_norm_hidden_states = hidden_states.to(torch::kCPU);
      torch::save(save_pre_norm_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_01_save_pre_norm_hidden_states_0.pt");
    }
    auto norm_hidden_states = norm1_->forward(hidden_states);
    // norm_hidden_states = (1 + scale_msa_img.unsqueeze({0, 1})) *
    // norm_hidden_states + shift_msa_img.unsqueeze({0, 1});
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_norm_hidden_states =
          norm_hidden_states.to(torch::kCPU);
      torch::save(save_norm_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_02_save_norm_hidden_states_0.pt");
      torch::Tensor save_scale_msa_img = scale_msa_img.to(torch::kCPU);
      torch::save(save_scale_msa_img,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_03_save_scale_msa_img_0.pt");
      torch::Tensor save_shift_msa_img = shift_msa_img.to(torch::kCPU);
      torch::save(save_shift_msa_img,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_04_save_shift_msa_img_0.pt");
    }
    norm_hidden_states =
        (1 + scale_msa_img) * norm_hidden_states + shift_msa_img;

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_norm_hidden_states =
          norm_hidden_states.to(torch::kCPU);
      torch::save(save_norm_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_05_save_norm_hidden_states_0.pt");
      torch::Tensor save_encoder_hidden_states =
          encoder_hidden_states.to(torch::kCPU);
      torch::save(save_encoder_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_06_save_encoder_hidden_states_0.pt");
    }

    auto norm_encoder_hidden_states =
        norm1_context_->forward(encoder_hidden_states);
    // norm_encoder_hidden_states = (1 + scale_msa_txt.unsqueeze({0, 1})) *
    // norm_encoder_hidden_states + shift_msa_txt.unsqueeze({0, 1});
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_pre_norm_encoder_hidden_states =
          norm_encoder_hidden_states.to(torch::kCPU);
      torch::save(save_pre_norm_encoder_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_07_save_pre_norm_encoder_hidden_states_0.pt");
      torch::Tensor save_scale_msa_txt = scale_msa_txt.to(torch::kCPU);
      torch::save(save_scale_msa_txt,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_08_save_scale_msa_txt_0.pt");
      torch::Tensor save_shift_msa_txt = shift_msa_txt.to(torch::kCPU);
      torch::save(save_shift_msa_txt,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_09_save_shift_msa_txt_0.pt");
    }
    norm_encoder_hidden_states =
        (1 + scale_msa_txt) * norm_encoder_hidden_states + shift_msa_txt;

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_norm_encoder_hidden_states =
          norm_encoder_hidden_states.to(torch::kCPU);
      torch::save(save_norm_encoder_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_10_norm_encoder_hidden_states_0.pt");
    }

    auto [attn_output, context_attn_output] =
        attn_->forward(norm_hidden_states,
                       norm_encoder_hidden_states,
                       image_rotary_emb,
                       flag_dit_timestep,
                       i);

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_attn_output = attn_output.to(torch::kCPU);
      torch::save(save_attn_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_11_attn_output_0.pt");
      torch::Tensor save_context_attn_output =
          context_attn_output.to(torch::kCPU);
      torch::save(save_context_attn_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_12_context_attn_output_0.pt");
    }
    attn_output = gate_msa_img * attn_output;
    torch::Tensor new_hidden_states = hidden_states + attn_output;
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_new_hidden_states = new_hidden_states.to(torch::kCPU);
      torch::save(save_new_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_13_save_new_hidden_states_0.pt");
    }

    auto norm_hs = norm2_->forward(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp_img) + shift_mlp_img;
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_norm_hs = norm_hs.to(torch::kCPU);
      torch::save(save_norm_hs,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_14_save_norm_hs_0.pt");
    }

    auto ff_output =
        ff_->forward(norm_hs, flag_dit_timestep, i, /*is_save*/ true);
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_ff_output = ff_output.to(torch::kCPU);
      torch::save(save_ff_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_19_save_ff_output_0.pt");
    }

    new_hidden_states = new_hidden_states + gate_mlp_img * ff_output;

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_new_hidden_states = new_hidden_states.to(torch::kCPU);
      torch::save(save_new_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_15_save_new_hidden_states_0.pt");
    }

    context_attn_output = gate_msa_txt * context_attn_output;
    torch::Tensor new_encoder_hidden_states =
        encoder_hidden_states + context_attn_output;

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_new_encoder_hidden_states =
          new_encoder_hidden_states.to(torch::kCPU);
      torch::save(save_new_encoder_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_16_save_new_encoder_hidden_states_0.pt");
    }
    auto norm_enc_hs = norm2_context_->forward(new_encoder_hidden_states);
    norm_enc_hs = norm_enc_hs * (1 + scale_mlp_txt) + shift_mlp_txt;
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_norm_enc_hs = norm_enc_hs.to(torch::kCPU);
      torch::save(save_norm_enc_hs,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_17_save_norm_enc_hs_0.pt");
    }

    auto ff_context_out = ff_context_->forward(
        norm_enc_hs, flag_dit_timestep, i, /*is_save*/ false);
    new_encoder_hidden_states =
        new_encoder_hidden_states + gate_mlp_txt * ff_context_out;

    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor save_new_encoder_hidden_states =
          new_encoder_hidden_states.to(torch::kCPU);
      torch::save(save_new_encoder_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/04_double_stream_transformer_tensor/"
                  "04_18_save_new_encoder_hidden_states_0.pt");
    }
    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
      new_encoder_hidden_states =
          torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
    }

    return std::make_tuple(new_encoder_hidden_states, new_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    ff_context_->load_state_dict(
        state_dict.get_dict_with_prefix("ff_context."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    ff_->verify_loaded_weights(prefix + "ff.");
    ff_context_->verify_loaded_weights(prefix + "ff_context.");
  }

 private:
  torch::nn::LayerNorm norm1_{nullptr};
  torch::nn::LayerNorm norm1_context_{nullptr};
  Flux2Attention attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  Flux2FeedForward ff_{nullptr};
  torch::nn::LayerNorm norm2_context_{nullptr};
  Flux2FeedForward ff_context_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2TransformerBlock);

class Flux2ParallelSelfAttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2ParallelSelfAttentionImpl(const ModelContext& context,
                                          const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    mlp_ratio_ = model_args.mlp_ratio();
    mlp_hidden_dim_ = static_cast<int64_t>(query_dim_ * mlp_ratio_);
    mlp_mult_factor_ = 2;

    int64_t fused_out_dim = query_dim_ * 3 + mlp_hidden_dim_ * mlp_mult_factor_;

    LinearType linear_type = LinearType::Default;
    std::optional<TpOptions> tp_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      linear_type = LinearType::TensorParallel;
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*is_save=*/true,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto qkv_mlp = DiTParallelLinearFactory::create(query_dim_,
                                                    fused_out_dim,
                                                    false,
                                                    options_,
                                                    linear_type,
                                                    std::nullopt,
                                                    tp_options);
    to_qkv_mlp_ = register_module("to_qkv_mlp", qkv_mlp);

    mlp_act_fn_ = register_module("mlp_act_fn", Flux2SwiGLU());
    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

    LinearType linear_out_type = LinearType::Default;
    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (FLAGS_dit_tp_size > 1) {
      linear_out_type = LinearType::TensorParallel;
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*tp_rank=*/parallel_args_.rank_,
          /*tp_size=*/FLAGS_dit_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/false,
          /*is_save=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_out = DiTParallelLinearFactory::create(query_dim_ + mlp_hidden_dim_,
                                                   out_dim_,
                                                   false,
                                                   options_,
                                                   linear_out_type,
                                                   std::nullopt,
                                                   tp_out_options);
    to_out_ = register_module("to_out", to_out);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& image_rotary_emb,
                        int64_t flag_dit_timestep,
                        int64_t i) {
    LOG(INFO) << "Flux2ParallelSelfAttentionImpl forward";
    LOG(INFO) << "hidden_states shape" << hidden_states.sizes();
    int64_t batch_size = hidden_states.size(0);

    bool is_save = false;

    if (flag_dit_timestep == 0 && i == 0) {
      is_save = true;
    }
    LOG(INFO) << "0000000000000 hidden_states:" << hidden_states.sizes();
    auto weight_qkv_and_mlp = to_qkv_mlp_->get_weight();
    LOG(INFO) << "weight_qkv_and_mlp shape" << weight_qkv_and_mlp.sizes();
    if (weight_qkv_and_mlp.defined()) {
      int64_t tp_rank = parallel_args_.rank_;
      int64_t tp_size = FLAGS_dit_tp_size > 1 ? FLAGS_dit_tp_size : 1;

      int64_t qkv_size = query_dim_;
      int64_t mlp_size = mlp_hidden_dim_ * mlp_mult_factor_;
      LOG(INFO) << "qkv_size" << qkv_size;
      LOG(INFO) << "mlp_size" << mlp_size;
      auto qkv_weight = weight_qkv_and_mlp.slice(0, 0, qkv_size * 3);
      auto mlp_weight =
          weight_qkv_and_mlp.slice(0, qkv_size * 3, qkv_size * 3 + mlp_size);

      if (tp_size > 1) {
        auto qkv_chunks = qkv_weight.chunk(3, 0);
        auto q_weight = qkv_chunks[0];
        auto k_weight = qkv_chunks[1];
        auto v_weight = qkv_chunks[2];
        LOG(INFO) << "q_weight shape" << q_weight.sizes();
        LOG(INFO) << "k_weight shape" << k_weight.sizes();
        LOG(INFO) << "v_weight shape" << v_weight.sizes();
        LOG(INFO) << "mlp_weight shape" << mlp_weight.sizes();
        torch::Tensor save_q_weight = q_weight.to(torch::kCPU);
        torch::save(save_q_weight,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/01_single_parallel_weight/"
                    "only_qkv_and_mlp/01_01_save_qkv_mlp_output_0.pt");
        torch::Tensor save_k_weight = k_weight.to(torch::kCPU);
        torch::save(save_k_weight,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/01_single_parallel_weight/"
                    "only_qkv_and_mlp/01_02_save_qkv_mlp_output_0.pt");
        torch::Tensor save_v_weight = v_weight.to(torch::kCPU);
        torch::save(save_v_weight,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/01_single_parallel_weight/"
                    "only_qkv_and_mlp/01_03_save_qkv_mlp_output_0.pt");
        torch::Tensor save_mlp_weight = mlp_weight.to(torch::kCPU);
        torch::save(save_mlp_weight,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/01_single_parallel_weight/"
                    "only_qkv_and_mlp/01_04_save_qkv_mlp_output_0.pt");

        auto q_weight_rank = q_weight.chunk(tp_size, 0)[tp_rank];
        auto k_weight_rank = k_weight.chunk(tp_size, 0)[tp_rank];
        auto v_weight_rank = v_weight.chunk(tp_size, 0)[tp_rank];
        auto mlp_weight_rank = mlp_weight.chunk(tp_size, 0)[tp_rank];

        auto rank_weight = torch::cat(
            {q_weight_rank, k_weight_rank, v_weight_rank, mlp_weight_rank}, 0);

        // Directly set the weight instead of using load_state_dict
        LOG(INFO) << "rank_weight shape" << rank_weight.sizes();
        to_qkv_mlp_->set_weight(rank_weight);
      }
    }
    auto qkv_mlp_output = to_qkv_mlp_->forward(hidden_states, is_save);

    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "-------------11111111qkv_mlp_output shape"
                << qkv_mlp_output.sizes();
      torch::Tensor save_qkv_mlp_output = qkv_mlp_output.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_qkv_mlp_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_01_save_qkv_mlp_output_0.pt");
    }
    auto tp_size = FLAGS_dit_tp_size >= 1 ? FLAGS_dit_tp_size : 1;
    int64_t qkv_size = query_dim_ * 3 / tp_size;

    LOG(INFO) << "qkv size" << qkv_size;
    LOG(INFO) << "tp size" << tp_size;

    int64_t mlp_size = mlp_hidden_dim_ * mlp_mult_factor_ / tp_size;

    LOG(INFO) << "mlp size" << mlp_size;
    auto qkv_output = qkv_mlp_output.slice(-1, 0, qkv_size);
    auto mlp_output = qkv_mlp_output.slice(-1, qkv_size, qkv_size + mlp_size);

    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "-------------2222222222qkv_output shape"
                << qkv_output.sizes();
      torch::Tensor save_qkv_output = qkv_output.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_qkv_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_02_save_qkv_output_0.pt");
      LOG(INFO) << "-------------33333333333mlp_output shape"
                << mlp_output.sizes();
      torch::Tensor save_mlp_output = mlp_output.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_mlp_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_03_save_mlp_output_0.pt");
    }

    auto q = qkv_output.slice(-1, 0, qkv_size / 3);
    auto k = qkv_output.slice(-1, qkv_size / 3, (qkv_size / 3) * 2);
    auto v = qkv_output.slice(-1, (qkv_size / 3) * 2, (qkv_size / 3) * 3);
    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "-------------44444444444q shape" << q.sizes();
      torch::Tensor save_q = q.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_q,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_04_save_q_0.pt");
      LOG(INFO) << "-------------55555555555k shape" << k.sizes();
      torch::Tensor save_k = k.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_k,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_05_save_k_0.pt");
      LOG(INFO) << "-------------66666666666v shape" << v.sizes();
      torch::Tensor save_v = v.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_v,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_06_save_v_0.pt");
    }
    LOG(INFO) << "q shape" << q.sizes();
    LOG(INFO) << "k shape" << k.sizes();
    LOG(INFO) << "v shape" << v.sizes();

    int64_t inner_dim = k.size(-1);
    int64_t attn_heads = inner_dim / head_dim_;
    int64_t head_dim = head_dim_;

    q = q.view({batch_size, -1, attn_heads, head_dim});
    k = k.view({batch_size, -1, attn_heads, head_dim});
    v = v.view({batch_size, -1, attn_heads, head_dim});

    if (norm_q_) q = std::get<0>(norm_q_->forward(q));
    if (norm_k_) k = std::get<0>(norm_k_->forward(k));
    // if (flag_dit_timestep == 0 and i == 0){
    //   LOG(INFO) << "-------------77777777777q shape" << q.sizes();
    //   torch::Tensor save_q = q.to(torch::kCPU);
    //   // LOG(INFO) << "-------------parallel_args_.rank_:" <<
    //   parallel_args_.rank_; torch::save(save_q,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/08_07_save_q_0.pt");
    //   LOG(INFO) << "-------------888888888888k shape" << k.sizes();
    //   torch::Tensor save_k = k.to(torch::kCPU);
    //   // LOG(INFO) << "-------------parallel_args_.rank_:" <<
    //   parallel_args_.rank_; torch::save(save_k,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/08_08_save_k_0.pt");
    // }
    if (image_rotary_emb.defined()) {
      q = apply_rotary_emb(q, image_rotary_emb, false);
      k = apply_rotary_emb(k, image_rotary_emb, false);
    }

    LOG(INFO) << "after image rotary";
    LOG(INFO) << "q shape" << q.sizes();
    LOG(INFO) << "k shape" << k.sizes();

#if defined(USE_NPU)
    int64_t head_num_ = q.size(2);
    int64_t head_dim_ = q.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(q,
                                                         k,
                                                         v,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);

    attn_output = attn_output.reshape({batch_size, -1, attn_heads * head_dim});

#elif defined(USE_CUDA)
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        q, k, v, torch::nullopt, 0.0, false);
    attn_output = attn_output.transpose(1, 2).reshape(
        {batch_size, -1, attn_heads * head_dim});
#else
    NOT_IMPLEMENTED();
#endif

    attn_output = attn_output.to(q.dtype());
    mlp_output = mlp_act_fn_(mlp_output);
    if (flag_dit_timestep == 0 and i == 0) {
      LOG(INFO) << "999999999999999single inner parallel attention attn output"
                << attn_output.sizes();
      torch::Tensor save_attn_output = attn_output.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_attn_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_09_save_attn_output_0.pt");
      LOG(INFO) << "10-10-10-10-10single inner parallel attention mlp output"
                << mlp_output.sizes();
      torch::Tensor save_mlp_output = mlp_output.to(torch::kCPU);
      // LOG(INFO) << "-------------parallel_args_.rank_:" <<
      // parallel_args_.rank_;
      torch::save(save_mlp_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/"
                  "08_cpp_single_inner_parallelattention_tensor/rank" +
                      std::to_string(parallel_args_.rank_) +
                      "/08_10_save_mlp_output_0.pt");
    }
    auto output =
        torch::cat(std::vector<torch::Tensor>{attn_output, mlp_output}, -1);

    output = to_out_->forward(output);
    // if (flag_dit_timestep == 0 and i == 0){
    //   LOG(INFO) << "11-11-11-11-11single inner parallel attention final
    //   output" << output.sizes(); torch::Tensor save_output =
    //   output.to(torch::kCPU);
    //   // LOG(INFO) << "-------------parallel_args_.rank_:" <<
    //   parallel_args_.rank_; torch::save(save_output,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank"
    //   + std::to_string(parallel_args_.rank_) + "/08_11_save_output_0.pt");
    // }

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    // auto fused_weight = state_dict.get_tensor("to_qkv_mlp_proj.");
    to_qkv_mlp_->load_state_dict(
        state_dict.get_dict_with_prefix("to_qkv_mlp_proj."));
    // if (fused_weight.defined()) {
    //   int64_t tp_rank = parallel_args_.rank_;
    //   int64_t tp_size = FLAGS_dit_tp_size > 1 ? FLAGS_dit_tp_size : 1;

    //   int64_t qkv_size = query_dim_;
    //   int64_t mlp_size = mlp_hidden_dim_ * mlp_mult_factor_;

    //   auto qkv_weight = fused_weight.slice(0, 0, qkv_size * 3);
    //   auto mlp_weight =
    //       fused_weight.slice(0, qkv_size * 3, qkv_size * 3 + mlp_size);

    //   if (tp_size > 1) {
    //     auto qkv_chunks = qkv_weight.chunk(3, 0);
    //     auto q_weight = qkv_chunks[0];
    //     auto k_weight = qkv_chunks[1];
    //     auto v_weight = qkv_chunks[2];

    //     auto q_weight_rank = q_weight.chunk(tp_size, 0)[tp_rank];
    //     auto k_weight_rank = k_weight.chunk(tp_size, 0)[tp_rank];
    //     auto v_weight_rank = v_weight.chunk(tp_size, 0)[tp_rank];
    //     auto mlp_weight_rank = mlp_weight.chunk(tp_size, 0)[tp_rank];

    //     auto rank_weight = torch::cat(
    //         {q_weight_rank, k_weight_rank, v_weight_rank, mlp_weight_rank},
    //         0);

    //     // Directly set the weight instead of using load_state_dict
    //     LOG(INFO) << "rank_weight shape" << rank_weight.sizes();
    //     to_qkv_mlp_->set_weight(rank_weight);

    //       torch::Tensor save_rank_weight = rank_weight.to(torch::kCPU);
    // torch::save(save_rank_weight,
    // "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/01_single_parallel_weight/rank"
    // + std::to_string(parallel_args_.rank_) +
    // "/01_single_parallel_qkvmlp_weight.pt");

    //   } else {
    //     // Directly set the weight instead of using load_state_dict
    //     to_qkv_mlp_->set_weight(fused_weight);
    //   }
    // }
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    to_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_out.");
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  float mlp_ratio_;
  int64_t mlp_hidden_dim_;
  int64_t mlp_mult_factor_;
  DiTParallelLinear to_qkv_mlp_{nullptr};
  Flux2SwiGLU mlp_act_fn_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  DiTParallelLinear to_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2ParallelSelfAttention);

class Flux2SingleTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit Flux2SingleTransformerBlockImpl(const ModelContext& context,
                                           int64_t inner_dim,
                                           const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();

    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({inner_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));

    attn_ = register_module("attn",
                            Flux2ParallelSelfAttention(context, parallel_args));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb_mod,
                        const torch::Tensor& image_rotary_emb,
                        bool split_hidden_states,
                        int64_t text_seq_len,
                        int64_t flag_dit_timestep,
                        int64_t i) {
    auto mod_params = Flux2ModulationImpl::split(temb_mod, 1);
    auto [shift, scale, gate] = mod_params[0];

    auto norm_hidden_states = norm_->forward(hidden_states);
    norm_hidden_states = (1 + scale) * norm_hidden_states + shift;
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor dit_inner_single_norm_hidden_states =
          norm_hidden_states.to(torch::kCPU);
      torch::save(dit_inner_single_norm_hidden_states,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/07_cpp_single_stream_transformer_tensor/"
                  "07_01_dit_inner_single_norm_hidden_states_0.pt");
    }
    auto attn_output = attn_->forward(
        norm_hidden_states, image_rotary_emb, flag_dit_timestep, i);
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor dit_inner_single_attn_output = attn_output.to(torch::kCPU);
      torch::save(dit_inner_single_attn_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/07_cpp_single_stream_transformer_tensor/"
                  "07_02_dit_inner_single_attn_output_0.pt");
    }
    auto output = hidden_states + gate * attn_output;
    if (flag_dit_timestep == 0 and i == 0) {
      torch::Tensor dit_inner_single_output = output.to(torch::kCPU);
      torch::save(dit_inner_single_output,
                  "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                  "dump_flux2_tensor/07_cpp_single_stream_transformer_tensor/"
                  "07_03_dit_inner_single_output_0.pt");
    }
    if (output.dtype() == torch::kFloat16) {
      output = torch::clamp(output, -65504.0f, 65504.0f);
    }

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
  }

 private:
  torch::nn::LayerNorm norm_{nullptr};
  Flux2ParallelSelfAttention attn_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2SingleTransformerBlock);

class Flux2Transformer2DModelImpl : public torch::nn::Module {
 public:
  explicit Flux2Transformer2DModelImpl(const ModelContext& context,
                                       const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto joint_attention_dim = model_args.joint_attention_dim();
    auto num_layers = model_args.num_layers();
    auto num_single_layers = model_args.num_single_layers();
    auto patch_size = model_args.patch_size();
    auto rope_theta = model_args.rope_theta();
    auto axes_dims_rope = model_args.axes_dims_rope();
    in_channels_ = model_args.in_channels();
    out_channels_ = model_args.out_channels();

    auto inner_dim = num_attention_heads * attention_head_dim;

    time_guidance_embed_ =
        Flux2TimestepGuidanceEmbeddings(context, inner_dim, false, true);
    register_module("time_guidance_embed", time_guidance_embed_);

    double_stream_modulation_img_ =
        register_module("double_stream_modulation_img",
                        Flux2Modulation(context, inner_dim, 2, false));
    double_stream_modulation_txt_ =
        register_module("double_stream_modulation_txt",
                        Flux2Modulation(context, inner_dim, 2, false));
    single_stream_modulation_ =
        register_module("single_stream_modulation",
                        Flux2Modulation(context, inner_dim, 1, false));

    x_embedder_ = register_module(
        "x_embedder",
        layer::AddMatmul(in_channels_, inner_dim, false, options_));
    context_embedder_ = register_module(
        "context_embedder",
        layer::AddMatmul(joint_attention_dim, inner_dim, false, options_));

    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());

    transformer_block_layers_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = Flux2TransformerBlock(context, inner_dim, parallel_args);
      transformer_blocks_->push_back(block);
      transformer_block_layers_.push_back(block);
    }

    single_transformer_block_layers_.reserve(num_single_layers);
    for (int64_t i = 0; i < num_single_layers; ++i) {
      auto block =
          Flux2SingleTransformerBlock(context, inner_dim, parallel_args);
      single_transformer_blocks_->push_back(block);
      single_transformer_block_layers_.push_back(block);
    }

    norm_out_ =
        register_module("norm_out", AdaLayerNormContinuous(context, false));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(inner_dim,
                         patch_size * patch_size * out_channels_,
                         false,
                         options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& image_rotary_emb,
                        const int flag_dit_timestep) {
    // if (flag_dit_timestep == 0){
    //   torch::Tensor pre_double_hidden_states = hidden_states.to(torch::kCPU);
    //   torch::save(pre_double_hidden_states,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor/04_14_pre_double_hidden_states_0.pt");
    // }
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);

    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    // if (flag_dit_timestep == 0){

    //   torch::Tensor embedd_hidden_states = hidden_states.to(torch::kCPU);
    //   torch::save(embedd_hidden_states,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/03_01_embedd_hidden_states_0.pt");
    //   torch::Tensor embedd_encoder_hidden_states =
    //   encoder_hidden_states.to(torch::kCPU);
    //   torch::save(embedd_encoder_hidden_states,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/03_02_embedd_encoder_hidden_states_0.pt");
    // }

    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    auto guidance_scaled = guidance.defined()
                               ? guidance.to(hidden_states.dtype()) * 1000.0f
                               : torch::Tensor();
    auto temb = time_guidance_embed_->forward(timestep_scaled, guidance_scaled);

    // if (flag_dit_timestep == 0){
    //   torch::Tensor save_temb = temb.to(torch::kCPU);
    //   torch::save(save_temb,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_05_save_temb_0.pt");
    // }
    auto double_stream_mod_img = double_stream_modulation_img_->forward(temb);
    auto double_stream_mod_txt = double_stream_modulation_txt_->forward(temb);
    auto single_stream_mod = single_stream_modulation_->forward(temb);

    LOG(INFO) << "变量double_stream_mod_img的类型是: "
              << typeid(double_stream_mod_img).name();

    // if (flag_dit_timestep == 0){
    //   torch::Tensor pre_double_double_stream_mod_img =
    //   double_stream_mod_img.to(torch::kCPU);
    //   torch::save(pre_double_double_stream_mod_img,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_06_pre_double_para_double_stream_mod_img_0.pt");
    //   torch::Tensor pre_double_double_stream_mod_txt =
    //   double_stream_mod_txt.to(torch::kCPU);
    //   torch::save(pre_double_double_stream_mod_txt,
    //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_07_pre_double_para_double_stream_mod_txt_0.pt");
    //   }

    /*if (flag_dit_timestep == 0){
      torch::Tensor pre_double_hidden_states = hidden_states.to(torch::kCPU);
      torch::save(pre_double_hidden_states,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_01_pre_double_para_hidden_states_0.pt");

      torch::Tensor pre_double_encoder_hidden_states =
    encoder_hidden_states.to(torch::kCPU);
      torch::save(pre_double_encoder_hidden_states,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_02_pre_double_para_encoder_hidden_states_0.pt");

      torch::Tensor pre_double_double_stream_mod_img =
    double_stream_mod_img.to(torch::kCPU);
      torch::save(pre_double_double_stream_mod_img,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_03_pre_double_para_double_stream_mod_img_0.pt");

      torch::Tensor pre_double_double_stream_mod_txt =
    double_stream_mod_txt.to(torch::kCPU);
      torch::save(pre_double_double_stream_mod_txt,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/05_cpp_pre_double_para_tensor/05_04_pre_double_para_double_stream_mod_txt_0.pt");

    }*/

    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      // if (flag_dit_timestep == 0 and i == 0){
      //   torch::Tensor pre_double_hidden_states =
      //   hidden_states.to(torch::kCPU); torch::save(pre_double_hidden_states,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/02_cpp_pre_double_transformer/02_01_pre_double_hidden_states_0.pt");

      //   torch::Tensor pre_double_encoder_hidden_states =
      //   encoder_hidden_states.to(torch::kCPU);
      //   torch::save(pre_double_encoder_hidden_states,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/02_cpp_pre_double_transformer/02_02_pre_double_para_encoder_hidden_states_0.pt");

      //   torch::Tensor pre_double_double_stream_mod_img =
      //   double_stream_mod_img.to(torch::kCPU);
      //   torch::save(pre_double_double_stream_mod_img,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/02_cpp_pre_double_transformer/02_03_pre_double_para_double_stream_mod_img_0.pt");

      //   torch::Tensor pre_double_double_stream_mod_txt =
      //   double_stream_mod_txt.to(torch::kCPU);
      //   torch::save(pre_double_double_stream_mod_txt,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/02_cpp_pre_double_transformer/02_04_pre_double_para_double_stream_mod_txt_0.pt");

      // }
      auto block = transformer_block_layers_[i];
      auto [new_encoder_hidden, new_hidden] =
          block->forward(hidden_states,
                         encoder_hidden_states,
                         double_stream_mod_img,
                         double_stream_mod_txt,
                         image_rotary_emb,
                         flag_dit_timestep,
                         i);

      if (flag_dit_timestep == 0 and i == 0) {
        torch::Tensor pre_double_new_hidden = new_hidden.to(torch::kCPU);
        torch::save(pre_double_new_hidden,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/02_cpp_pre_double_transformer/"
                    "02_05_pre_double_new_hidden_0.pt");

        torch::Tensor pre_double_new_encoder_hidden =
            new_encoder_hidden.to(torch::kCPU);
        torch::save(pre_double_new_encoder_hidden,
                    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/"
                    "dump_flux2_tensor/02_cpp_pre_double_transformer/"
                    "02_06_pre_double_new_encoder_hidden_0.pt");
      }
      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }

    hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);
    /*if (flag_dit_timestep == 0){
      torch::Tensor dit_inner_double_hidden_states =
    hidden_states.to(torch::kCPU); torch::save(dit_inner_double_hidden_states,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/03_cpp_dit_inner_tensor/03_01_dit_inner_double_hidden_states_0.pt");
    }*/
    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      // if (flag_dit_timestep == 0 and i == 0){
      //   torch::Tensor dit_inner_double_hidden_states =
      //   hidden_states.to(torch::kCPU);
      //   torch::save(dit_inner_double_hidden_states,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/07_cpp_single_stream_transformer_tensor/07_01_dit_inner_output_concat_hidden_states_0.pt");
      // }
      auto block = single_transformer_block_layers_[i];
      hidden_states = block->forward(hidden_states,
                                     single_stream_mod,
                                     image_rotary_emb,
                                     false,
                                     0,
                                     flag_dit_timestep,
                                     i);

      // if (flag_dit_timestep == 0 and i == 0){
      //   torch::Tensor dit_inner_single_hidden_states =
      //   hidden_states.to(torch::kCPU);
      //   torch::save(dit_inner_single_hidden_states,
      //   "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/07_cpp_single_stream_transformer_tensor/07_02_dit_inner_single_output_hidden_states_0.pt");
      // }
    }

    int64_t start = encoder_hidden_states.size(1);
    int64_t length = hidden_states.size(1) - start;
    auto output_hidden =
        hidden_states.narrow(1, start, std::max(length, int64_t(0)));
    hidden_states = output_hidden;

    auto output_hidden_final = norm_out_->forward(hidden_states, temb);
    auto final_return_proj = proj_out_->forward(output_hidden_final);
    /*if (flag_dit_timestep == 0){
      torch::Tensor dit_inner_norm_out = output_hidden_final.to(torch::kCPU);
      torch::save(dit_inner_norm_out,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/03_cpp_dit_inner_tensor/03_03_dit_inner_norm_out_0.pt");
      torch::Tensor dit_inner_final_return_proj =
    final_return_proj.to(torch::kCPU); torch::save(dit_inner_final_return_proj,
    "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/03_cpp_dit_inner_tensor/03_04_dit_inner_final_return_proj_0.pt");
    }*/

    return proj_out_->forward(output_hidden_final);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      context_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("context_embedder."));
      x_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("x_embedder."));
      time_guidance_embed_->load_state_dict(
          state_dict->get_dict_with_prefix("time_guidance_embed."));
      double_stream_modulation_img_->load_state_dict(
          state_dict->get_dict_with_prefix("double_stream_modulation_img."));
      double_stream_modulation_txt_->load_state_dict(
          state_dict->get_dict_with_prefix("double_stream_modulation_txt."));
      single_stream_modulation_->load_state_dict(
          state_dict->get_dict_with_prefix("single_stream_modulation."));
      for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
        auto block = transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        auto block = single_transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    context_embedder_->verify_loaded_weights(prefix + "context_embedder.");
    x_embedder_->verify_loaded_weights(prefix + "x_embedder.");
    time_guidance_embed_->verify_loaded_weights(prefix +
                                                "time_guidance_embed.");
    double_stream_modulation_img_->verify_loaded_weights(
        prefix + "double_stream_modulation_img.");
    double_stream_modulation_txt_->verify_loaded_weights(
        prefix + "double_stream_modulation_txt.");
    single_stream_modulation_->verify_loaded_weights(
        prefix + "single_stream_modulation.");
    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "single_transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

  int64_t in_channels() { return in_channels_; }

 private:
  int64_t in_channels_;
  int64_t out_channels_;
  layer::AddMatmul context_embedder_{nullptr};
  layer::AddMatmul x_embedder_{nullptr};
  Flux2TimestepGuidanceEmbeddings time_guidance_embed_{nullptr};
  Flux2Modulation double_stream_modulation_img_{nullptr};
  Flux2Modulation double_stream_modulation_txt_{nullptr};
  Flux2Modulation single_stream_modulation_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<Flux2TransformerBlock> transformer_block_layers_;
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  std::vector<Flux2SingleTransformerBlock> single_transformer_block_layers_;
  AdaLayerNormContinuous norm_out_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2Transformer2DModel);

class Flux2DiTModelImpl : public torch::nn::Module {
 public:
  explicit Flux2DiTModelImpl(const ModelContext& context,
                             const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    flux2_transformer_2d_model_ =
        register_module("flux2_transformer_2d_model_",
                        Flux2Transformer2DModel(context, parallel_args));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& image_rotary_emb,
                        const int flag_dit_timestep) {
    torch::Tensor output =
        flux2_transformer_2d_model_->forward(hidden_states_input,
                                             encoder_hidden_states_input,
                                             timestep,
                                             guidance,
                                             image_rotary_emb,
                                             flag_dit_timestep);
    return output;
  }
  int64_t in_channels() { return flux2_transformer_2d_model_->in_channels(); }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux2_transformer_2d_model_->load_model(std::move(loader));
    flux2_transformer_2d_model_->verify_loaded_weights("");
  }

 private:
  Flux2Transformer2DModel flux2_transformer_2d_model_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2DiTModel);

REGISTER_MODEL_ARGS(Flux2Transformer2DModel, [&] {
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{32, 32, 32, 32}));
  LOAD_ARG_OR(eps, "eps", 1e-6);
  LOAD_ARG_OR(in_channels, "in_channels", 128);
  LOAD_ARG_OR(joint_attention_dim, "joint_attention_dim", 15360);
  LOAD_ARG_OR(mlp_ratio, "mlp_ratio", 3.0f);
  LOAD_ARG_OR(num_layers, "num_layers", 8);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 48);
  LOAD_ARG_OR(out_channels, "out_channels", 128);
  LOAD_ARG_OR(patch_size, "patch_size", 1);
  LOAD_ARG_OR(rope_theta, "rope_theta", 2000.0f);
  LOAD_ARG_OR(timestep_guidance_channels, "timestep_guidance_channels", 256);
});

}  // namespace xllm
