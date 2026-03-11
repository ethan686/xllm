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
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "models/dit/transformer_flux.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {
/*
inline torch::Tensor apply_rotary_emb(const torch::Tensor& x,
                                      const torch::Tensor& freqs_cis,
                                      bool head_first = true) {
  torch::Tensor cos;
  torch::Tensor sin;
  if (head_first) {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(1);
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(1);
  } else {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(2);
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(2);
  }

#if defined(USE_NPU)
  return at_npu::native::custom_ops::npu_rotary_mul(x, cos, sin, "interleave");
#elif defined(USE_CUDA)
  std::vector<int64_t> reshape_shape;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    reshape_shape.push_back(x.size(i));
  }
  reshape_shape.push_back(-1);
  reshape_shape.push_back(2);
  torch::Tensor reshaped = x.reshape(reshape_shape);
  torch::Tensor x_real = reshaped.select(-1, 0);
  torch::Tensor x_imag = reshaped.select(-1, 1);
  torch::Tensor neg_x_imag = -x_imag;
  auto x_rotated = torch::stack({neg_x_imag, x_real}, -1).flatten(3);
  return (x.to(torch::kFloat32) * cos.to(torch::kFloat32) +
          x_rotated.to(torch::kFloat32) * sin.to(torch::kFloat32))
      .to(x.dtype());
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  CHECK_EQ(dim % 2, 0) << "Dimension must be even";

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);
#if defined(USE_NPU)
  freqs_outer = freqs_outer.to(torch::kFloat32);
#endif
  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);
    auto sin_vals = torch::sin(freqs_outer);

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);
  }
  LOG(FATAL) << "get_1d_rotary_pos_embed returned empty tensor, which should "
                "not happen. use_real: "
             << use_real
             << " repeat_interleave_real: " << repeat_interleave_real;
  return torch::Tensor();
}
*/
class Flux2SwiGLUImpl : public torch::nn::Module {
public:
    Flux2SwiGLUImpl() {
        gate_fn_ = torch::nn::SiLU();
    }

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
  explicit Flux2FeedForwardImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto eps = model_args.mlp_ratio();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;

    linear_in_ = register_module(
        "linear_in",
        layer::AddMatmul(inner_dim, inner_dim_ * 6, false, options_));
    act_fn_ = register_module("act_fn", Flux2SwiGLU());
    linear_out_ = register_module(
        "linear_out",
        layer::AddMatmul(inner_dim_ * 3, dim_, false, options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto out = linear_in_->forward(hidden_states);
    out = act_fn_->forward(out);
    out = linear_out_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_in_->load_state_dict(state_dict.get_dict_with_prefix("linear_in.weight"));
    linear_out_->load_state_dict(state_dict.get_dict_with_prefix("linear_out.weight"));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_in_->verify_loaded_weights(prefix + "linear_in.weight");
    linear_out_->verify_loaded_weights(prefix + "linear_out.weight");
  }

 private:
  int64_t dim_;
  int64_t inner_dim_;
  float mult_;
  layer::AddMatmul linear_in_{nullptr};
  Flux2SwiGLU act_fn_{nullptr};
  layer::AddMatmul linear_out_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2FeedForward);

class Flux2AttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2AttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    added_kv_proj_dim_ = query_dim_;

    to_out_ = register_module(
        "to_out",
        layer::AddMatmul(out_dim_, query_dim_, /*with_bias=*/true, options_));

    fused_qkv_ = register_module(
        "fused_qkv",
        layer::FusedAddMatmul(
            query_dim_, 3 * out_dim_, /*with_bias=*/true, options_));

    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

    if (added_kv_proj_dim_ > 0) {
      to_add_out_ = register_module(
        "to_add_out",
        layer::AddMatmul(
            out_dim_, added_kv_proj_dim_, /*with_bias=*/true, options_));
      norm_added_q_ = register_module(
          "norm_added_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
      norm_added_k_ = register_module(
          "norm_added_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

      fused_add_qkv_ = register_module(
        "fused_add_qkv",
        layer::FusedAddMatmul(
            added_kv_proj_dim_, 3 * out_dim_, /*with_bias=*/true, options_));
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& image_rotary_emb) {
    int64_t input_ndim = hidden_states.dim();

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
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);

    auto qkv = fused_qkv_->forward(hidden_states_reshaped);

    auto chunks = qkv.chunk(3, -1);
    torch::Tensor query = chunks[0];
    torch::Tensor key = chunks[1];
    torch::Tensor value = chunks[2];

    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = heads_;

    int64_t head_dim = inner_dim / attn_heads;
    query = query.view({batch_size, -1, attn_heads, head_dim});
    key = key.view({batch_size, -1, attn_heads, head_dim});
    value = value.view({batch_size, -1, attn_heads, head_dim});
    if (norm_q_) query = std::get<0>(norm_q_->forward(query));
    if (norm_k_) key = std::get<0>(norm_k_->forward(key));

    auto encoder_qkv = fused_add_qkv_->forward(encoder_hidden_states_reshaped);

    auto encoder_chunks = encoder_qkv.chunk(3, -1);
    torch::Tensor encoder_hidden_states_query_proj = encoder_chunks[0];
    torch::Tensor encoder_hidden_states_key_proj = encoder_chunks[1];
    torch::Tensor encoder_hidden_states_value_proj = encoder_chunks[2];

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          std::get<0>(norm_added_q_->forward(encoder_hidden_states_query_proj));

    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          std::get<0>(norm_added_k_->forward(encoder_hidden_states_key_proj));
    // TODO some are right some are wrong query1& key1.
    // encoder_hidden_states_query_proj
    auto query1 = torch::cat({encoder_hidden_states_query_proj, query}, 1);
    auto key1 = torch::cat({encoder_hidden_states_key_proj, key}, 1);
    auto value1 = torch::cat({encoder_hidden_states_value_proj, value}, 1);
    if (image_rotary_emb.defined()) {
      query1 = apply_rotary_emb(query1, image_rotary_emb, false);
      key1 = apply_rotary_emb(key1, image_rotary_emb, false);
    }
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
    attn_output = attn_output.to(query.dtype());

    int64_t encoder_length = encoder_hidden_states_reshaped.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);
    hidden_output = to_out_->forward(hidden_output);
    encoder_output = to_add_out_->forward(encoder_output);
    return std::make_tuple(hidden_output, encoder_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    fused_qkv_->load_state_dict(state_dict, {"to_q", "to_k", "to_v"});
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));

    if (added_kv_proj_dim_ > 0) {
      norm_added_q_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_q."));
      norm_added_k_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_k."));
      fused_add_qkv_->load_state_dict(state_dict,
                                    {"add_q_proj", "add_k_proj", "add_v_proj"});
      to_add_out_->load_state_dict(
          state_dict.get_dict_with_prefix("to_add_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fused_qkv_->verify_loaded_weights(prefix + "to_q|to_k|to_v.");
    to_out_->verify_loaded_weights(prefix + "to_out.0.");

    if (added_kv_proj_dim_ > 0) {
      fused_add_qkv_->verify_loaded_weights(prefix +
                                          "add_q_proj|add_k_proj|add_v_proj.");
      to_add_out_->verify_loaded_weights(prefix + "to_add_out.");
    }
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  int64_t added_kv_proj_dim_;
  layer::FusedAddMatmul fused_qkv_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  layer::AddMatmul to_out_{nullptr};
  layer::RMSNorm norm_added_q_{nullptr};
  layer::RMSNorm norm_added_k_{nullptr};
  layer::FusedAddMatmul fused_add_qkv_{nullptr};
  layer::AddMatmul to_add_out_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2Attention);

class Flux2ModulationImpl : public torch::nn::Module {
 public:
  explicit Flux2ModulationImpl(const ModelContext& context, int64_t dim, int64_t mod_param_sets, bool bias = false)
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
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear.weight"));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.weight");
  }

  static std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> split(
      const torch::Tensor& mod, int64_t mod_param_sets) {
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
      auto param_tuple = std::make_tuple(
          mod_params[start_idx],
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
        layer::AddMatmul(in_channels, time_embed_dim, sample_proj_bias, options_));

    act_ = register_module("act", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim > 0) ? out_dim : time_embed_dim;
    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(time_embed_dim, time_embed_dim_out, sample_proj_bias, options_));
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

torch::Tensor get_timestep_embedding(
    const torch::Tensor& timesteps,
    int embedding_dim,
    bool flip_sin_to_cos = false,
    float downscale_freq_shift = 1.0f,
    float scale = 1.0f,
    int max_period = 10000
) {
    int half_dim = embedding_dim / 2;
    auto exponent = -std::log(static_cast<float>(max_period)) * 
        torch::arange(0, half_dim, torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(timesteps.device()));
    exponent = exponent / (half_dim - downscale_freq_shift);

    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
    emb = scale * emb;
    emb = torch::cat({torch::sin(emb), torch::cos(emb)}, /*dim=*/-1);

    if (flip_sin_to_cos) {
        emb = torch::cat({
            emb.slice(/*dim=*/-1, /*start=*/half_dim),
            emb.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)
        }, /*dim=*/-1);
    }

    if (embedding_dim % 2 == 1) {
        emb = torch::nn::functional::pad(emb, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
    }

    return emb;
}
};
TORCH_MODULE(Flux2Timesteps);

class Flux2TimestepGuidanceEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit Flux2TimestepGuidanceEmbeddingsImpl(const ModelContext& context, int64_t embedding_dim, bool bias = false, bool guidance_embeds = true)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    in_channels_ = model_args.timestep_guidance_channels();
    embedding_dim_ = embedding_dim;
    guidance_embeds_ = guidance_embeds;

    time_proj_ = register_module(
        "time_proj",
        Flux2Timesteps(in_channels_, true, 0.0));
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
    auto timesteps_emb = timestep_embedder_->forward(timesteps_proj.to(timestep.dtype()));

    if (guidance_embeds_ && guidance.defined()) {
      auto guidance_proj = time_proj_->forward(guidance);
      auto guidance_emb = guidance_embedder_->forward(guidance_proj.to(guidance.dtype()));
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
  explicit Flux2TransformerBlockImpl(const ModelContext& context, int64_t dim)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto eps = model_args.eps();

    norm1_ = register_module(
        "norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));

    norm1_context_ = register_module(
        "norm1_context",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));

    attn_ = register_module("attn", Flux2Attention(context));

    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));

    ff_ = register_module("ff", Flux2FeedForward(context));

    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));

    ff_context_ = register_module("ff_context", Flux2FeedForward(context));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb_img,
      const torch::Tensor& temb_txt,
      const torch::Tensor& image_rotary_emb) {
    auto img_mod_params = Flux2ModulationImpl::split(temb_img, 2);
    auto txt_mod_params = Flux2ModulationImpl::split(temb_txt, 2);

    auto [shift_msa_img, scale_msa_img, gate_msa_img] = img_mod_params[0];
    auto [shift_mlp_img, scale_mlp_img, gate_mlp_img] = img_mod_params[1];

    auto [shift_msa_txt, scale_msa_txt, gate_msa_txt] = txt_mod_params[0];
    auto [shift_mlp_txt, scale_mlp_txt, gate_mlp_txt] = txt_mod_params[1];

    auto norm_hidden_states = norm1_->forward(hidden_states);
    // norm_hidden_states = (1 + scale_msa_img.unsqueeze({0, 1})) * norm_hidden_states + shift_msa_img.unsqueeze({0, 1});
    norm_hidden_states = (1 + scale_msa_img.unsqueeze(0).unsqueeze(1)) * norm_hidden_states + shift_msa_img.unsqueeze(0).unsqueeze(1);

    auto norm_encoder_hidden_states = norm1_context_->forward(encoder_hidden_states);
    // norm_encoder_hidden_states = (1 + scale_msa_txt.unsqueeze({0, 1})) * norm_encoder_hidden_states + shift_msa_txt.unsqueeze({0, 1});
    norm_encoder_hidden_states = (1 + scale_msa_txt.unsqueeze(0).unsqueeze(1)) * norm_encoder_hidden_states + shift_msa_txt.unsqueeze(0).unsqueeze(1);

    auto [attn_output, context_attn_output] = attn_->forward(
        norm_hidden_states,
        norm_encoder_hidden_states,
        image_rotary_emb
    );

    attn_output = gate_msa_img.unsqueeze(0).unsqueeze(1) * attn_output;
    torch::Tensor new_hidden_states = hidden_states + attn_output;

    auto norm_hs = norm2_->forward(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp_img.unsqueeze(0).unsqueeze(1)) + shift_mlp_img.unsqueeze(0).unsqueeze(1);
    auto ff_output = ff_->forward(norm_hs);
    new_hidden_states = new_hidden_states + gate_mlp_img.unsqueeze(0).unsqueeze(1) * ff_output;

    context_attn_output = gate_msa_txt.unsqueeze(0).unsqueeze(1) * context_attn_output;
    torch::Tensor new_encoder_hidden_states = encoder_hidden_states + context_attn_output;

    auto norm_enc_hs = norm2_context_->forward(new_encoder_hidden_states);
    norm_enc_hs = norm_enc_hs * (1 + scale_mlp_txt.unsqueeze(0).unsqueeze(1)) + shift_mlp_txt.unsqueeze(0).unsqueeze(1);
    auto ff_context_out = ff_context_->forward(norm_enc_hs);
    new_encoder_hidden_states = new_encoder_hidden_states + gate_mlp_txt.unsqueeze(0).unsqueeze(1) * ff_context_out;

    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
        new_encoder_hidden_states = torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
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
};
TORCH_MODULE(Flux2TransformerBlock);

class Flux2ParallelSelfAttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2ParallelSelfAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    mlp_ratio_ = model_args.mlp_ratio();
    mlp_hidden_dim_ = static_cast<int64_t>(query_dim_ * mlp_ratio_);
    mlp_mult_factor_ = 2;

    to_qkv_mlp_proj_ = register_module(
        "to_qkv_mlp_proj",
        layer::AddMatmul(query_dim_, query_dim_ * 3 + mlp_hidden_dim_ * mlp_mult_factor_, false, options_));
    mlp_act_fn_ = register_module("mlp_act_fn", Flux2SwiGLU());
    norm_q_ = register_module(
        "norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ = register_module(
        "norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));
    to_out_ = register_module(
        "to_out",
        layer::AddMatmul(query_dim_ + mlp_hidden_dim_, out_dim_, false, options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                     const torch::Tensor& image_rotary_emb) {
    int64_t batch_size = hidden_states.size(0);

    auto hidden_states_proj = to_qkv_mlp_proj_->forward(hidden_states);
    auto qkv_mlp = torch::split(hidden_states_proj, {query_dim_ * 3, mlp_hidden_dim_ * mlp_mult_factor_}, -1);

    auto qkv = qkv_mlp[0];
    auto mlp_hidden_states = qkv_mlp[1];

    // auto [q, k, v] = torch::chunk(qkv, 3, -1);
    auto qkv_chunks = torch::chunk(qkv, 3, -1);
    auto q = qkv_chunks[0];
    auto k = qkv_chunks[1];  
    auto v = qkv_chunks[2];


    int64_t inner_dim = k.size(-1);
    int64_t attn_heads = heads_;
    int64_t head_dim = inner_dim / attn_heads;

    q = q.view({batch_size, -1, attn_heads, head_dim});
    k = k.view({batch_size, -1, attn_heads, head_dim});
    v = v.view({batch_size, -1, attn_heads, head_dim});

    if (norm_q_) q = std::get<0>(norm_q_->forward(q));
    if (norm_k_) k = std::get<0>(norm_k_->forward(k));

    if (image_rotary_emb.defined()) {
      q = apply_rotary_emb(q, image_rotary_emb, false);
      k = apply_rotary_emb(k, image_rotary_emb, false);
    }

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

    mlp_hidden_states = mlp_act_fn_->forward(mlp_hidden_states);

    //auto output = torch::cat({attn_output, mlp_hidden_states}, -1);
    auto output = torch::cat(std::vector<torch::Tensor>{attn_output, mlp_hidden_states}, -1);
    output = to_out_->forward(output);

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    to_qkv_mlp_proj_->load_state_dict(state_dict.get_dict_with_prefix("to_qkv_mlp_proj."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_qkv_mlp_proj_->verify_loaded_weights(prefix + "to_qkv_mlp_proj.");
    to_out_->verify_loaded_weights(prefix + "to_out.");
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  float mlp_ratio_;
  int64_t mlp_hidden_dim_;
  int64_t mlp_mult_factor_;
  layer::AddMatmul to_qkv_mlp_proj_{nullptr};
  Flux2SwiGLU mlp_act_fn_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  layer::AddMatmul to_out_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2ParallelSelfAttention);

class Flux2SingleTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit Flux2SingleTransformerBlockImpl(const ModelContext& context, int64_t inner_dim)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();

    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({inner_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));

    attn_ = register_module("attn", Flux2ParallelSelfAttention(context));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb_mod,
      const torch::Tensor& image_rotary_emb,
      bool split_hidden_states = false,
      int64_t text_seq_len = 0) {
    auto mod_params = Flux2ModulationImpl::split(temb_mod, 1);
    auto [shift, scale, gate] = mod_params[0];

    auto norm_hidden_states = norm_->forward(hidden_states);
    norm_hidden_states = (1 + scale) * norm_hidden_states + shift;

    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);
    auto output = hidden_states + gate * attn_output;

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
};
TORCH_MODULE(Flux2SingleTransformerBlock);

class Flux2Transformer2DModelImpl : public torch::nn::Module {
 public:
  explicit Flux2Transformer2DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
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

    time_guidance_embed_ = Flux2TimestepGuidanceEmbeddings(context, inner_dim, false, true);
    register_module("time_guidance_embed", time_guidance_embed_);

    double_stream_modulation_img_ = register_module("double_stream_modulation_img", Flux2Modulation(context, inner_dim, 2, false));
    double_stream_modulation_txt_ = register_module("double_stream_modulation_txt", Flux2Modulation(context, inner_dim, 2, false));
    single_stream_modulation_ = register_module("single_stream_modulation", Flux2Modulation(context, inner_dim, 1, false));

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
      auto block = Flux2TransformerBlock(context, inner_dim);
      transformer_blocks_->push_back(block);
      transformer_block_layers_.push_back(block);
    }

    single_transformer_block_layers_.reserve(num_single_layers);
    for (int64_t i = 0; i < num_single_layers; ++i) {
      auto block = Flux2SingleTransformerBlock(context, inner_dim);
      single_transformer_blocks_->push_back(block);
      single_transformer_block_layers_.push_back(block);
    }

    norm_out_ = register_module("norm_out", AdaLayerNormContinuous(context));
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
                        const torch::Tensor& image_rotary_emb) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    auto guidance_scaled = guidance.defined() ? guidance.to(hidden_states.dtype()) * 1000.0f : torch::Tensor();
    auto temb = time_guidance_embed_->forward(timestep_scaled, guidance_scaled);

    auto double_stream_mod_img = double_stream_modulation_img_->forward(temb);
    auto double_stream_mod_txt = double_stream_modulation_txt_->forward(temb);
    auto single_stream_mod = single_stream_modulation_->forward(temb);

    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      auto [new_encoder_hidden, new_hidden] = block->forward(
          hidden_states, encoder_hidden_states, double_stream_mod_img, double_stream_mod_txt, image_rotary_emb);
      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }

    hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);

    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      hidden_states = block->forward(hidden_states, single_stream_mod,
                                    image_rotary_emb, false, 0);
    }

    int64_t start = encoder_hidden_states.size(1);
    int64_t length = hidden_states.size(1) - start;
    auto output_hidden =
        hidden_states.narrow(1, start, std::max(length, int64_t(0)));
    hidden_states = output_hidden;

    auto output_hidden_final = norm_out_->forward(hidden_states, temb);
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
    time_guidance_embed_->verify_loaded_weights(prefix + "time_guidance_embed.");
    double_stream_modulation_img_->verify_loaded_weights(prefix + "double_stream_modulation_img.");
    double_stream_modulation_txt_->verify_loaded_weights(prefix + "double_stream_modulation_txt.");
    single_stream_modulation_->verify_loaded_weights(prefix + "single_stream_modulation.");
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
};
TORCH_MODULE(Flux2Transformer2DModel);

class Flux2DiTModelImpl : public torch::nn::Module {
 public:
  explicit Flux2DiTModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    flux2_transformer_2d_model_ = register_module(
        "flux2_transformer_2d_model_", Flux2Transformer2DModel(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& image_rotary_emb) {
    torch::Tensor output =
        flux2_transformer_2d_model_->forward(hidden_states_input,
                                            encoder_hidden_states_input,
                                            timestep,
                                            guidance,
                                            image_rotary_emb);
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
};
TORCH_MODULE(Flux2DiTModel);

REGISTER_MODEL_ARGS(Flux2Transformer2DModel, [&] {

  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(axes_dims_rope, "axes_dims_rope",
              (std::vector<int64_t>{32, 32, 32, 32}));
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
