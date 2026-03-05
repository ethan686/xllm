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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <regex>
#include <unordered_map>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_mistral_decoder_layer_impl.h"
#include "llm_model_base.h"
#include "models/model_registry.h"
#include "xllm/core/layers/common/add_matmul.h"
#include "xllm/core/layers/common/attention_mask.h"

namespace xllm {

// ==================== Utility Functions ====================

/**
 * @brief SiLU (Sigmoid Linear Unit) activation function
 * @param x Input tensor
 * @return x * sigmoid(x)
 */
inline torch::Tensor silu(const torch::Tensor& x) {
  return x * torch::sigmoid(x);
}

// ==================== Rotary Position Embedding ====================

/**
 * @brief Rotary Position Embedding (RoPE) implementation for Mistral models
 * 
 * RoPE rotates query and key vectors based on position information,
 * allowing the model to capture relative position dependencies.
 */
class MistralRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  /**
   * @brief Constructor
   * @param head_dim Dimension of each attention head
   * @param rope_theta Base for the rotational frequency (default 10000.0)
   */
  explicit MistralRotaryEmbeddingImpl(int64_t head_dim, double rope_theta = 10000.0) 
      : attention_scaling_(1.0f) {
    TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, 
                "head_dim must be positive and even, got ", head_dim);
    
    int64_t dim = head_dim;
    std::vector<float> inv_freq_vec;
    inv_freq_vec.reserve(dim / 2);
    
    for (int i = 0; i < dim / 2; ++i) {
      inv_freq_vec.push_back(1.0f / std::pow(rope_theta, 2.0 * i / dim));
    }
    
    inv_freq_ = register_buffer(
        "inv_freq", 
        torch::tensor(inv_freq_vec)
            .reshape({1, 1, -1})  // [1, 1, dim/2]
            .to(torch::kFloat32));
  }

  /**
   * @brief Forward pass - compute cos and sin embeddings for positions
   * @param x Input tensor (used for dtype and device info)
   * @param position_ids Position indices [batch_size, seq_len]
   * @return Pair of (cos, sin) embeddings
   */
  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x, 
      const torch::Tensor& position_ids) const {
    
    TORCH_CHECK(position_ids.dim() == 2, 
                "position_ids must be 2D [batch, seq_len], got ", position_ids.dim(), "D");
    
    // Expand inv_freq to match batch size
    auto inv_freq_expanded = inv_freq_.expand(
        {position_ids.size(0), -1, -1}).to(torch::kFloat32);
    
    // Prepare position IDs for matrix multiplication
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);
    
    // Compute frequencies: [batch, 1, dim/2] @ [batch, 1, seq_len] -> [batch, dim/2, seq_len]
    auto freqs = torch::matmul(inv_freq_expanded, position_ids_expanded)
                    .transpose(1, 2);  // [batch, seq_len, dim/2]
    
    // Duplicate frequencies to match head dimension
    auto emb = torch::cat({freqs, freqs}, -1);  // [batch, seq_len, dim]
    
    // Compute cos and sin with scaling
    auto cos = emb.cos() * attention_scaling_;
    auto sin = emb.sin() * attention_scaling_;
    
    return {cos.to(x.dtype()), sin.to(x.dtype())};
  }

 private:
  torch::Tensor inv_freq_;
  float attention_scaling_;
};
TORCH_MODULE(MistralRotaryEmbedding);

// ==================== Decoder Layer ====================

/**
 * @brief Single Mistral decoder layer wrapper
 * 
 * This is a thin wrapper around NPU-optimized decoder layer implementation.
 */
class MistralDecoderLayerImpl : public LlmDecoderLayerImplBase<layer::NpuMistralDecoderLayer> {
 public:
  MistralDecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : LlmDecoderLayerImplBase<layer::NpuMistralDecoderLayer>(context, layer_id) {
    decoder_layer_ = register_module(
        "decoder_layer", 
        layer::NpuMistralDecoderLayer(context));
  }

  torch::Tensor forward(
      torch::Tensor& x,
      torch::Tensor& m_cos_pos,
      torch::Tensor& m_sin_pos,
      torch::Tensor& cu_seq_len,
      std::vector<int>& cu_seq_len_vec,
      ModelInputParams& input_params,
      int node_id) {
    
    TORCH_CHECK(decoder_layer_ != nullptr, "Decoder layer not initialized");
    
    return decoder_layer_->forward(
        x, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec, 
        input_params, node_id);
  }

  // Weight management
  void load_state_dict(const StateDict& state_dict) {
    TORCH_CHECK(decoder_layer_ != nullptr, "Decoder layer not initialized");
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    TORCH_CHECK(decoder_layer_ != nullptr, "Decoder layer not initialized");
    decoder_layer_->verify_loaded_weights();
  }
  
  void merge_loaded_weights() {
    TORCH_CHECK(decoder_layer_ != nullptr, "Decoder layer not initialized");
    decoder_layer_->merge_loaded_weights();
  }

 private:
  layer::NpuMistralDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(MistralDecoderLayer);

// ==================== Main Mistral Model ====================

/**
 * @brief Main Mistral language model implementation
 * 
 * Implements the core transformer architecture with:
 * - Token embedding
 * - Stack of decoder layers
 * - Final layer normalization
 * - Rotary position embeddings
 */
class MistralModelImpl : public LlmModelImplBase<MistralDecoderLayer> {
 public:
  struct Output {
    torch::Tensor last_hidden_state;
    std::shared_ptr<Cache> past_key_values;
    std::vector<torch::Tensor> hidden_states;
    std::vector<torch::Tensor> attentions;
  };

  explicit MistralModelImpl(const ModelContext& context)
      : LlmModelImplBase<MistralDecoderLayer>("mistral", context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    
    // Validate arguments
    TORCH_CHECK(model_args.vocab_size() > 0, "vocab_size must be positive");
    TORCH_CHECK(model_args.mm_hidden_size() > 0, "hidden_size must be positive");
    TORCH_CHECK(model_args.mm_num_hidden_layers() > 0, "num_layers must be positive");
    
    hidden_size_ = model_args.mm_hidden_size();
    
    // 1. Token embedding layer
    embed_tokens_ = register_module(
        "embed_tokens",
        torch::nn::Embedding(
            torch::nn::EmbeddingOptions(model_args.vocab_size(), hidden_size_)
                .padding_idx(model_args.pad_token_id())));
    
    if (options.has_value()) {
      embed_tokens_->weight.set_data(embed_tokens_->weight.to(options.value()));
    }
    
    // 2. Decoder layers
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.mm_num_hidden_layers());
    
    for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); ++i) {
      auto layer = MistralDecoderLayer(context, i);
      layers_.push_back(layer);
      blocks_->push_back(layer);
    }
    
    // 3. Final layer normalization
    norm_ = register_module(
        "norm",
        torch::nn::RMSNorm(torch::nn::RMSNormOptions(hidden_size_)
                               .eps(model_args.mm_layer_norm_eps())));
    
    // 4. Rotary position embeddings
    rotary_emb_ = register_module(
        "rotary_emb", 
        MistralRotaryEmbedding(
            model_args.mm_head_dim(), 
            model_args.mm_rope_theta()));
    
    // 5. Attention mask layer (lazy initialization)
    attention_mask_layer_ = std::make_shared<layer::AttentionMask>();
  }

  Output forward(
      torch::Tensor input_ids,
      torch::Tensor attention_mask,
      torch::Tensor position_ids,
      std::shared_ptr<Cache>& past_key_values,
      torch::Tensor inputs_embeds,
      bool use_cache = true,
      bool output_hidden_states = false,
      bool output_attentions = false,
      torch::Tensor cache_position = {}) {
    
    // 1. Get hidden states
    torch::Tensor hidden_states;
    if (inputs_embeds.defined()) {
        hidden_states = inputs_embeds;
    } else {
        hidden_states = embed_tokens_->forward(input_ids);
    }
    
    const auto batch_size = hidden_states.size(0);
    const auto seq_len = hidden_states.size(1);
    
    // 2. Prepare position IDs
    if (!position_ids.defined()) {
        int64_t past_len = (past_key_values && use_cache) 
            ? past_key_values->get_seq_length() : 0;
        position_ids = torch::arange(seq_len, hidden_states.device())
                           .unsqueeze(0)
                           .expand({batch_size, -1})
                           .add(past_len);
    }
    
    // 3. Get rotary position embeddings
    auto [cos, sin] = rotary_emb_->forward(hidden_states, position_ids);
    
    // 4. Prepare cu_seq_len for variable sequences
    torch::Tensor cu_seq_len;
    std::vector<int> cu_seq_len_vec(batch_size + 1, 0);
    
    if (attention_mask.defined()) {
        auto seq_lens = attention_mask.sum(-1);  // [batch_size]
        for (int i = 0; i < batch_size; ++i) {
            cu_seq_len_vec[i + 1] = cu_seq_len_vec[i] + seq_lens[i].item<int>();
        }
    } else {
        for (int i = 0; i <= batch_size; ++i) {
            cu_seq_len_vec[i] = i * seq_len;
        }
    }
    cu_seq_len = torch::tensor(cu_seq_len_vec, torch::kInt32)
                    .to(hidden_states.device());
    
    // 5. Prepare input parameters
    ModelInputParams input_params;
    input_params.batch_size = batch_size;
    input_params.num_tokens = cu_seq_len_vec.back();
    input_params.is_prefill = !past_key_values || past_key_values->empty();
    input_params.use_cache = use_cache;
    // 设置其他必要参数...
    
    // 6. Process through decoder layers
    std::vector<torch::Tensor> all_hidden_states;
    if (output_hidden_states) {
        all_hidden_states.reserve(layers_.size() + 1);
        all_hidden_states.push_back(hidden_states);
    }
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        hidden_states = layers_[i]->forward(
            hidden_states,
            cos,
            sin,
            cu_seq_len,
            cu_seq_len_vec,
            input_params,
            i
        );
        
        if (output_hidden_states) {
            all_hidden_states.push_back(hidden_states);
        }
    }
    
    // 7. Final normalization
    hidden_states = norm_->forward(hidden_states);
    
    return Output{
        hidden_states,
        past_key_values,
        std::move(all_hidden_states),
        {}  // attentions (empty for now)
    };
  }

  // ==================== Weight Management ====================

  void load_state_dict(const StateDict& state_dict) {
    // Load token embeddings
    weight::load_weight(state_dict,
                        "embed_tokens.weight",
                        embed_tokens_->weight,
                        is_embed_tokens_loaded_);
    
    // Load decoder layers
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    
    // Load final norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    TORCH_CHECK(is_embed_tokens_loaded_, 
                "weight is not loaded for ", prefix, "embed_tokens.weight");
    
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(
          prefix + "layers." + std::to_string(i) + ".");
    }
    
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  // ==================== Getters ====================

  torch::nn::Embedding get_embed_tokens() const { return embed_tokens_; }
  int64_t get_hidden_size() const { return hidden_size_; }

 private:
  // Status flags
  bool is_embed_tokens_loaded_ = false;
  
  // Model dimensions
  int64_t hidden_size_;
  
  // Model components
  torch::nn::Embedding embed_tokens_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MistralDecoderLayer> layers_;  // For direct access
  torch::nn::RMSNorm norm_{nullptr};
  MistralRotaryEmbedding rotary_emb_{nullptr};
  
  // Utility layers
  std::shared_ptr<layer::AttentionMask> attention_mask_layer_{nullptr};
};
TORCH_MODULE(MistralModel);

// ==================== KV Cache Implementation ====================

/**
 * @brief Simple KV cache implementation for Mistral models
 * 
 * Stores key and value tensors for each layer, allowing efficient
 * incremental decoding without recomputation.
 */
class MistralCache : public Cache {
 public:
  MistralCache() = default;
  
  // Prevent copying
  MistralCache(const MistralCache&) = delete;
  MistralCache& operator=(const MistralCache&) = delete;
  
  // Allow moving
  MistralCache(MistralCache&&) = default;
  MistralCache& operator=(MistralCache&&) = default;

  int64_t get_seq_length() const override {
    if (key_cache_.empty()) return 0;
    return key_cache_[0].size(2);  // Sequence length is at dim 2
  }

  std::pair<torch::Tensor, torch::Tensor> update(
      int64_t layer_idx,
      torch::Tensor key,
      torch::Tensor value,
      const std::unordered_map<std::string, torch::Tensor>& /*kwargs*/) override {
    
    TORCH_CHECK(layer_idx >= 0, "layer_idx must be non-negative");
    TORCH_CHECK(key.defined() && value.defined(), "key and value must be defined");
    
    // Resize caches if needed
    if (layer_idx >= static_cast<int64_t>(key_cache_.size())) {
      key_cache_.resize(layer_idx + 1);
      value_cache_.resize(layer_idx + 1);
    }
    
    // Update cache
    if (!key_cache_[layer_idx].defined()) {
      // First token - initialize cache
      key_cache_[layer_idx] = key;
      value_cache_[layer_idx] = value;
    } else {
      // Subsequent tokens - concatenate along sequence dimension
      key_cache_[layer_idx] = torch::cat({key_cache_[layer_idx], key}, 2);
      value_cache_[layer_idx] = torch::cat({value_cache_[layer_idx], value}, 2);
    }
    
    return {key_cache_[layer_idx], value_cache_[layer_idx]};
  }

  std::pair<torch::Tensor, torch::Tensor> get(int64_t layer_idx) const override {
    TORCH_CHECK(layer_idx >= 0 && layer_idx < static_cast<int64_t>(key_cache_.size()),
                "layer_idx ", layer_idx, " out of range [0, ", key_cache_.size(), ")");
    
    return {key_cache_[layer_idx], value_cache_[layer_idx]};
  }
  
  void clear() override {
    key_cache_.clear();
    value_cache_.clear();
  }
  
  bool empty() const override {
    return key_cache_.empty();
  }

 private:
  std::vector<torch::Tensor> key_cache_;
  std::vector<torch::Tensor> value_cache_;
};

} // namespace xllm