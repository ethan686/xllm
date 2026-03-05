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

#include <memory>
#include <vector>
#include <utility>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "llm_model_base.h"
#include "models/model_registry.h"
#include "mistral.h"

namespace xllm {

/**
 * @brief Output structure for Mistral3 model (without LM head)
 */
struct Mistral3ModelOutputWithPast {
  torch::Tensor last_hidden_state;           // Final layer output
  std::shared_ptr<Cache> past_key_values;    // KV cache for incremental decoding
  std::vector<torch::Tensor> hidden_states;  // All layers outputs (if output_hidden_states=True)
  std::vector<torch::Tensor> attentions;     // Attention weights (if output_attentions=True)
  std::vector<torch::Tensor> image_hidden_states;  // Always empty for text-only model
  
  // Default constructor
  Mistral3ModelOutputWithPast() = default;
  
  // Constructor for easy creation
  Mistral3ModelOutputWithPast(
      torch::Tensor last_hidden_state,
      std::shared_ptr<Cache> past_key_values,
      std::vector<torch::Tensor> hidden_states,
      std::vector<torch::Tensor> attentions,
      std::vector<torch::Tensor> image_hidden_states = {})
      : last_hidden_state(std::move(last_hidden_state))
      , past_key_values(std::move(past_key_values))
      , hidden_states(std::move(hidden_states))
      , attentions(std::move(attentions))
      , image_hidden_states(std::move(image_hidden_states)) {}
};

/**
 * @brief Mistral3 model (without LM head)
 * 
 * This model wraps the base MistralModel and provides Mistral3-specific
 * output structure and forward interface.
 */
class Mistral3ModelImpl : public LlmModelImplBase<MistralDecoderLayer> {
 public:
  explicit Mistral3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<MistralDecoderLayer>("mistral3", context.get_model_args()) {
    try {
      const auto& model_args = context.get_model_args();
      
      // Validate arguments
      validate_args(model_args);
      
      language_model_ = register_module("language_model", MistralModel(context));
      
      LOG(INFO) << "Mistral3Model initialized successfully";
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to initialize Mistral3Model: " << e.what();
      throw;
    }
  }

  /**
   * @brief Forward pass through the model
   * @param input_ids Token indices [batch_size, seq_len]
   * @param attention_mask Attention mask [batch_size, seq_len]
   * @param position_ids Position indices [batch_size, seq_len]
   * @param past_key_values KV cache for incremental decoding
   * @param inputs_embeds Pre-computed embeddings (optional)
   * @param use_cache Whether to return KV cache
   * @param output_attentions Whether to output attention weights
   * @param output_hidden_states Whether to output all hidden states
   * @param cache_position Position in cache for incremental decoding
   * @return Model output structure
   */
  Mistral3ModelOutputWithPast forward(
      const torch::Tensor& input_ids,
      const torch::Tensor& attention_mask,
      const torch::Tensor& position_ids,
      std::shared_ptr<Cache>& past_key_values,
      const torch::Tensor& inputs_embeds,
      bool use_cache = true,
      bool output_attentions = false,
      bool output_hidden_states = false,
      const torch::Tensor& cache_position = {}) {
    
    check_initialized();
    
    // Forward through base language model
    auto lm_outputs = language_model_->forward(
        input_ids, attention_mask, position_ids, past_key_values,
        inputs_embeds, use_cache, output_hidden_states, output_attentions,
        cache_position);
    
    // Convert to Mistral3 output format
    return Mistral3ModelOutputWithPast{
        lm_outputs.last_hidden_state,
        lm_outputs.past_key_values,
        std::move(lm_outputs.hidden_states),
        std::move(lm_outputs.attentions),
        {}  // empty image_hidden_states for text-only model
    };
  }

  /**
   * @brief Load state dictionary from pretrained weights
   * @param state_dict Dictionary containing model weights
   */
  void load_state_dict(const StateDict& state_dict) {
    check_initialized();
    language_model_->load_state_dict(
        state_dict.get_dict_with_prefix("model."));
    LOG(INFO) << "Mistral3Model state dict loaded successfully";
  }

  /**
   * @brief Verify that all weights are loaded
   * @param prefix Prefix for weight names
   */
  void verify_loaded_weights(const std::string& prefix) const {
    check_initialized();
    language_model_->verify_loaded_weights(prefix + "model.");
    LOG(INFO) << "Mistral3Model weights verified successfully";
  }

 private:
  void check_initialized() const {
    TORCH_CHECK(language_model_ != nullptr, 
                "Mistral3Model not initialized. Call constructor first.");
  }

  void validate_args(const ModelArgs& model_args) const {
    TORCH_CHECK(model_args.vocab_size() > 0, 
                "vocab_size must be positive, got ", model_args.vocab_size());
    TORCH_CHECK(model_args.mm_hidden_size() > 0, 
                "hidden_size must be positive, got ", model_args.mm_hidden_size());
    TORCH_CHECK(model_args.mm_num_hidden_layers() > 0, 
                "num_hidden_layers must be positive, got ", model_args.mm_num_hidden_layers());
  }

  MistralModel language_model_{nullptr};
};
TORCH_MODULE(Mistral3Model);

/**
 * @brief Output structure for Mistral3 with LM head
 */
struct Mistral3CausalLMOutputWithPast {
  torch::Tensor logits;                          // LM head output [batch_size, seq_len, vocab_size]
  std::shared_ptr<Cache> past_key_values;        // KV cache for incremental decoding
  std::vector<torch::Tensor> hidden_states;      // All layers outputs (if output_hidden_states=True)
  std::vector<torch::Tensor> attentions;         // Attention weights (if output_attentions=True)
  std::vector<torch::Tensor> image_hidden_states; // Always empty for text-only model
  
  // Default constructor
  Mistral3CausalLMOutputWithPast() = default;
  
  // Constructor for easy creation
  Mistral3CausalLMOutputWithPast(
      torch::Tensor logits,
      std::shared_ptr<Cache> past_key_values,
      std::vector<torch::Tensor> hidden_states,
      std::vector<torch::Tensor> attentions,
      std::vector<torch::Tensor> image_hidden_states = {})
      : logits(std::move(logits))
      , past_key_values(std::move(past_key_values))
      , hidden_states(std::move(hidden_states))
      , attentions(std::move(attentions))
      , image_hidden_states(std::move(image_hidden_states)) {}
};

/**
 * @brief Mistral3 model for conditional generation (text-only)
 * 
 * This class adds a language modeling head on top of the base Mistral3Model
 * for text generation tasks.
 */
class Mistral3ForConditionalGenerationImpl : public LlmForCausalLMImplBase<Mistral3Model> {
 public:
  explicit Mistral3ForConditionalGenerationImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Mistral3Model>(context) {
    try {
      const auto& model_args = context.get_model_args();
      const auto& options = context.get_tensor_options();
      
      // Validate arguments
      validate_args(model_args);
      
      LOG(INFO) << "Mistral3ForConditionalGeneration initialized with "
                << "vocab_size=" << model_args.vocab_size()
                << ", hidden_size=" << model_args.mm_hidden_size();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to initialize Mistral3ForConditionalGeneration: " << e.what();
      throw;
    }
  }

  /**
   * @brief Forward pass for conditional generation
   * @param input_ids Token indices [batch_size, seq_len]
   * @param attention_mask Attention mask [batch_size, seq_len]
   * @param position_ids Position indices [batch_size, seq_len]
   * @param past_key_values KV cache for incremental decoding
   * @param inputs_embeds Pre-computed embeddings (optional)
   * @param use_cache Whether to return KV cache
   * @param output_attentions Whether to output attention weights
   * @param output_hidden_states Whether to output all hidden states
   * @param cache_position Position in cache for incremental decoding
   * @param logits_to_keep Number of logits to keep (for memory efficiency)
   * @return Model output with logits
   */
  Mistral3CausalLMOutputWithPast forward(
      const torch::Tensor& input_ids,
      const torch::Tensor& attention_mask,
      const torch::Tensor& position_ids,
      std::shared_ptr<Cache>& past_key_values,
      const torch::Tensor& inputs_embeds,
      bool use_cache = true,
      bool output_attentions = false,
      bool output_hidden_states = false,
      const torch::Tensor& cache_position = {},
      int64_t logits_to_keep = 0) {
    
    check_initialized();
    
    // Forward through base model
    auto outputs = model_->forward(
        input_ids, attention_mask, position_ids, past_key_values,
        inputs_embeds, use_cache, output_attentions, output_hidden_states,
        cache_position);
    
    // Compute logits from last_hidden_state
    torch::Tensor logits = compute_logits(outputs.last_hidden_state, logits_to_keep);
    
    return Mistral3CausalLMOutputWithPast{
        std::move(logits),
        outputs.past_key_values,
        std::move(outputs.hidden_states),
        std::move(outputs.attentions),
        {}  // empty image_hidden_states
    };
  }

  /**
   * @brief Load state dictionary from pretrained weights
   * @param state_dict Dictionary containing model weights
   */
  void load_state_dict(const StateDict& state_dict) {
    check_initialized();
    model_->load_state_dict(state_dict->get_dict_with_prefix("language_model."));
    lm_head_->load_state_dict(state_dict->get_dict_with_prefix("language_model.lm_head."));
    LOG(INFO) << "Mistral3ForConditionalGeneration state dict loaded successfully";
  }

  /**
   * @brief Verify that all weights are loaded
   * @param prefix Prefix for weight names
   */
  void verify_loaded_weights(const std::string& prefix) const {
    check_initialized();
    model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights(prefix + "lm_head.");
    LOG(INFO) << "Mistral3ForConditionalGeneration weights verified successfully";
  }

  /**
   * @brief Load model from a DiT folder loader
   * @param loader Folder loader containing model weights
   */
  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    LOG(INFO) << "Loading Mistral3ForConditionalGeneration from ModelLoader...";
    
    check_initialized();
    
    try {
      for (const auto& state_dict : loader->get_state_dicts()) {
        model_->load_state_dict(state_dict->get_dict_with_prefix("language_model."));
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("language_model.lm_head."));
      }
      
      model_->verify_loaded_weights("language_model.");
      lm_head_->verify_loaded_weights("language_model.lm_head.");
      
      LOG(INFO) << "Mistral3ForConditionalGeneration loaded successfully.";
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to load Mistral3ForConditionalGeneration: " << e.what();
      throw;
    }
  }

 private:
  void check_initialized() const {
    TORCH_CHECK(model_ != nullptr, 
                "Mistral3ForConditionalGeneration not initialized. Call constructor first.");
  }

  void validate_args(const ModelArgs& model_args) const {
    TORCH_CHECK(model_args.vocab_size() > 0, 
                "vocab_size must be positive, got ", model_args.vocab_size());
    TORCH_CHECK(model_args.mm_hidden_size() > 0, 
                "hidden_size must be positive, got ", model_args.mm_hidden_size());
  }

  torch::Tensor compute_logits(const torch::Tensor& hidden_states, 
                                int64_t logits_to_keep) const {
    if (logits_to_keep <= 0 || logits_to_keep >= hidden_states.size(1)) {
      // Keep all logits
      return lm_head_(hidden_states);
    }
    
    // Keep only the last `logits_to_keep` tokens for memory efficiency
    int64_t start_idx = hidden_states.size(1) - logits_to_keep;
    return lm_head_(hidden_states.slice(1, start_idx, hidden_states.size(1)));
  }

  // model_ and lm_head_ are inherited from LlmForCausalLMImplBase
  // No need to declare them here
};
TORCH_MODULE(Mistral3ForConditionalGeneration);

// Register the causal model
REGISTER_CAUSAL_MODEL(mistral3, Mistral3ForCausalLM);

// Register the model args
REGISTER_MODEL_ARGS(mistral3, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 131072);
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 5120);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 32768);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(mm_num_key_value_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(mm_head_dim, "head_dim", 128);
  LOAD_ARG_OR(mm_layer_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(mm_rope_theta, "rope_theta", 1e9);
});

} // namespace xllm