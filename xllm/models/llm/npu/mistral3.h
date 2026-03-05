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

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "llm_model_base.h"
#include "models/model_registry.h"
#include "mistral.h"

namespace xllm {

// Output structure for Mistral3 model
struct Mistral3ModelOutputWithPast {
  torch::Tensor last_hidden_state;           // Final layer output
  std::shared_ptr<Cache> past_key_values;
  std::vector<torch::Tensor> hidden_states;  // All layers outputs (if output_hidden_states=True)
  std::vector<torch::Tensor> attentions;
  std::vector<torch::Tensor> image_hidden_states;  // Always empty for text-only
};

// Mistral3 model (without LM head)
class Mistral3ModelImpl : public LlmModelImplBase<MistralDecoderLayer> {
 public:
  explicit Mistral3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<MistralDecoderLayer>("mistral3", context.get_model_args()) {
    auto model_args = context.get_model_args();
    
    language_model_ = register_module("language_model", MistralModel(context));
  }

  Mistral3ModelOutputWithPast forward(
      torch::Tensor input_ids,
      torch::Tensor attention_mask,
      torch::Tensor position_ids,
      std::shared_ptr<Cache>& past_key_values,
      torch::Tensor inputs_embeds,
      bool use_cache,
      bool output_attentions,
      bool output_hidden_states,
      torch::Tensor cache_position) {
    
    auto lm_outputs = language_model_->forward(
        input_ids, attention_mask, position_ids, past_key_values,
        inputs_embeds, use_cache, output_hidden_states, output_attentions,
        cache_position);
    
    return Mistral3ModelOutputWithPast{
        lm_outputs.last_hidden_state,  // This is the last layer output
        lm_outputs.past_key_values,
        lm_outputs.hidden_states,      // This contains all layers if output_hidden_states=True
        lm_outputs.attentions,
        {}  // empty image_hidden_states
    };
  }

  void load_state_dict(const StateDict& state_dict) {
    language_model_->load_state_dict(
        state_dict.get_dict_with_prefix("model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    language_model_->verify_loaded_weights(prefix + "model.");
  }

 private:
  MistralModel language_model_ = nullptr;
};
TORCH_MODULE(Mistral3Model);

// Output structure for Mistral3 with LM head
struct Mistral3CausalLMOutputWithPast {
  torch::Tensor logits;                          // LM head output
  std::shared_ptr<Cache> past_key_values;
  std::vector<torch::Tensor> hidden_states;      // All layers outputs (if output_hidden_states=True)
  std::vector<torch::Tensor> attentions;
  std::vector<torch::Tensor> image_hidden_states; // Always empty for text-only
};

// Mistral3 model for conditional generation (text-only)
class Mistral3ForConditionalGenerationImpl : public LlmForCausalLMImplBase<Mistral3Model> {
 public:
  explicit Mistral3ForConditionalGenerationImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Mistral3Model>(context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    
    // LlmForCausalLMImplBase already registers model_ and lm_head_
    // No additional initialization needed
  }

  Mistral3CausalLMOutputWithPast forward(
      torch::Tensor input_ids,
      torch::Tensor attention_mask,
      torch::Tensor position_ids,
      std::shared_ptr<Cache>& past_key_values,
      torch::Tensor inputs_embeds,
      bool use_cache,
      bool output_attentions,
      bool output_hidden_states,
      torch::Tensor cache_position,
      int64_t logits_to_keep) {
    
    // Forward through base model
    auto outputs = model_->forward(
        input_ids, attention_mask, position_ids, past_key_values,
        inputs_embeds, use_cache, output_attentions, output_hidden_states,
        cache_position);
    
    // Compute logits from last_hidden_state
    auto hidden_states = outputs.last_hidden_state;
    int64_t start_idx = hidden_states.size(1) - logits_to_keep;
    auto logits = lm_head_(
        hidden_states.slice(1, start_idx, hidden_states.size(1)));
    
    return Mistral3CausalLMOutputWithPast{
        logits,
        outputs.past_key_values,
        outputs.hidden_states,    // Pass through all hidden states
        outputs.attentions,
        {}  // empty image_hidden_states
    };
  }

  void load_state_dict(const StateDict& state_dict) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("language_model."));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("language_model.lm_head."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights(prefix + "lm_head.");
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    LOG(INFO) << "Loading Mistral3ForConditionalGeneration from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("language_model."));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("language_model.lm_head."));
    }
    
    model_->verify_loaded_weights("language_model.");
    lm_head_->verify_loaded_weights("language_model.lm_head.");
    LOG(INFO) << "Mistral3ForConditionalGeneration loaded successfully.";
  }

 private:
  // model_ and lm_head_ are inherited from LlmForCausalLMImplBase
  // No need to declare them here
};
TORCH_MODULE(Mistral3ForConditionalGeneration);

// register the causal model
REGISTER_CAUSAL_MODEL(mistral3, Mistral3ForConditionalGeneration);

// register the model args
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