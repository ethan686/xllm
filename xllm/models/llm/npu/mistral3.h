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
// Mistral3 model (without LM head)
class Mistral3ModelImpl : public torch::nn::Module {
 public:
  explicit Mistral3ModelImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options) {   
    language_model_ = register_module(
        "language_model",
        MistralModel(args, quant_args, parallel_args, options));
  }

  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {   
    return language_model_->forward(
        tokens, positions, kv_caches, input_params);
  }

  void load_state_dict(const StateDict& state_dict) {
    language_model_->load_state_dict(
        state_dict.get_dict_with_prefix("language_model.model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    language_model_->verify_loaded_weights(prefix + "language_model.model.");
  }

 private:
  MistralModel language_model_{nullptr};
};
TORCH_MODULE(Mistral3Model);

// Mistral3 model for conditional generation (text-only)
class Mistral3ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Mistral3ForConditionalGenerationImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options) {
    // register submodules
    model_ = register_module(
        "model", Mistral3Model(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }
    
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
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
  // parameter members, must be registered
  Mistral3Model model_{nullptr};
  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(Mistral3ForConditionalGeneration);

// Model registration
REGISTER_CAUSAL_MODEL(mistral3, Mistral3ForConditionalGeneration);

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
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
});

} // namespace xllm