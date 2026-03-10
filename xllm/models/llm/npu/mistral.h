#pragma once

#include <torch/torch.h>

#include "core/layers/common/activation.h"
#include "core/layers/common/attention.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rotary_embedding.h"  // 使用现有的 rotary_embedding
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_mistral_decoder_layer_impl.h"

// Mistral model compatible with huggingface weights
namespace xllm {
torch::Tensor silu(torch::Tensor x) {
  return x * torch::sigmoid(x);
}

// ==================== Mistral Decoder Layer ====================

class MistralDecoderLayerImpl : public torch::nn::Module {
 public:
  MistralDecoderLayerImpl(const ModelContext& context) {
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuMistralDecoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        int node_id) {
    return decoder_layer_(
        x, cos_pos, sin_pos, attn_mask, kv_cache, input_params, node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {}

 private:
  layer::NpuMistralDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(MistralDecoderLayer);

std::tuple<torch::Tensor, torch::Tensor> get_mistral_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  // auto inv_freq = 1.0 / torch::pow(10000, torch::arange(0, dim, 2, options) /
  // dim);
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  auto emb = torch::cat({freqs, freqs}, -1);
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  return std::make_tuple(rope_cos, rope_sin);
}

// ==================== Mistral Model ====================

class MistralModelImpl : public torch::nn::Module {
 public:
  MistralModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(context.get_model_args().n_layers());
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    std::tie(cos_pos_, sin_pos_) =
        get_mistral_rotary_embedding(128,
                                   model_args.max_position_embeddings(),
                                   model_args.rope_theta(),
                                   options);
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    max_seq_len_ = 0;

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = MistralDecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::Tensor h = npu_embed_tokens_(tokens, 0);
    auto cos_pos = cos_pos_.index_select(0, positions);
    auto sin_pos = sin_pos_.index_select(0, positions);
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    // torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
    // max_seq_len_ = std::max(max_of_seq.item<int>(), max_seq_len_);
    torch::Tensor max_of_seq = torch::max(input_params.kv_seq_lens);
    max_seq_len_ = FLAGS_enable_chunked_prefill
                       ? std::max(max_of_seq.item<int>(), max_seq_len_)
                       : 128;
    auto attn_mask = attn_mask_.get_attn_mask(
        max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

    if (FLAGS_enable_chunked_prefill) {
      int batch_size = input_params.q_seq_lens_vec.size();
      std::vector<torch::Tensor> req_mask_vec;
      req_mask_vec.reserve(batch_size);

      for (int i = 0; i < batch_size; i++) {
        int start =
            input_params.kv_seq_lens_vec[i] - input_params.q_seq_lens_vec[i];
        int end = input_params.kv_seq_lens_vec[i];

        auto req_mask_slice = attn_mask.slice(0, start, end);
        req_mask_vec.emplace_back(req_mask_slice);
      }
      attn_mask = torch::cat(req_mask_vec, 0);
    }
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      layer(h, cos_pos, sin_pos, attn_mask, kv_caches[i], input_params_new, i);
    }
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix("embed_tokens."));
    // rotary_emb 没有需要加载的权重（都是buffer）
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

 private: 
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  int max_seq_len_ = 0;
  int device_id_ = 0;
  layer::AttentionMask attn_mask_;
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<MistralDecoderLayer> layers_;
};
TORCH_MODULE(MistralModel);

// ==================== Mistral For Causal LM ====================

class MistralForCausalLMImpl : public torch::nn::Module {
 public:
  MistralForCausalLMImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    // register submodules
    model_ = register_module(
        "model", MistralModel(context));

    lm_head_ = register_module("lm_head",
                               xllm::layer::ColumnParallelLinear(model_args.hidden_size(),
                                                    model_args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.get_dict_with_prefix("model."));
    lm_head_->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "Loading MistralForCausalLM from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("language_model."));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("language_model.lm_head."));
    }
    
    model_->verify_loaded_weights("language_model.");
    lm_head_->verify_loaded_weights("language_model.lm_head.");
    LOG(INFO) << "MistralForCausalLM loaded successfully.";
  }
 private:
  // parameter members, must be registered
  MistralModel model_{nullptr};
  xllm::layer::ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(MistralForCausalLM);

// ==================== Registration ====================

REGISTER_CAUSAL_MODEL(mistral, MistralForCausalLM);
REGISTER_DEFAULT_CHAT_TEMPLATE(mistral, MistralChatTemplate);
REGISTER_MODEL_ARGS(mistral, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mistral");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 32000);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 14336);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096 * 32);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  
  // DeepSeek YARN scaling parameters (optional)
  LOAD_ARG_OR_FUNC(rope_scaling_rope_type, "rope_scaling_rope_type", [&] {
    return std::string("default");
  });
  LOAD_ARG_OR_FUNC(rope_scaling_factor, "rope_scaling_factor", 1.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_original_max_position_embeddings, 
                   "rope_scaling_original_max_position_embeddings", 4096);
  LOAD_ARG_OR_FUNC(rope_extrapolation_factor, "rope_extrapolation_factor", 1.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_attn_factor, "rope_scaling_attn_factor", 1.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_beta_fast, "rope_scaling_beta_fast", 32.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_beta_slow, "rope_scaling_beta_slow", 1.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_mscale, "rope_scaling_mscale", 1.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_mscale_all_dim, "rope_scaling_mscale_all_dim", 1.0f);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace xllm