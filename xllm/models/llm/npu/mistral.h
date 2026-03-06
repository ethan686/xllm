#pragma once

#include <torch/torch.h>

#include "core/layers/common/activation.h"
#include "core/layers/common/attention.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rotary_embedding.h"  // 使用现有的 rotary_embedding
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/quant_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "core/layers/common/rms_norm.h"

// Mistral model compatible with huggingface weights
namespace xllm {

using InputParameters = xllm::InputParameters;
using ActivationFunc = xllm::ActivationFunc;
using CodedChatTemplate = xllm::CodedChatTemplate;
using AttentionMetadata = xllm::AttentionMetadata;

// ==================== Mistral MLP ====================

class MistralMLPImpl : public torch::nn::Module {
 public:
  MistralMLPImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func("silu", options.device());
    CHECK(act_func_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    gate_up_proj_ = register_module(
        "gate_up_proj",
        ColumnParallelLinear(
            hidden_size,
            std::vector<int64_t>{intermediate_size, intermediate_size},
            /*bias=*/false,
            /*gather_output=*/false,
            quant_args,
            parallel_args,
            options));
    down_proj_ =
        register_module("down_proj",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor x) {
    const auto gate_up = gate_up_proj_(x);
    return down_proj_(act_func_(gate_up[0]) * gate_up[1]);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_up_proj_->verify_loaded_weights(prefix + "[gate_proj,up_proj].");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  std::shared_ptr<ActivationFunc> act_func_{nullptr};
};
TORCH_MODULE(MistralMLP);

// ==================== Mistral Attention ====================

class MistralAttentionImpl : public torch::nn::Module {
 public:
  MistralAttentionImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options,
                       std::shared_ptr<RotaryEmbeddingBase> rotary_emb) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;
    const int64_t n_local_kv_heads =
        std::max<int64_t>(1, n_kv_heads / world_size);

    n_local_heads_ = n_local_heads;
    n_local_kv_heads_ = n_local_kv_heads;
    head_dim_ = head_dim;
    scaling_ = 1.0 / std::sqrt(static_cast<float>(head_dim));

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                QKVColumnParallelLinear(hidden_size,
                                                        n_heads,
                                                        n_kv_heads,
                                                        head_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args,
                                                        options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(hidden_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // 保存 rotary embedding
    rotary_emb_ = rotary_emb;
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    
    // QKV projection
    const auto qkv = qkv_proj_(x);
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];
    
    // Reshape for attention
    query = query.view({batch_size, seq_len, n_local_heads_, head_dim_})
                .transpose(1, 2);
    key = key.view({batch_size, seq_len, n_local_kv_heads_, head_dim_})
              .transpose(1, 2);
    value = value.view({batch_size, seq_len, n_local_kv_heads_, head_dim_})
                .transpose(1, 2);
    
    // Apply rotary embeddings using the existing implementation
    // 准备 cu_query_lens (用于 kernel 调用)
    torch::Tensor cu_query_lens;
    if (input_params.is_prefill) {
      // prefill 阶段
      cu_query_lens = torch::tensor({0, seq_len}, torch::kInt32)
                          .to(x.device());
    } else {
      // decode 阶段
      cu_query_lens = torch::tensor({0, 1}, torch::kInt32)
                          .to(x.device());
    }
    
    // 使用现有的 RotaryEmbedding forward 方法
    if (auto* rotary = dynamic_cast<RotaryEmbeddingImpl*>(rotary_emb_.get())) {
      // 对于标准 RoPE
      rotary->forward(query, key, positions, cu_query_lens, 
                      input_params.is_prefill ? seq_len : 1,
                      input_params.is_prefill);
    } else if (auto* mrotary = dynamic_cast<MRotaryEmbeddingImpl*>(rotary_emb_.get())) {
      // 对于 Multi-modal RoPE (如果需要)
      AttentionMetadata attn_metadata;
      attn_metadata.is_prefill = input_params.is_prefill;
      attn_metadata.is_chunked_prefill = false;
      attn_metadata.q_cu_seq_lens = cu_query_lens;
      attn_metadata.max_query_len = input_params.is_prefill ? seq_len : 1;
      mrotary->forward(query, key, positions, attn_metadata);
    }
    
    // Update KV cache
    std::tie(key, value) = kv_cache.update(key, value, input_params);
    
    // Repeat k/v heads for GQA
    if (n_local_kv_heads_ < n_local_heads_) {
      auto n_rep = n_local_heads_ / n_local_kv_heads_;
      key = repeat_kv(key, n_rep);
      value = repeat_kv(value, n_rep);
    }
    
    // Compute attention
    auto attn_weights = torch::matmul(query, key.transpose(-2, -1)) * scaling_;
    
    // Apply causal mask
    auto causal_mask = create_causal_mask(seq_len, x.device());
    attn_weights = attn_weights + causal_mask;
    
    // Softmax
    attn_weights = torch::softmax(attn_weights, -1, torch::kFloat32)
                       .to(query.dtype());
    
    // Apply attention to values
    auto attn_output = torch::matmul(attn_weights, value);
    attn_output = attn_output.transpose(1, 2)
                      .contiguous()
                      .view({batch_size, seq_len, -1});
    
    return o_proj_(attn_output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(
        state_dict, {"q_proj.", "k_proj.", "v_proj."}, {"k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    o_proj_->verify_loaded_weights(prefix + "o_proj.");
  }

 private:
  // Helper to repeat kv heads
  torch::Tensor repeat_kv(const torch::Tensor& x, int64_t n_rep) {
    if (n_rep == 1) return x;
    auto shape = x.sizes();
    return x.unsqueeze(2)
        .expand({shape[0], shape[1], n_rep, shape[2], shape[3]})
        .reshape({shape[0], shape[1], shape[2] * n_rep, shape[3]});
  }

  // Helper to create causal mask
  torch::Tensor create_causal_mask(int64_t seq_len, torch::Device device) {
    auto mask = torch::full({seq_len, seq_len}, 
                           -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().device(device));
    return torch::triu(mask, 1).unsqueeze(0).unsqueeze(0);
  }

  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t head_dim_;
  float scaling_;

  QKVColumnParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_{nullptr};
};
TORCH_MODULE(MistralAttention);

// ==================== Mistral Decoder Layer ====================

class MistralDecoderLayerImpl : public torch::nn::Module {
 public:
  MistralDecoderLayerImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          std::shared_ptr<RotaryEmbeddingBase> rotary_emb) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        MistralAttention(args, quant_args, parallel_args, options, rotary_emb));
    mlp_ = register_module(
        "mlp", MistralMLP(args, quant_args, parallel_args, options));
    input_layernorm_ = register_module(
        "input_layernorm",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + self_attn_(input_layernorm_(x), positions, kv_cache, input_params);
    return h + mlp_(post_attention_layernorm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
  }

 private:
  // parameter members, must be registered
  MistralAttention self_attn_{nullptr};
  MistralMLP mlp_{nullptr};
  RMSNorm input_layernorm_{nullptr};
  RMSNorm post_attention_layernorm_{nullptr};
};
TORCH_MODULE(MistralDecoderLayer);

// ==================== Mistral Model ====================

class MistralModelImpl : public torch::nn::Module {
 public:
  MistralModelImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options) {
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    // 创建 RotaryEmbedding 实例 - 使用现有的工厂函数
    rotary_emb_ = layer::create_mla_rotary_embedding(
        args,
        args.head_dim(),  // rotary_dim
        args.max_position_embeddings(),
        /*interleaved=*/false,  // Mistral uses half-half mode
        options);

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = MistralDecoderLayer(
          args, quant_args, parallel_args, options, rotary_emb_);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens_(tokens);

    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return norm_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("embed_tokens."));
    // rotary_emb 没有需要加载的权重（都是buffer）
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
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
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MistralDecoderLayer> layers_;

  RMSNorm norm_{nullptr};
};
TORCH_MODULE(MistralModel);

// ==================== Mistral For Causal LM ====================

class MistralForCausalLMImpl : public torch::nn::Module {
 public:
  MistralForCausalLMImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options) {
    // register submodules
    model_ = register_module(
        "model", MistralModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
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
                        const InputParameters& input_params) {
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
    model_->load_state_dict(state_dict.select("model."));
    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  // parameter members, must be registered
  MistralModel model_{nullptr};
  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(MistralForCausalLM);

// ==================== Chat Template ====================

class MistralChatTemplate final : public CodedChatTemplate {
 public:
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message.empty()) {
      ss << system_message;
    }

    for (size_t i = 0; i < messages.size(); ++i) {
      if (i % 2 == 0) {
        ss << "[INST] " << messages[i] << " ";
      } else {
        ss << "[/INST] " << messages[i] << "</s>";
      }
    }
    ss << "[/INST]";
    return ss.str();
  }
};

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