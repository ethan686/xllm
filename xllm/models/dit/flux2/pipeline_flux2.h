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
#include "pipeline_flux2_base.h"
#include "transformer_flux2.h"
#include "image_processor.h"
//#include "models/dit/mistral3_encoder.h"
#include "models/dit/flowmatch_euler_discrete_scheduler.h"
#include "autoencoder_kl_flux2.h"
//#include "core/framework/chat_template/jinja_chat_template.h"
// pipeline_flux2 compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2.py

namespace xllm {

class Flux2PipelineImpl : public Flux2PipelineBaseImpl {
 public:
  Flux2PipelineImpl(const DiTModelContext& context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    tokenizer_max_length_ = 512;
    default_sample_size_ = 128;
    LOG(INFO) << "Initializing Flux2 pipeline...";
    flux2_image_processor_ = Flux2ImageProcessor(context.get_model_context("vae"), vae_scale_factor_ * 2);
    vae_ = Flux2Vae(context.get_model_context("vae"));
    pos_embed_ = register_module(
        "pos_embed",
        Flux2PosEmbed(context.get_model_args("transformer").rope_theta(),
                     context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = Flux2DiTModel(context.get_model_context("transformer"));
    //mistral3_ = Mistral3EncoderModel(context.get_model_context("text_encoder"));
    scheduler_ = FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
    register_module("vae", vae_);
    register_module("flux2_image_processor", flux2_image_processor_);
    register_module("transformer", transformer_);
    //register_module("mistral3", mistral3_);
    register_module("scheduler", scheduler_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto images = input.images.defined() ? std::make_optional(input.images)
                                           : std::nullopt;

    auto output = forward_impl(
        prompts,                                  // prompt
        generation_params.height,                 // height
        generation_params.width,                  // width
        generation_params.num_inference_steps,    // num_inference_steps
        generation_params.guidance_scale,         // guidance_scale
        generation_params.num_images_per_prompt,  // num_images_per_prompt
        seed,                                     // seed
        latents,                                  // latents
        prompt_embeds,                            // prompt_embeds
        images,                                   // images
        generation_params.max_sequence_length     // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "Flux2Pipeline loading model from" << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    //auto mistral3_loader = loader->take_component_loader("text_encoder");
    //auto tokenizer_loader = loader->take_component_loader("tokenizer");
    LOG(INFO)
        << "Flux2 model components loaded, start to load weights to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    //mistral3_->load_model(std::move(mistral3_loader));
    //mistral3_->to(options_.device());
    //tokenizer_ = tokenizer_loader->tokenizer();
    //chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_loader->tokenizer_args());
  }

 private:
  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents * 4, adjusted_height / 2, adjusted_width / 2};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = prepare_latent_image_ids(latents.value());
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor = randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(latents_tensor);
    return {packed_latents, latent_image_ids};
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_image_latents(
      const std::vector<torch::Tensor>& images,
      int64_t batch_size,
      int64_t seed) {
    std::vector<torch::Tensor> image_latents;
    for (const auto& image : images) {
      auto image_latent = vae_->encode(image, seed);
      auto patched_latent = patchify_latents(image_latent);
      auto latents_bn_mean = vae_->get_bn_running_mean().view({1, -1, 1, 1}).to(image_latent.device(), image_latent.dtype());
      auto latents_bn_std = torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) + vae_->get_batch_norm_eps());
      image_latent = (patched_latent - latents_bn_mean) / latents_bn_std;
      image_latents.push_back(image_latent);
    }
    auto concatenated_latents = pack_latents_for_images(image_latents);
    auto image_latent_ids = _prepare_image_ids(image_latents);

    concatenated_latents = concatenated_latents.unsqueeze(0);
    auto repeated_latents = concatenated_latents.repeat({batch_size, 1, 1});
    auto repeated_ids = image_latent_ids.repeat({batch_size, 1, 1});
    return {repeated_latents, repeated_ids};
  }

  torch::Tensor pack_latents_for_images(const std::vector<torch::Tensor>& image_latents) {
    std::vector<torch::Tensor> packed_latents;
    for (const auto& latent : image_latents) {
      auto packed = pack_latents(latent);
      packed = packed.squeeze(0);
      packed_latents.push_back(packed);
    }
    return torch::cat(packed_latents, 0);
  }

  torch::Tensor forward_impl(
      std::optional<std::vector<std::string>> prompt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_inference_steps = 50,
      float guidance_scale = 4.0f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> images = std::nullopt,
      int64_t max_sequence_length = 512,
      const std::vector<int64_t>& hidden_states_layers = {10, 20, 30},
      const std::string& system_message = SYSTEM_MESSAGE) {
    torch::NoGradGuard no_grad;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;
    // encode prompt

    torch::Tensor encoded_prompt_embeds = torch::randn(
        at::IntArrayRef({total_batch_size, max_sequence_length, 6444}),
        torch::dtype(torch::kFloat32)
    );

    torch::Tensor text_ids = torch::zeros({total_batch_size, max_sequence_length}, torch::kInt64);
    /* auto [encoded_prompt_embeds, text_ids] =
        encode_prompt(prompt,
                  prompt_embeds,
                  num_images_per_prompt,
                  max_sequence_length,
                  hidden_states_layers,
                  system_message);
    */
    // process images
    std::vector<torch::Tensor> condition_images_list;
    if (images.has_value()) {
      auto input_images = images.value();
      if (input_images.dim() == 3) {
        input_images = input_images.unsqueeze(0);
      }
      for (int64_t i = 0; i < input_images.size(0); ++i) {
        auto img = input_images[i];
        flux2_image_processor_->check_image_input(img);
        int64_t image_width = img.size(-1);
        int64_t image_height = img.size(-2);
        if (image_width * image_height > 1024 * 1024) {
          img = flux2_image_processor_->resize_to_target_area(img, 1024 * 1024);
          image_width = img.size(-1);
          image_height = img.size(-2);
        }
        int64_t multiple_of = vae_scale_factor_ * 2;
        image_width = (image_width / multiple_of) * multiple_of;
        image_height = (image_height / multiple_of) * multiple_of;
        img = flux2_image_processor_->preprocess(img, image_height, image_width);
        condition_images_list.push_back(img);
      }
    }
    // prepare latent
    int64_t num_channels_latents = transformer_->in_channels() / 4;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);

    // prepare image latents if condition_images_list is not empty
    torch::Tensor image_latents;
    torch::Tensor image_latent_ids;
    if (!condition_images_list.empty()) {
      std::tie(image_latents, image_latent_ids) = prepare_image_latents(condition_images_list, total_batch_size, seed.has_value() ? seed.value() : 42);
    }

    // prepare timestep
    std::vector<float> new_sigmas;
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                      (num_inference_steps - 1) *
                                      (1.0f - 1.0f / num_inference_steps));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = compute_empirical_mu(image_seq_len, num_inference_steps);
    auto [timesteps, num_inference_steps_actual] = flux2_retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);
    //int64_t num_warmup_steps =
    //    std::max(static_cast<int64_t>(timesteps.numel()) -
    //                 num_inference_steps_actual * scheduler_->order(),
    //             static_cast<int64_t>(0LL));
    // prepare guidance
    torch::Tensor guidance;
    torch::TensorOptions options =
        torch::dtype(torch::kFloat32).device(options_.device());

    guidance = torch::full(at::IntArrayRef({1}), guidance_scale, options);
    guidance = guidance.expand({prepared_latents.size(0)});

    // image rotary positional embeddings outplace computation
    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(options_.dtype());
    // denosing loop
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());
    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);

      torch::Tensor latent_model_input = prepared_latents.to(options_.dtype());
      torch::Tensor latent_image_ids_input = latent_image_ids;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({prepared_latents, image_latents}, 1).to(options_.dtype());
        latent_image_ids_input = torch::cat({latent_image_ids, image_latent_ids}, 1);
        auto [rot_emb1_img, rot_emb2_img] =
            pos_embed_->forward_cache(text_ids,
                                      latent_image_ids_input,
                                      height / (vae_scale_factor_ * 2),
                                      width / (vae_scale_factor_ * 2));
        image_rotary_emb = torch::stack({rot_emb1_img, rot_emb2_img}, 0).to(options_.dtype());
      }
      torch::Tensor noise_pred = transformer_->forward(latent_model_input,
                                                       encoded_prompt_embeds,
                                                       timestep,
                                                       guidance,
                                                       image_rotary_emb);
      if (image_latents.defined()) {
        noise_pred = noise_pred.narrow(1, 0, prepared_latents.size(1));
      }
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }
    torch::Tensor image;
    torch::Tensor unpacked_latents =
        unpack_latents_with_ids(prepared_latents, latent_image_ids);
    auto latents_bn_mean = vae_->get_bn_running_mean().view({1, -1, 1, 1}).to(unpacked_latents.device(), unpacked_latents.dtype());
    auto latents_bn_std = torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) + vae_->get_batch_norm_eps()).to(unpacked_latents.device(), unpacked_latents.dtype());
    unpacked_latents = unpacked_latents * latents_bn_std + latents_bn_mean;
    unpacked_latents = unpatchify_latents(unpacked_latents);
    image = vae_->decode(unpacked_latents);
    image = flux2_image_processor_->postprocess(image);
    return image;
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  Flux2Vae vae_{nullptr};
  Flux2ImageProcessor flux2_image_processor_{nullptr};
  Flux2DiTModel transformer_{nullptr};
  // Mistral3EncoderModel mistral3_{nullptr};
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int tokenizer_max_length_;
  int default_sample_size_;
  int vae_scale_factor_;
  Flux2PosEmbed pos_embed_{nullptr};
  //std::unique_ptr<JinjaChatTemplate> chat_template_;
};
TORCH_MODULE(Flux2Pipeline);

REGISTER_DIT_MODEL(flux2, Flux2Pipeline);
};  // namespace xllm
