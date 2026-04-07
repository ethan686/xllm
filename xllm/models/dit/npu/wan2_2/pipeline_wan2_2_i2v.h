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
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "autoencoder_kl_wan.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/model_registry.h"
#include "transformer_wan2_2.h"
#include "umt5_encoder.h"
#include "uni_pc_multi_step_scheduler.h"
#include "video_processor.h"

namespace xllm {

class Wan2_2I2VPipelineImpl : public torch::nn::Module {
 public:
  Wan2_2I2VPipelineImpl(const DiTModelContext& context) {
    options_ = context.get_tensor_options();
    const auto& model_args = context.get_model_args("vae");
    zdim_ = model_args.zdim();
    latents_mean_ = model_args.latents_mean();
    latents_std_ = model_args.latents_std();

    model_args = context.get_model_args("scheduler");
    num_train_timesteps_ = model_args.num_train_timesteps();

    LOG(INFO) << "Initializing Wan2_2I2V pipeline...";
    vae_ = WANVAE(context.get_model_context("vae"));
    transformer_ = Wan2_2DiTModel(context.get_model_context("transformer"));
    transformer_2_ = Wan2_2DiTModel(context.get_model_context("transformer_2"));
    umt5_ = UMt5EncoderModel(context.get_model_context("text_encoder"));
    scheduler_ =
        UniPCMultiStepScheduler(context.get_model_context("scheduler"));
    video_processor_ = VideoProcessor(context.get_model_context("vae"),
                                      true,
                                      true,
                                      false,
                                      false,
                                      false,
                                      4,
                                      vae_scale_factor_spatial_);
    register_module("vae", vae_);
    register_module("transformer", transformer_);
    register_module("transformer_2", transformer_2_);
    register_module("umt5", umt5_);
    register_module("scheduler", scheduler_);
    register_module("video_processor_", video_processor_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto images = input.images.defined() ? std::make_optional(input.images)
                                         : std::nullopt;
    auto last_images = input.last_images.defined()
                           ? std::make_optional(input.last_images)
                           : std::nullopt;
    auto prompts = std::make_optional(input.prompts);
    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;
    auto image_embeds = input.image_embeds.defined()
                            ? std::make_optional(input.image_embeds)
                            : std::nullopt;

    auto output = forward_impl(images,
                               last_images,
                               prompts,
                               negative_prompts,
                               generation_params.height,
                               generation_params.width,
                               generation_params.num_frames,
                               generation_params.num_inference_steps,
                               generation_params.guidance_scale,
                               generation_params.guidance_scale_2,
                               generation_params.num_videos_per_prompt,
                               seed,
                               latents,
                               image_embeds,
                               prompt_embeds,
                               negative_prompt_embeds,
                               generation_params.max_sequence_length);

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "Wan2_2I2VPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto transformer_2_loader = loader->take_component_loader("transformer_2");
    auto vae_loader = loader->take_component_loader("vae");
    auto umt5_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");

    LOG(INFO) << "Wan2_2I2VPipeline model components loaded, start to load "
                 "weights to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    transformer_2_->load_model(std::move(transformer_2_loader));
    transformer_2_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    umt5_->load_model(std::move(umt5_loader));
    umt5_->to(options_.device());
    tokenizer_ = tokenizer_loader->tokenizer();
  }

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare_latents(
      torch::Tensor image,
      int64_t batch_size,
      int64_t num_channels_latents = 16,
      int64_t height = 480,
      int64_t width = 832,
      int64_t num_frames = 81,
      std::optional<torch::Tensor> last_image = std::nullopt,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t num_latent_frames =
        (num_frames - 1) / vae_scale_factor_temporal_ + 1;
    int64_t latent_height = height / vae_scale_factor_spatial_;
    int64_t latent_width = width / vae_scale_factor_spatial_;

    std::vector<int64_t> shape = {batch_size,
                                  num_channels_latents,
                                  num_latent_frames,
                                  latent_height,
                                  latent_width};
    torch::Tensor latents_tensor;
    if (latents.has_value()) {
      latents_tensor = latents.value().to(options_.device());
    } else {
      latents_tensor = randn_tensor(shape, seed, options_);
    }

    image = image.unsqueeze(2);
    torch::Tensor video_condition;

    if (expand_timesteps_) {
      video_condition = image;
    } else if (!last_image.has_value()) {
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 1, height, width},
          image.options());
      video_condition = torch::cat({image, zeros}, 2);
    } else {
      auto last_img = last_image.value().unsqueeze(2);
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 2, height, width},
          image.options());
      video_condition = torch::cat({image, zeros, last_img}, 2);
    }
    video_condition = video_condition.to(options_.device(), vae_->dtype());

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(latents_tensor.device(), latents_tensor.dtype());
    torch::Tensor latents_std =
        1.0 / torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
                  .view({1, num_channels_latents, 1, 1, 1})
                  .to(latents_tensor.device(), latents_tensor.dtype());

    torch::Tensor latent_condition = vae_->encode(video_condition);
    latent_condition = (latent_condition - latents_mean) * latents_std;

    if (latent_condition.size(0) == 1 && batch_size > 1) {
      latent_condition = latent_condition.repeat({batch_size, 1, 1, 1, 1});
    }

    if (expand_timesteps_) {
      torch::Tensor first_frame_mask = torch::ones(
          {1, 1, num_latent_frames, latent_height, latent_width},
          options_.dtype(torch::kFloat32).device(options_.device()));
      first_frame_mask.slice(2, 0, 1) = 0;
      return {latents_tensor, latent_condition, first_frame_mask};
    }

    torch::Tensor mask_lat_size =
        torch::ones({batch_size, 1, num_frames, latent_height, latent_width},
                    options_.dtype(torch::kFloat32).device(options_.device()));

    if (!last_image.has_value()) {
      for (int64_t i = 1; i < num_frames; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    } else {
      for (int64_t i = 1; i < num_frames - 1; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    }

    torch::Tensor first_frame_mask = mask_lat_size.slice(2, 0, 1);
    first_frame_mask = torch::repeat_interleave(
        first_frame_mask, vae_scale_factor_temporal_, 2);

    torch::Tensor rest_mask = mask_lat_size.slice(2, 1);
    mask_lat_size = torch::cat({first_frame_mask, rest_mask}, 2);

    mask_lat_size = mask_lat_size.view({batch_size,
                                        -1,
                                        vae_scale_factor_temporal_,
                                        latent_height,
                                        latent_width});
    mask_lat_size = mask_lat_size.transpose(1, 2);
    mask_lat_size = mask_lat_size.to(latent_condition.device());

    torch::Tensor combined_condition =
        torch::cat({mask_lat_size, latent_condition}, 1);

    return {latents_tensor, combined_condition, first_frame_mask};
  }

  torch::Tensor get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                     int64_t num_videos_per_prompt = 1,
                                     int64_t max_sequence_length = 512) {
    int64_t batch_size = prompt.size();

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(options_.device());

    torch::Tensor prompt_embeds = umt5_->forward(input_ids);
    prompt_embeds = prompt_embeds.to(options_);

    auto seq_lens = (input_ids > 0).sum(1).to(torch::kLong);

    std::vector<torch::Tensor> trimmed_embeds;
    trimmed_embeds.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t seq_len = seq_lens[i].item<int64_t>();
      auto trimmed = prompt_embeds[i].slice(0, 0, seq_len);
      int64_t padding_len = max_sequence_length - trimmed.size(0);
      if (padding_len > 0) {
        auto zeros =
            torch::zeros({padding_len, trimmed.size(1)}, trimmed.options());
        trimmed = torch::cat({trimmed, zeros}, 0);
      }
      trimmed_embeds.push_back(trimmed);
    }
    prompt_embeds = torch::stack(trimmed_embeds, 0);

    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_videos_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_videos_per_prompt, seq_len, -1});
    return prompt_embeds;
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> negative_prompt,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> negative_prompt_embeds,
      bool do_classifier_free_guidance = true,
      int64_t num_videos_per_prompt = 1,
      int64_t max_sequence_length = 226) {
    torch::Tensor prompt_embeds_tensor;
    torch::Tensor negative_prompt_embeds_tensor;

    if (prompt_embeds.has_value()) {
      prompt_embeds_tensor = prompt_embeds.value();
    } else if (prompt.has_value()) {
      prompt_embeds_tensor = get_t5_prompt_embeds(
          prompt.value(), num_videos_per_prompt, max_sequence_length);
    }

    int64_t batch_size;
    if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds_tensor.size(0);
    } else if (prompt.has_value()) {
      batch_size = prompt.value().length();
    }

    if (do_classifier_free_guidance) {
      if (negative_prompt.has_value()) {
        negative_prompt_embeds_tensor =
            get_t5_prompt_embeds(negative_prompt.value(),
                                 num_videos_per_prompt,
                                 max_sequence_length);
      }
    } else if (!negative_prompt_embeds_tensor.defined()) {
      int64_t seq_len = prompt_embeds_tensor.size(1);
      int64_t hidden_dim = prompt_embeds_tensor.size(2);
      negative_prompt_embeds_tensor = torch::zeros(
          {batch_size, seq_len, hidden_dim}, prompt_embeds_tensor.options());
    }

    return {prompt_embeds_tensor, negative_prompt_embeds_tensor};
  }

  torch::Tensor forward_impl(
      std::optional<torch::Tensor> images = std::nullopt,
      std::optional<torch::Tensor> last_images = std::nullopt,
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_frames = 16,
      int64_t num_inference_steps = 28,
      float guidance_scale = 5.0f,
      float guidance_scale_2 = -1.0f,
      int64_t num_videos_per_prompt = 1,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> image_embeds = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds.value().size(0);
    }

    int64_t total_batch_size = batch_size * num_videos_per_prompt;
    bool has_neg_prompt =
        negative_prompt.has_value() || (negative_prompt_embeds.has_value());

    bool do_classifier_free_guidance = guidance_scale > 1.0f;

    if (num_frames % vae_scale_factor_temporal_ != 1) {
      LOG(WARNING) << "num_frames - 1 has to be divisible by "
                   << vae_scale_factor_temporal_
                   << ". Rounding to the nearest number.";
      num_frames =
          num_frames / vae_scale_factor_temporal_ * vae_scale_factor_temporal_ +
          1;
    }
    num_frames = std::max(num_frames, static_cast<int64_t>(1));

    int64_t patch_size_h = transformer_->patch_size()[1];
    int64_t patch_size_w = transformer_->patch_size()[2];
    int64_t h_multiple_of = vae_scale_factor_spatial_ * patch_size_h;
    int64_t w_multiple_of = vae_scale_factor_spatial_ * patch_size_w;
    int64_t calc_height = height / h_multiple_of * h_multiple_of;
    int64_t calc_width = width / w_multiple_of * w_multiple_of;
    if (height != calc_height || width != calc_width) {
      LOG(WARNING) << "height and width must be multiples of (" << h_multiple_of
                   << ", " << w_multiple_of
                   << ") for proper patchification. Adjusting (" << height
                   << ", " << width << ") -> (" << calc_height << ", "
                   << calc_width << ").";
      height = calc_height;
      width = calc_width;
    }

    if (boundary_ratio_ > 0.0f && guidance_scale_2 < 0.0f) {
      guidance_scale_2 = guidance_scale;
    }

    auto [encoded_prompt_embeds, encoded_negative_embeds] =
        encode_prompt(prompt,
                      negative_prompt,
                      prompt_embeds,
                      negative_prompt_embeds,
                      do_classifier_free_guidance,
                      num_videos_per_prompt,
                      max_sequence_length);

    // wan2.2 image_dimm is null, so not use the encode_image function

    scheduler_->set_timesteps(num_inference_steps, options_.device());
    torch::Tensor timesteps = scheduler_->timesteps();

    int64_t num_channels_latents = zdim_;
    torch::Tensor preprocessed_image =
        video_processor_->preprocess(images.value(), height, width);
    preprocessed_image =
        preprocessed_image.to(options_.device(), torch::kFloat32);

    std::optional<torch::Tensor> preprocessed_last_image;
    if (last_images.has_value()) {
      torch::Tensor last_img =
          video_processor_->preprocess(last_images.value(), height, width);
      preprocessed_last_image = last_img.to(options_.device(), torch::kFloat32);
    }

    torch::Tensor prepared_latents, latent_condition, first_frame_mask;
    std::tie(prepared_latents, latent_condition, first_frame_mask) =
        prepare_latents(preprocessed_image,
                        total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        num_frames,
                        preprocessed_last_image,
                        seed,
                        latents);

    float boundary_timestep =
        boundary_ratio_ > 0.0f ? boundary_ratio_ * num_train_timesteps_ : -1.0f;

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i];

      Wan2_2DiTModel current_model;
      float current_guidance;

      if (boundary_timestep < 0 || t.item<float>() >= boundary_timestep) {
        current_model = transformer_;
        current_guidance = guidance_scale;
      } else {
        current_model = transformer_2_;
        current_guidance = guidance_scale_2;
      }

      torch::Tensor latent_model_input;
      torch::Tensor timestep_input;

      if (expand_timesteps_) {
        latent_model_input = (1 - first_frame_mask) * latent_condition +
                             first_frame_mask * prepared_latents;
        latent_model_input = latent_model_input.to(prepared_latents.dtype());

        torch::Tensor temp_ts = (first_frame_mask[0][0]
                                     .slice(1, 0, first_frame_mask.size(2), 2)
                                     .slice(2, 0, first_frame_mask.size(3), 2) *
                                 t)
                                    .flatten();
        timestep_input =
            temp_ts.unsqueeze(0).expand({prepared_latents.size(0), -1});
      } else {
        latent_model_input =
            torch::cat({prepared_latents, latent_condition}, 1);
        latent_model_input = latent_model_input.to(prepared_latents.dtype());
        timestep_input = t.expand(prepared_latents.size(0));
      }

      torch::Tensor noise_pred = current_model->forward(latent_model_input,
                                                        timestep_input,
                                                        encoded_prompt_embeds,
                                                        torch::Tensor());

      if (do_classifier_free_guidance) {
        torch::Tensor noise_uncond =
            current_model->forward(latent_model_input,
                                   timestep_input,
                                   encoded_negative_embeds,
                                   torch::Tensor());
        noise_pred =
            noise_uncond + current_guidance * (noise_pred - noise_uncond);
        noise_uncond.reset();
      }

      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    if (expand_timesteps_) {
      prepared_latents = (1 - first_frame_mask) * latent_condition +
                         first_frame_mask * prepared_latents;
    }

    torch::Tensor video;
    prepared_latents = prepared_latents.to(vae_->dtype());

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(prepared_latents.device(), prepared_latents.dtype());
    torch::Tensor latents_std =
        1.0 / torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
                  .view({1, num_channels_latents, 1, 1, 1})
                  .to(prepared_latents.device(), prepared_latents.dtype());

    prepared_latents = prepared_latents / latents_std + latents_mean;
    video = vae_->decode(prepared_latents);
    video = video_processor_->postprocess_video(video);

    return video;
  }

 private:
  UniPCMultiStepScheduler scheduler_{nullptr};
  WANVAE vae_{nullptr};
  Wan2_2DiTModel transformer_{nullptr};
  Wan2_2DiTModel transformer_2_{nullptr};
  UMt5EncoderModel umt5_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_{nullptr};
  VideoProcessor video_processor_{nullptr};

  float vae_scaling_factor_;
  float vae_shift_factor_;
  int64_t vae_scale_factor_spatial_ = 8;
  int64_t vae_scale_factor_temporal_ = 4;
  float boundary_ratio_ = 0.9f;
  bool expand_timesteps_ = false;
  int64_t zdim_ = 16;
  float num_train_timesteps_ = 1000.0f;
  std::vector<int> latents_mean_;
  std::vector<int> latents_std_;
  torch::TensorOptions options_;
};
TORCH_MODULE(Wan2_2I2VPipeline);

REGISTER_DIT_MODEL(wan2_2, Wan2_2I2VPipeline);

}  // namespace xllm
