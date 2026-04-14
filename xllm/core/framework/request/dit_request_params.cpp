/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_request_params.h"

#include "butil/base64.h"
#include "core/common/instance_name.h"
#include "core/common/macros.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "mm_codec.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_request_id(const std::string& prefix) {
  return prefix + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::pair<int, int> splitResolution(const std::string& s) {
  size_t pos = s.find('*');
  int width = std::stoi(s.substr(0, pos));
  int height = std::stoi(s.substr(pos + 1));
  return {width, height};
}

// Decode a base64-encoded image string into a torch tensor via OpenCV.
bool decodeBase64Image(const std::string& base64, torch::Tensor& out) {
  std::string raw_bytes;
  if (!butil::Base64Decode(base64, &raw_bytes)) {
    LOG(ERROR) << "Base64 decode failed";
    return false;
  }
  OpenCVImageDecoder decoder;
  if (!decoder.decode(raw_bytes, out)) {
    LOG(ERROR) << "Image decode failed";
    return false;
  }
  return true;
}

// Shared helper: populate DiTInputParams from an image-like input proto.
// Both ImageInput and VideoInput have the same prompt/embed/image fields.
template <typename InputProto>
void fillInputParams(DiTInputParams& input_params, const InputProto& input) {
  input_params.prompt = input.prompt();
  if (input.has_prompt_2()) {
    input_params.prompt_2 = input.prompt_2();
  }
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_negative_prompt_2()) {
    input_params.negative_prompt_2 = input.negative_prompt_2();
  }
  if (input.has_prompt_embed()) {
    input_params.prompt_embed = util::proto_to_torch(input.prompt_embed());
  }
  if (input.has_pooled_prompt_embed()) {
    input_params.pooled_prompt_embed =
        util::proto_to_torch(input.pooled_prompt_embed());
  }
  if (input.has_negative_prompt_embed()) {
    input_params.negative_prompt_embed =
        util::proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_negative_pooled_prompt_embed()) {
    input_params.negative_pooled_prompt_embed =
        util::proto_to_torch(input.negative_pooled_prompt_embed());
  }
  if constexpr (std::is_same_v<InputProto, proto::Input>) {
    if (input.has_latent()) {
      input_params.latent = util::proto_to_torch(input.latent());
    }
    if (input.has_masked_image_latent()) {
      input_params.masked_image_latent =
          util::proto_to_torch(input.masked_image_latent());
    }
    if (input.has_mask_image()) {
      decodeBase64Image(input.mask_image(), input_params.mask_image);
    }
  }
  if (input.has_image()) {
    decodeBase64Image(input.image(), input_params.image);
  }
  if (input.has_condition_image()) {
    decodeBase64Image(input.condition_image(), input_params.condition_image);
  }
  if (input.has_control_image()) {
    decodeBase64Image(input.control_image(), input_params.control_image);
  }
}

// Shared helper: populate generation params from a parameters proto.
// Both ImageParameters and VideoParameters share most fields.
template <typename ParamsProto>
void fillGenerationParams(DiTGenerationParams& generation_params,
                          const ParamsProto& params) {
  if (params.has_size()) {
    auto [w, h] = splitResolution(params.size());
    generation_params.width = w;
    generation_params.height = h;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_enable_cfg_renorm()) {
    generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
  }
  if (params.has_cfg_renorm_min()) {
    generation_params.cfg_renorm_min = params.cfg_renorm_min();
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Image generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("imggen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    fillInputParams(input_params, request.input());
  }

  generation_params.num_images_per_prompt = 1;
  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fillGenerationParams(generation_params, params);
    if (params.has_num_images_per_prompt()) {
      generation_params.num_images_per_prompt =
          static_cast<uint32_t>(params.num_images_per_prompt());
    }
  }
}

// ---------------------------------------------------------------------------
// Video generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::VideoGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("vidgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  generation_params.force_video_output = true;

  if (request.has_input()) {
    fillInputParams(input_params, request.input());
  }

  generation_params.num_images_per_prompt = 1;
  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fillGenerationParams(generation_params, params);
    if (params.has_num_videos_per_prompt()) {
      generation_params.num_images_per_prompt =
          static_cast<uint32_t>(params.num_videos_per_prompt());
    }
    if (params.has_num_frames()) {
      generation_params.num_frames = params.num_frames();
    }
    if (params.has_fps()) {
      generation_params.video_fps = params.fps();
    }
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (input_params.prompt.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  return true;
}

}  // namespace xllm
