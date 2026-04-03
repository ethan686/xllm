# -*- coding: utf-8 -*-
import sys
import time
import requests
import json
import numpy as np
from typing import Callable, Optional, Union
from safetensors.torch import load_file
import io
import base64
import PIL.Image
import PIL.ImageOps
import torch
import os
# 1. 配置后端地址（根据实际部署修改IP和端口）
api_base = "http://127.0.0.1:18018" # 后端服务基础地址
api_endpoint = f"{api_base}/v1/image/generation" # 图像生成接口路径
model_name = "flux2" # 模型名，需与后端支持的模型匹配（flux-schnell/flux-dev）

def load_tensor(
    image: Union[str, PIL.Image.Image],
    convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Load `image` (URL / local path / PIL.Image) and convert to torch.Tensor.

    Args:
        image (str or PIL.Image.Image): URL (http/https) or filesystem path, or PIL Image.
        convert_method (Callable, optional): 如果提供，会在读取后对 PIL.Image 做自定义转换并返回 PIL.Image。
            若为 None，则会默认调用 `.convert("RGB")`。
        device (torch.device, optional): 将返回 tensor 放到哪个 device（例如 torch.device('cuda')）。
            若为 None，则不做 device 转移（默认 CPU）。
        dtype (torch.dtype, optional): 返回 tensor 的 dtype（例如 torch.float32）。若为 None，则使用 torch.float32。

    Returns:
        torch.Tensor: shape [C, H, W], dtype float, values in [0,1], 在指定 device（若提供）。
    """
    # 读取为 PIL.Image
    if isinstance(image, str):
        # if image.startswith("http://") or image.startswith("https://"):
        #     # 从网络读取流式数据
        #     resp = requests.get(image, stream=True, timeout=500)
        #     resp.raise_for_status()
        #     pil_image = PIL.Image.open(resp.raw)
        # elif os.path.isfile(image):
        if os.path.isfile(image):
            pil_image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    # 处理 EXIF 方向
    pil_image = PIL.ImageOps.exif_transpose(pil_image)

    # 自定义转换或默认 RGB
    if convert_method is not None:
        pil_image = convert_method(pil_image)
    else:
        pil_image = pil_image.convert("RGB")

    # 转 numpy 再转 tensor；确保是 contiguous 并复制内存避免引用 PIL 缓冲
    np_img = np.asarray(pil_image, dtype=np.float32)  # H x W x C, float32
    # 若是灰度单通道，扩展通道
    if np_img.ndim == 2:
        np_img = np_img[:, :, None]
    if np_img.shape[2] == 4:
        # RGBA -> RGB（简单裁掉 alpha），若需按 alpha 合成请自定义 convert_method
        np_img = np_img[:, :, :3]

    # 归一化到 [0,1]
    np_img = np_img / 255.0

    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # C x H x W

    # dtype & device
    target_dtype = dtype or torch.float32
    tensor = tensor.to(dtype=target_dtype)
    # if device is not None:
    #     tensor = tensor.to(device)

    # clone 确保独立内存（可选，但保险）
    return tensor.clone()

def base64_to_image(base64_string, output_path):
    """
    将Base64字符串保存为图片文件
    
    Args:
        base64_string: Base64编码的字符串
        output_path: 输出图片路径（如：'output.jpg', 'output.png'）
    """
    try:
        # 解码Base64字符串
        image_data = base64.b64decode(base64_string)
        
        # 保存为文件
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
        print(f"图片已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"转换失败: {e}")
        return None

def image_to_base64(img: Union[str, PIL.Image.Image]) -> str:
    """
    将图片文件路径或 PIL.Image 转成 Base64 字符串
    """
    if isinstance(img, str):
        pil_image = PIL.Image.open(img)
    elif isinstance(img, PIL.Image.Image):
        pil_image = img
    else:
        raise ValueError("img必须是文件路径或PIL.Image对象")

    # 转RGB
    pil_image = PIL.ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    # 保存到内存 buffer
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    b64_str = base64.b64encode(byte_data).decode("utf-8")
    return b64_str

def create_tensor(data, name, datatype="FP32"):
        """
        构造符合proto::Tensor格式的字典（修复后：直接对应Proto的4个顶层字段）

        Args:
        data: numpy数组或Python列表，张量数据
        name: 张量名称（对应Proto的name字段）
        datatype: 数据类型（对应Proto的datatype字段），默认FP32

        Returns:
        dict: 完全匹配proto::Tensor结构的字典
        """
        # 转换为numpy数组以便获取形状
        if not isinstance(data, np.ndarray):
                data = np.array(data)

        # 1. 处理形状：确保为正整数（避免后端报无效维度错误）
        shape = list(data.shape)
        print(shape)
        if any(dim <= 0 for dim in shape):
                raise ValueError(f"张量{name}的形状包含非正整数：{shape}，需全部为正")

        # 2. 处理数据：展平后存入对应类型的contents字段
        contents = {}
        flat_data = data.flatten().tolist()
        if datatype == "FP32":
                contents["fp32_contents"] = flat_data # 对应Proto的TensorContents.fp32_contents
        elif datatype == "INT64":
                contents["int64_contents"] = flat_data
        elif datatype == "BOOL":
                contents["bool_contents"] = flat_data
        else:
                raise ValueError(f"不支持的数据类型：{datatype}，仅支持FP32/INT64/BOOL")

        # 3. 直接返回Proto要求的4个顶层字段（无多余层级）
        return {
        "name": name, # 顶层name字段
        "datatype": datatype, # 顶层datatype字段（修复报错的核心）
        "shape": shape, # 顶层shape字段（修复size[0]的核心）
        "contents": contents # 顶层contents字段
        }


def test_image_generation():
        """测试图像生成接口（使用修复后的Tensor结构）"""
        st = time.time()
        try:
                # 生成示例嵌入向量（形状需符合模型要求，此处保持原逻辑）
                pooled_prompt_embeds = np.random.rand(768).astype(np.float32) # 1D: [768]
                prompt_embeds = np.random.rand(2, 768).astype(np.float32) # 2D: [2, 768]
                ip_adapter_image_embeds = np.random.rand(1, 4, 768).astype(np.float32) # 3D: [1,4,768]
                latents = np.ones((1, 4, 32, 32), dtype=np.float32) # 4D: [1,4,32,32]（确保shape全部为正）

                # 2. 构造请求参数（Tensor结构已修复，其他逻辑不变
                payload = {
                "model": model_name,
                "input": {
                "prompt": "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colo",
                "prompt_2": "soft lighting, watercolor texture",
                "negative_prompt": " ",
                "negative_prompt_2": "",
                },
                "parameters": {
                "size": "1024*1024",
                "num_inference_steps": 50, # 注意：flux-schnell推荐4步，dev推荐50步，28步可能非最优
                "guidance_scale": 3.5,
                "true_cfg_scale": 3.0,
                "num_images_per_prompt": 1,
                "seed": 42,
                "max_sequence_length": 2048
                },
                "user": "test_user",
                "service_request_id": f"req-{int(time.time())}"
                }


                # 3. 发送请求（后续逻辑不变）
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                url=api_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=60 * 5
                )
                response.raise_for_status()
                result = response.json()

                # 4. 解析响应（后续逻辑不变）
                print(f"接口响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                print(f"请求耗时: {time.time() - st:.2f}s")
                if result.get("output") and result["output"].get("results"):
                        for idx, image_result in enumerate(result["output"]["results"]):
                                print(f"\n生成图片 {idx + 1}:")
                if image_result.get("url"):
                        print(f"URL: {image_result['url']}")
                elif image_result.get("image"):
                        print(f"尺寸: {image_result.get('width')}x{image_result.get('height')}")
                        base64_to_image(image_result['image'], "./result.png")
                else:
                        print(f"生成失败: {result.get('message', '未返回结果')}")

        except requests.exceptions.RequestException as e:
                print(f"请求异常: {str(e)}")
        except json.JSONDecodeError:
                print("响应格式错误，无法解析为JSON")
        except Exception as e:
                print(f"处理失败: {str(e)}")


if __name__ == "__main__":
        test_image_generation()
