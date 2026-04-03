# # ------------------------------1、计算两个c++侧的的余弦相似度----------------------------------

# import torch
# from torch.nn.functional import cosine_similarity
# # c++1侧save
# hidden_cpp_1 = torch.jit.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank0/08_01_save_qkv_mlp_output_0.pt", map_location= "cpu")
# hidden_cpp_1 = list(hidden_cpp_1.parameters())[0].to("cpu").float()
# # c++2侧save
# hidden_cpp_2 = torch.jit.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank1/08_01_save_qkv_mlp_output_0.pt", map_location= "cpu")
# hidden_cpp_2 = list(hidden_cpp_2.parameters())[0].to("cpu").float()



# max_abs_diff = (hidden_cpp_2-hidden_cpp_1).abs().max().item()

# sim = cosine_similarity(hidden_cpp_2, hidden_cpp_1, dim=-1)
# print(sim)  ## 99.99%
# print(max_abs_diff) ## 0.00x


## ------------------------------2、计算c++、python侧的的余弦相似度----------------------------------

import torch
from torch.nn.functional import cosine_similarity

# ===================== 你的原有加载代码（完全不变） =====================
# # c++侧save
hidden_cpp = torch.jit.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor/rank0/08_11_save_output_0.pt", map_location="cpu")

hidden_cpp = list(hidden_cpp.parameters())[0].to("cpu").float()

# python侧save
hidden_py = torch.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/08_python_single_inner_parallel_attention_tensor/08_11_python_hidden_states_0.pt", map_location="cpu")
print("hidden_py.shape:",hidden_py.shape)
# ===================== 【关键改动】计算整体余弦相似度 =====================
# 展平成一维 → 计算全局整体相似度
sim = cosine_similarity(hidden_py, hidden_cpp, dim=-1)

# 最大绝对误差（不变）
max_abs_diff = (hidden_py - hidden_cpp).abs().max().item()

# ===================== 输出结果 =====================
# print("python与C++整体余弦相似度: {:.8f}".format(sim))
# print("python与C++最大绝对误差: {:.8f}".format(max_abs_diff))

print(sim)  ## 99.99%
print(max_abs_diff) ## 0.00x

'''
# python侧1111111111111 save
hidden_cpp_1 = torch.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/02_after_dit_tensor/03_00_0_python_test_concat_rotary_emb_0.pt", map_location="cpu")

# python侧2222222222222 save
hidden_py_1 = torch.load("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/02_after_dit_tensor/03_00_0_python_test_concat_rotary_emb_1.pt", map_location="cpu")

# ===================== 【关键改动】计算整体余弦相似度 =====================
# 展平成一维 → 计算全局整体相似度
sim = cosine_similarity(hidden_py_1.flatten(), hidden_cpp_1.flatten(), dim=-1).item()

# 最大绝对误差（不变）
max_abs_diff = (hidden_py_1 - hidden_cpp_1).abs().max().item()

# ===================== 输出结果 =====================
print("两个python整体余弦相似度: {:.8f}".format(sim))
print("两个python最大绝对误差: {:.8f}".format(max_abs_diff))
'''
