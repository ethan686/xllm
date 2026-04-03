'''import torch
from torch.nn.functional import cosine_similarity
import os
import re
from pathlib import Path

def extract_sequence_number(filename):
    """
    从文件名中提取序号（如 '03_00'）
    支持格式：03_00_0_CPP_... 或 03_00_0_python_...
    """
    # 匹配模式：数字_数字_数字（如 03_00_0）
    match = re.search(r'(\d+_\d+_\d+)', filename)
    if match:
        return match.group(1)
    # 备用模式：数字_数字（如 03_00）
    match = re.search(r'(\d+_\d+)', filename)
    if match:
        return match.group(1)
    return None

def load_tensor_pair(cpp_dir, py_dir, seq_num):
    """
    加载指定序号的C++和Python张量
    """
    # 在cpp_dir中查找包含seq_num的文件
    cpp_files = [f for f in os.listdir(cpp_dir) if seq_num in f and f.endswith('.pt')]
    # 在py_dir中查找包含seq_num的文件
    py_files = [f for f in os.listdir(py_dir) if seq_num in f and f.endswith('.pt')]
    
    if not cpp_files or not py_files:
        return None, None, None, None
    
    # 取第一个匹配的文件（假设每个序号只有一个文件）
    cpp_file = os.path.join(cpp_dir, cpp_files[0])
    py_file = os.path.join(py_dir, py_files[0])
    
    # 加载张量
    try:
        # C++侧保存的张量
        hidden_cpp = torch.jit.load(cpp_file, map_location="cpu")
        hidden_cpp = list(hidden_cpp.parameters())[0].to("cpu").float()
        
        # Python侧保存的张量
        hidden_py = torch.load(py_file, map_location="cpu")
        
        return hidden_cpp, hidden_py, cpp_file, py_file
    except Exception as e:
        print(f"加载文件时出错 [{seq_num}]: {e}")
        return None, None, None, None

def compare_tensors(hidden_cpp, hidden_py, seq_num):
    """
    计算余弦相似度和最大绝对误差
    """
    # 展平成一维 → 计算全局整体相似度
    sim = cosine_similarity(hidden_py.flatten(), hidden_cpp.flatten(), dim=-1).item()
    
    # 最大绝对误差
    max_abs_diff = (hidden_py - hidden_cpp).abs().max().item()
    
    return sim, max_abs_diff

def main():
    # 目录路径
    cpp_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/04_double_stream_transformer_tensor"
    py_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/04_python_double_stream_transformer_tensor"
    
    # 验证目录是否存在
    if not os.path.exists(cpp_dir):
        print(f"错误：C++目录不存在: {cpp_dir}")
        return
    if not os.path.exists(py_dir):
        print(f"错误：Python目录不存在: {py_dir}")
        return
    
    # 获取所有唯一的序号
    cpp_files = [f for f in os.listdir(cpp_dir) if f.endswith('.pt')]
    py_files = [f for f in os.listdir(py_dir) if f.endswith('.pt')]
    
    # 提取所有序号
    cpp_seqs = set()
    for f in cpp_files:
        seq = extract_sequence_number(f)
        if seq:
            cpp_seqs.add(seq)
    
    py_seqs = set()
    for f in py_files:
        seq = extract_sequence_number(f)
        if seq:
            py_seqs.add(seq)
    
    # 取交集（两个目录都存在的序号）
    common_seqs = sorted(list(cpp_seqs.intersection(py_seqs)))
    
    if not common_seqs:
        print("警告：未找到两个目录中共同的序号")
        return
    
    print(f"找到 {len(common_seqs)} 个共同序号: {common_seqs}")
    print("=" * 80)
    
    # 存储结果
    results = []
    
    # 按序号顺序处理
    for seq_num in common_seqs:
        print(f"处理序号: {seq_num}")
        
        # 加载张量对
        hidden_cpp, hidden_py, cpp_file, py_file = load_tensor_pair(cpp_dir, py_dir, seq_num)
        
        if hidden_cpp is None or hidden_py is None:
            print(f"  ⚠️  跳过序号 {seq_num}（文件加载失败）")
            continue
        
        # 计算相似度和误差
        sim, max_abs_diff = compare_tensors(hidden_cpp, hidden_py, seq_num)
        
        # 记录结果
        results.append({
            'seq_num': seq_num,
            'cpp_file': os.path.basename(cpp_file),
            'py_file': os.path.basename(py_file),
            'cosine_similarity': sim,
            'max_abs_error': max_abs_diff,
            'cpp_shape': list(hidden_cpp.shape),
            'py_shape': list(hidden_py.shape)
        })
        
        # 输出当前结果
        print(f"  C++文件: {os.path.basename(cpp_file)}")
        print(f"  Python文件: {os.path.basename(py_file)}")
        print(f"  余弦相似度: {sim:.8f}")
        print(f"  最大绝对误差: {max_abs_diff:.8f}")
        print(f"  C++形状: {hidden_cpp.shape}")
        print(f"  Python形状: {hidden_py.shape}")
        print("-" * 60)
    
    # 输出汇总报告
    if results:
        print("\n" + "=" * 80)
        print("汇总报告")
        print("=" * 80)
        
        # 按相似度排序
        sorted_by_sim = sorted(results, key=lambda x: x['cosine_similarity'], reverse=True)
        sorted_by_err = sorted(results, key=lambda x: x['max_abs_error'])
        
        print(f"\n📊 统计信息:")
        print(f"  总文件对数: {len(results)}")
        print(f"  平均余弦相似度: {sum(r['cosine_similarity'] for r in results)/len(results):.8f}")
        print(f"  平均最大绝对误差: {sum(r['max_abs_error'] for r in results)/len(results):.8f}")
        
        print(f"\n🏆 最高相似度（前3）:")
        for i, r in enumerate(sorted_by_sim[:3], 1):
            print(f"  {i}. 序号 {r['seq_num']}: {r['cosine_similarity']:.8f}")
        
        print(f"\n⚠️  最大误差（前3）:")
        for i, r in enumerate(sorted_by_err[:3], 1):
            print(f"  {i}. 序号 {r['seq_num']}: {r['max_abs_error']:.8f}")
        
        # 输出详细表格
        print(f"\n📋 详细结果:")
        print("-" * 100)
        print(f"{'序号':<12} {'C++文件':<40} {'Python文件':<40} {'相似度':<12} {'最大误差':<12} {'C++形状':<20} {'Python形状':<20}")
        print("-" * 100)
        for r in results:
            cpp_name = r['cpp_file'][:37] + "..." if len(r['cpp_file']) > 37 else r['cpp_file']
            py_name = r['py_file'][:37] + "..." if len(r['py_file']) > 37 else r['py_file']
            print(f"{r['seq_num']:<12} {cpp_name:<40} {py_name:<40} {r['cosine_similarity']:<12.8f} {r['max_abs_error']:<12.8f} {str(r['cpp_shape']):<20} {str(r['py_shape']):<20}")
        
        # 保存结果到CSV文件
        import csv
        csv_file = "tensor_comparison_results.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['seq_num', 'cpp_file', 'py_file', 'cosine_similarity', 'max_abs_error', 'cpp_shape', 'py_shape'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n💾 结果已保存到: {csv_file}")
    else:
        print("未成功处理任何文件对")

if __name__ == "__main__":
    main()
'''

import torch
from torch.nn.functional import cosine_similarity
import os
import re
from pathlib import Path
import pandas as pd

def extract_base_sequence(filename):
    """
    提取基础序号（如 '03_00'），忽略后续的 _0、_CPP、_python 等后缀
    """
    # 匹配模式：数字_数字（如 03_00）
    match = re.match(r'(\d+_\d+)', filename)
    if match:
        return match.group(1)
    return None

def find_matching_files(cpp_dir, py_dir, base_seq):
    """
    根据基础序号查找匹配的C++和Python文件
    """
    cpp_files = []
    py_files = []
    
    # 遍历C++目录
    for f in os.listdir(cpp_dir):
        if f.endswith('.pt'):
            seq = extract_base_sequence(f)
            if seq == base_seq:
                cpp_files.append(f)
    
    # 遍历Python目录
    for f in os.listdir(py_dir):
        if f.endswith('.pt'):
            seq = extract_base_sequence(f)
            if seq == base_seq:
                py_files.append(f)
    
    return cpp_files, py_files

def load_and_compare(cpp_file_path, py_file_path, base_seq):
    """
    加载并比较两个张量
    """
    try:
        # 加载C++张量
        hidden_cpp = torch.jit.load(cpp_file_path, map_location="cpu")
        # 处理C++保存的模型结构
        if hasattr(hidden_cpp, 'parameters'):
            hidden_cpp = list(hidden_cpp.parameters())[0].to("cpu").float()
        else:
            hidden_cpp = hidden_cpp.to("cpu").float()
        
        # 加载Python张量
        hidden_py = torch.load(py_file_path, map_location="cpu")
        print("---------hidden_cpp.shape:", hidden_cpp.shape)
        print("---------hidden_py.shape:", hidden_py.shape)
        # 确保形状一致
        if hidden_cpp.shape != hidden_py.shape:
            print(f"警告: 形状不匹配 [{base_seq}]: C++ {hidden_cpp.shape} vs Python {hidden_py.shape}")
            # 尝试调整形状（如果需要）
            if hidden_cpp.numel() == hidden_py.numel():
                hidden_cpp = hidden_cpp.flatten()
                hidden_py = hidden_py.flatten()
            else:
                return None, None, None, None
        
        # 计算整体余弦相似度
        sim = cosine_similarity(
            hidden_cpp.flatten(), 
            hidden_py.flatten(), 
            dim=-1
        ).item()
        
        # 计算最大绝对误差
        max_abs_diff = (hidden_cpp - hidden_py).abs().max().item()
        
        return sim, max_abs_diff, hidden_cpp.shape, hidden_cpp.dtype
        
    except Exception as e:
        print(f"处理文件时出错 [{base_seq}]: {e}")
        return None, None, None, None

def main():
    # 目录路径
    cpp_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/09_cpp_flux2_dit_and_output_tensor"
    py_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/09_python_flux2_dit_and_output_tensor"
    
    # ## 01_noise_pred
    # cpp_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/10_cpp_dit_loop_noise_pred/01_noise_pred"
    # py_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/10_python_dit_loop_noise_pred/01_noise_pred"
    # ## 02_latents
    # cpp_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/10_cpp_dit_loop_noise_pred/02_prepared_latents"
    # py_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/10_python_dit_loop_noise_pred/02_latents"
    # ## 03_scheduler_latents
    # cpp_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/10_cpp_dit_loop_noise_pred/03_scheduler_out"
    # py_dir = "/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/10_python_dit_loop_noise_pred/03_scheduler_latents"
    # 获取所有基础序号
    base_seqs = set()
    
    # 从C++目录提取序号
    for f in os.listdir(cpp_dir):
        if f.endswith('.pt'):
            seq = extract_base_sequence(f)
            if seq:
                base_seqs.add(seq)
    
    # 从Python目录提取序号（确保完整性）
    for f in os.listdir(py_dir):
        if f.endswith('.pt'):
            seq = extract_base_sequence(f)
            if seq:
                base_seqs.add(seq)
    
    # 按序号排序
    sorted_seqs = sorted(base_seqs)
    
    print(f"找到 {len(sorted_seqs)} 个序号: {sorted_seqs}")
    print("=" * 80)
    
    # 存储结果
    results = []
    
    for base_seq in sorted_seqs:
        # 查找匹配文件
        cpp_files, py_files = find_matching_files(cpp_dir, py_dir, base_seq)
        
        if not cpp_files or not py_files:
            print(f"警告: 序号 {base_seq} 缺少匹配文件")
            print(f"  C++文件: {cpp_files}")
            print(f"  Python文件: {py_files}")
            continue
        
        # 取第一个匹配的文件
        cpp_file = os.path.join(cpp_dir, cpp_files[0])
        py_file = os.path.join(py_dir, py_files[0])
        
        print(f"处理序号: {base_seq}")
        print(f"  C++文件: {cpp_files[0]}")
        print(f"  Python文件: {py_files[0]}")
        
        # 加载并比较
        sim, max_abs_diff, shape, dtype = load_and_compare(cpp_file, py_file, base_seq)
        
        if sim is not None:
            results.append({
                '序号': base_seq,
                'C++文件': cpp_files[0],
                'Python文件': py_files[0],
                '余弦相似度': sim,
                '最大绝对误差': max_abs_diff,
                '张量形状': str(shape),
                '数据类型': str(dtype)
            })
            
            print(f"  余弦相似度: {sim:.8f}")
            print(f"  最大绝对误差: {max_abs_diff:.8f}")
            print(f"  张量形状: {shape}")
            print(f"  数据类型: {dtype}")
        else:
            print(f"  处理失败")
        
        print("-" * 60)

    return results

if __name__ == "__main__":
    results = main()
