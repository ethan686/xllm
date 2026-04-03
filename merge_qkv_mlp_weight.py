import os
import torch
import torch.nn as nn
from pathlib import Path

class TensorContainer(nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(tensor, requires_grad=False)
    
    def forward(self):
        return self.param

base_dir = Path("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/01_single_parallel_weight")
rank0_dir = base_dir / "rank0"
rank1_dir = base_dir / "rank1"
output_dir = base_dir / "merged_qkv_mlp_tensor"

output_dir.mkdir(parents=True, exist_ok=True)

# 需要修改的值
qkv_size = 9216
mlp_size = 18432
split_dim = -2  # -1 表示倒数第一维，-2 表示倒数第二维

rank0_files = sorted([f for f in os.listdir(rank0_dir) if f.endswith('.pt')])
rank1_files = sorted([f for f in os.listdir(rank1_dir) if f.endswith('.pt')])

print(f"rank0 files count: {len(rank0_files)}")
print(f"rank1 files count: {len(rank1_files)}")

rank0_files_set = set(rank0_files)
rank1_files_set = set(rank1_files)
common_files = rank0_files_set & rank1_files_set
common_files = sorted(common_files)

print(f"Common files count: {len(common_files)}")

for filename in common_files:
    rank0_path = rank0_dir / filename
    rank1_path = rank1_dir / filename
    
    module0 = torch.jit.load(rank0_path, map_location='cpu')
    module1 = torch.jit.load(rank1_path, map_location='cpu')
    
    tensor0 = list(module0.parameters())[0]
    tensor1 = list(module1.parameters())[0]
    
    print(f"\nProcessing {filename}:")
    print(f"  rank0 shape: {tensor0.shape}")
    print(f"  rank1 shape: {tensor1.shape}")
    
    # 使用 torch.narrow 在指定维度上切分
    # torch.narrow(input, dim, start, length) 在 dim 维度上从 start 开始切分 length 个元素
    # 这行代码的作用是将 负数维度索引转换为正数维度索引 ，因为 torch.narrow() 函数只接受正数维度索引。
    actual_dim = split_dim if split_dim >= 0 else tensor0.dim() + split_dim
    
    qkv0 = torch.narrow(tensor0, actual_dim, 0, qkv_size)
    mlp0 = torch.narrow(tensor0, actual_dim, qkv_size, mlp_size)
    qkv1 = torch.narrow(tensor1, actual_dim, 0, qkv_size)
    mlp1 = torch.narrow(tensor1, actual_dim, qkv_size, mlp_size)   
    
    print(f"  qkv0 shape: {qkv0.shape}")
    print(f"  mlp0 shape: {mlp0.shape}")
    print(f"  qkv1 shape: {qkv1.shape}")
    print(f"  mlp1 shape: {mlp1.shape}")
    
    merged_qkv = torch.cat([qkv0, qkv1], dim=split_dim)
    merged_mlp = torch.cat([mlp0, mlp1], dim=split_dim)
    
    print(f"  merged_qkv shape: {merged_qkv.shape}")
    print(f"  merged_mlp shape: {merged_mlp.shape}")
    
    base_name = filename.rsplit('.', 1)[0]
    
    qkv_container = TensorContainer(merged_qkv)
    qkv_module = torch.jit.script(qkv_container)
    qkv_output_path = output_dir / f"{base_name}_qkv.pt"
    qkv_module.save(qkv_output_path)
    print(f"  Saved qkv to {qkv_output_path}")
    
    mlp_container = TensorContainer(merged_mlp)
    mlp_module = torch.jit.script(mlp_container)
    mlp_output_path = output_dir / f"{base_name}_mlp.pt"
    mlp_module.save(mlp_output_path)
    print(f"  Saved mlp to {mlp_output_path}")

print("\nDone!")
