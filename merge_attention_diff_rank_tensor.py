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

base_dir = Path("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/08_cpp_single_inner_parallelattention_tensor")
rank0_dir = base_dir / "rank0"
rank1_dir = base_dir / "rank1"
output_dir = base_dir / "total_attention_tensor"

output_dir.mkdir(parents=True, exist_ok=True)

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
    output_path = output_dir / filename
    
    module0 = torch.jit.load(rank0_path, map_location='cpu')
    module1 = torch.jit.load(rank1_path, map_location='cpu')
    
    tensor0 = list(module0.parameters())[0]
    tensor1 = list(module1.parameters())[0]
    
    merged_tensor = torch.cat([tensor0, tensor1], dim=-1)
    
    print(f"Processing {filename}:")
    print(f"  rank0 shape: {tensor0.shape}")
    print(f"  rank1 shape: {tensor1.shape}")
    print(f"  merged shape: {merged_tensor.shape}")
    
    container = TensorContainer(merged_tensor)
    scripted_module = torch.jit.script(container)
    scripted_module.save(output_path)
    print(f"  Saved to {output_path}")

print("\nDone!")
