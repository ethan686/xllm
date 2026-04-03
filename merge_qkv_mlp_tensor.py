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

def merge_attention_diff_rank_tensor():
    base_dir = Path("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/dump_flux2_tensor/01_single_parallel_weight")
    rank0_dir = base_dir / "rank0"
    rank1_dir = base_dir / "rank1"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    qkv_size = 9216
    mlp_size = 18432

    split_dim = -2

    actual_dim = split_dim if split_dim >= 0 else tensor.dim() + split_dim
    total_size = tensor.size(actual_dim)
    chunk_size = total_size // 3

    q_tensor = torch.narrow(tensor, actual_dim, 0, chunk_size)
    k_tensor = torch.narrow(tensor, actual_dim, chunk_size, chunk_size)
    v_tensor = torch.narrow(tensor, actual_dim, chunk_size * 2, chunk_size)
            
            mlp_tensor = torch.narrow(tensor, actual_dim, qkv_size, mlp_size)
            
            print(f"Q tensor shape: {q_tensor.shape}")
            print(f"K tensor shape: {k_tensor.shape}")
            print(f"V tensor shape: {v_tensor.shape}")
            
            q_container = TensorContainer(q_tensor)
            q_module = torch.jit.script(q_container)
            q_output_path = output_dir / f"{base_name}_qkv.pt"
            q_module.save(q_output_path)
            print(f"Saved Q to {q_output_path}")
            
            k_container = TensorContainer(k_tensor)
            k_module = torch.jit.script(k_container)
            k_output_path = output_dir / f"{base_name}_kkv.pt"
            k_module.save(k_output_path)
            print(f"Saved K to {k_output_path}")
            
            v_container = TensorContainer(v_tensor)
            v_module = torch.jit.script(v_container)
            v_output_path = output_dir / f"{base_name}_qkv.pt"
            v_module.save(v_output_path)
            print(f"Saved V to {v_output_path}")

        else:
            print("Warning: No common files found in rank0 and rank1 目录")

            continue

    print("Done!")
else:
    print("Warning: 目录不存在，跳过处理")
print(f"Done!")
