import os
import torch
from pathlib import Path

base_dir = Path("/export/home/weinan5/wangshuibin/10_new_flux2_tp_xllm/02_python_tensor/01_python_single_attn_qkv_mlp_dump_weight")
input_file = base_dir / "01_01_qkv_weight.pt"
output_dir = base_dir / "split_python_qkv_weight"

output_dir.mkdir(parents=True, exist_ok=True)

split_dim = -2

tensor = torch.load(input_file, map_location='cpu')
print(f"Input tensor shape: {tensor.shape}")

actual_dim = split_dim if split_dim >= 0 else tensor.dim() + split_dim
total_size = tensor.size(actual_dim)
chunk_size = total_size // 3

print(f"Total size on dim {actual_dim}: {total_size}")
print(f"Chunk size: {chunk_size}")

q_tensor = torch.narrow(tensor, actual_dim, 0, chunk_size)
k_tensor = torch.narrow(tensor, actual_dim, chunk_size, chunk_size)
v_tensor = torch.narrow(tensor, actual_dim, chunk_size * 2, chunk_size)

print(f"Q tensor shape: {q_tensor.shape}")
print(f"K tensor shape: {k_tensor.shape}")
print(f"V tensor shape: {v_tensor.shape}")

torch.save(q_tensor, output_dir / "01_01_q_weight.pt")
print(f"Saved Q to {output_dir / '01_01_q_weight.pt'}")

torch.save(k_tensor, output_dir / "01_02_k_weight.pt")
print(f"Saved K to {output_dir / '01_02_k_weight.pt'}")

torch.save(v_tensor, output_dir / "01_03_v_weight.pt")
print(f"Saved V to {output_dir / '01_03_v_weight.pt'}")

print("\nDone!")
