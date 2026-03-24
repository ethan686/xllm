import torch
from torch.nn.functional import cosine_similarity
# c++侧save
hidden_cpp = torch.jit.load("xx.pt", map_location= "cpu")
hidden_cpp = list(hidden_cpp.parameters())[0].to("cpu").float()
# python侧save
hidden_py = torch.load() 

max_abs_diff = (hidden_py-hidden_cpp).abs().max().item()

sim = cosine_similarity(hidden_py, hidden_cpp, dim=-1)
print(sim)  ## 99.99%
print(max_abs_diff) ## 0.00x
