import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RealNPU(nn.Module):
    """
    Real Neural Power Unit (RealNPU)
    Paper: Neural Power Units (Heim et al., 2020)
    
    Logic:
    Mô phỏng phép nhân lũy thừa y = product(x_i ^ w_i) cho cả số âm.
    Tách biệt xử lý Độ lớn (Magnitude) và Dấu (Sign).
    
    Công thức:
    Magnitude = exp(W * log(|x| + eps))
    Sign      = cos(pi * W * (x < 0))
    Output    = Magnitude * Sign
    """
    def __init__(self, in_dim, out_dim, epsilon=1e-7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon

        # NPU chỉ dùng 1 ma trận trọng số W để học số mũ
        self.W = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier Uniform
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        # 1. Xử lý Magnitude (Độ lớn)
        # Log(|x|)
        x_abs = torch.abs(x) + self.epsilon
        log_x = torch.log(x_abs)
        
        # exp(W * log_x)
        magnitude = torch.exp(F.linear(log_x, self.W))
        
        # 2. Xử lý Sign (Dấu)
        # Tạo vector chỉ thị k: k=1 nếu x<0, k=0 nếu x>=0
        k = (x < 0).float()
        
        # Tính tổng trọng số của các số âm: g = W * k
        sign_input = F.linear(k, self.W)
        
        # Sign = cos(pi * g)
        # Nếu g là số chẵn -> cos = 1 (Dương)
        # Nếu g là số lẻ -> cos = -1 (Âm)
        # NPU hoạt động tốt nhất khi W hội tụ về số nguyên.
        sign = torch.cos(math.pi * sign_input)
        
        # 3. Kết hợp
        return magnitude * sign

    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'