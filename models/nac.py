import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NAC(nn.Module):
    """
    Neural Accumulator (NAC)
    Paper: Neural Arithmetic Logic Units (Trask et al., 2018)
    
    Logic:
    W = tanh(W_hat) * sigmoid(M_hat)
    y = x * W^T
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Khởi tạo 2 ma trận tham số W_hat và M_hat
        self.W_hat = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = nn.Parameter(torch.Tensor(out_dim, in_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo Xavier Uniform (Glorot) được khuyến nghị trong paper benchmark
        # giúp gradient lan truyền tốt hơn qua tanh và sigmoid
        nn.init.xavier_uniform_(self.W_hat)
        nn.init.xavier_uniform_(self.M_hat)

    def forward(self, x):
        # Tính trọng số W ép về khoảng {-1, 0, 1}
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        
        # Thực hiện phép biến đổi tuyến tính: y = xW^T
        return F.linear(x, W)

    def extra_repr(self):
        # Hiển thị thông tin khi print(model)
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)