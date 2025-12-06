import torch
import torch.nn as nn
import torch.nn.functional as F

class NPU(nn.Module):
    """
    Complex Neural Power Unit (NPU)
    Paper: Neural Power Units (Heim et al., 2020)
    
    Logic:
    Sử dụng số phức để tính toán logarit của số âm một cách tự nhiên.
    log(x) = log(|x|) + i * pi (nếu x < 0)
    
    Output = Real(exp(W_complex * log(x_complex)))
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Trọng số là số phức: W = W_real + i * W_imag
        # Trong bài toán số học thực tế (Real arithmetic), ta thường kỳ vọng W_imag -> 0
        # Nhưng để đúng kiến trúc NPU, ta phải khai báo trọng số phức.
        self.W_real = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.W_imag = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Khởi tạo Xavier cho phần thực
        nn.init.xavier_uniform_(self.W_real)
        # Phần ảo thường khởi tạo nhỏ hoặc 0
        nn.init.uniform_(self.W_imag, -0.1, 0.1)

    def forward(self, x):
        # 1. Chuyển input sang số phức
        # PyTorch tự động xử lý log số âm thành số phức: log(-1) = i*pi
        x_complex = x.type(torch.complex64)
        
        # 2. Tạo trọng số phức
        W = torch.complex(self.W_real, self.W_imag)
        
        # 3. Tính toán: y = exp(W * log(x))
        # log(x_complex)
        log_x = torch.log(x_complex)
        
        # Linear transform: W * log(x)
        # F.linear hỗ trợ input/weight phức
        z = F.linear(log_x, W)
        
        # Exp
        y_complex = torch.exp(z)
        
        # 4. Lấy phần thực làm output
        return y_complex.real

    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'