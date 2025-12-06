import torch
import torch.nn as nn
import math

class NMU(nn.Module):
    """
    Neural Multiplication Unit (NMU)
    Paper: Neural Arithmetic Units (Madsen et al., 2020)
    
    Logic:
    y = Product_over_inputs(W * x + (1 - W))
    
    Yêu cầu: 
    - W phải nằm trong khoảng [0, 1].
    - W đóng vai trò như một "cổng mềm" (soft gate):
        + W -> 1: Chọn input đó để nhân.
        + W -> 0: Bỏ qua input đó (nhân với 1).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo trọng số trong khoảng [0, 0.5] hoặc [0, 1]
        # Madsen et al. khuyến nghị khởi tạo uniform quanh 0.5 hoặc nhỏ hơn 
        # để bắt đầu "trung lập".
        nn.init.uniform_(self.W, 0, 0.5)

    def forward(self, x):
        # 1. Kẹp trọng số (Clamping)
        # Bắt buộc W phải thuộc [0, 1] để logic nhân hoạt động đúng.
        # Dù có regularization, ta vẫn cần kẹp trong forward pass để đảm bảo tính toán.
        W_clamped = torch.clamp(self.W, 0.0, 1.0)
        
        # 2. Chuẩn bị Broadcasting
        # Input x: (Batch_Size, In_Dim)
        # Weight W: (Out_Dim, In_Dim)
        # Output mong muốn: (Batch_Size, Out_Dim)
        
        # Ta cần tạo ra tensor trung gian để nhân element-wise:
        # x -> (Batch, 1, In)
        # W -> (1, Out, In)
        
        # Công thức: term = W * x + (1 - W)
        # Broadcasting tự động của PyTorch:
        # (Batch, 1, In) * (Out, In) -> (Batch, Out, In)
        
        # x.unsqueeze(1) biến x thành shape (Batch, 1, In_Dim)
        input_term = x.unsqueeze(1) 
        
        # Tính toán các thành phần nhân tử
        # term_ij = w_ij * x_j + (1 - w_ij)
        terms = W_clamped * input_term + (1.0 - W_clamped)
        
        # 3. Tính tích (Product)
        # Nhân dọc theo chiều input (dim=2) để ra kết quả cuối cùng
        y = torch.prod(terms, dim=2)
        
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)