import torch
import torch.nn as nn
import torch.nn.functional as F

class NAU(nn.Module):
    """
    Neural Addition Unit (NAU)
    Paper: Neural Arithmetic Units (Madsen et al., 2020)
    
    Architecture:
    y = x * W^T
    
    Điểm khác biệt so với Linear layer thường:
    - Không có bias.
    - Không dùng activation function (như tanh/sigmoid của NAC).
    - Phụ thuộc hoàn toàn vào Sparsity Regularization trong quá trình training 
      để ép W về {-1, 0, 1}.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Ma trận trọng số W
        self.W = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo Xavier Uniform (Glorot)
        # Giúp ổn định variance của gradient
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        # NAU thực chất là phép nhân ma trận tuyến tính
        # Regularization sẽ được tính bên ngoài vòng lặp training
        return F.linear(x, self.W)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)