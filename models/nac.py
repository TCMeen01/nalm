import torch
import torch.nn as nn
from models.nalu import nac_w_optimal_r
from training.utils import calc_sparsity_loss

class NAC(nn.Module):
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Khởi tạo 2 ma trận tham số W_hat và M_hat
        self.W_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.M_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.to(device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        r = nac_w_optimal_r(self.in_dim, self.out_dim)
        torch.nn.init.uniform_(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform_(self.M_hat, a=-r, b=r)

    def sparsity_loss(self):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return calc_sparsity_loss(W)

    def regularization_loss(self):
        return 0

    def forward(self, x):
        x = x.to(self.device)
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        
        return torch.matmul(x, W)
    

class NAC_add(NAC):
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__(in_dim, out_dim, device)

    def forward(self, x):
        return super().forward(x)
    
class NAC_mul(NAC):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-7):
        super().__init__(in_dim, out_dim, device)
        self.epsilon = epsilon

    def forward(self, x):
        m_log = super().forward(torch.log(torch.abs(x) + self.epsilon))
        return torch.exp(m_log)