import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.nac import NAC

class NALU(nn.Module):
    """
    Neural Arithmetic Logic Unit (NALU)
    Paper: Neural Arithmetic Logic Units (Trask et al., 2018)
    
    Architecture:
    1. Arithmetic Path (a): NAC(x) -> Học cộng/trừ
    2. Multiplicative Path (m): exp(NAC(log(|x| + eps))) -> Học nhân/chia
    3. Gate (g): sigmoid(G * x) -> Chọn giữa (a) và (m)
    
    Output: y = g * a + (1 - g) * m
    """
    def __init__(self, in_dim, out_dim, epsilon=1e-7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon

        # 1. Arithmetic Path (NAC cho cộng trừ)
        self.nac_add = NAC(in_dim, out_dim)

        # 2. Multiplicative Path (NAC cho nhân chia - hoạt động trong log space)
        # Lưu ý: Ta dùng một instance NAC riêng biệt. 
        # (Một số implementation cũ dùng chung weight với nac_add, nhưng benchmark chuẩn thường tách ra)
        self.nac_log = NAC(in_dim, out_dim)

        # 3. Gate Parameter (G)
        # Học trọng số để chọn operation
        self.G = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo tham số cho Gate
        nn.init.xavier_uniform_(self.G)
        
        # Lưu ý: NAC con đã tự reset trong __init__ của nó, nhưng gọi lại cũng không sao
        self.nac_add.reset_parameters()
        self.nac_log.reset_parameters()

    def forward(self, x):
        # --- 1. Arithmetic Path (a) ---
        a = self.nac_add(x)

        # --- 2. Multiplicative Path (m) ---
        # Chuyển input sang log space: log(|x| + epsilon)
        # Epsilon cực kỳ quan trọng để tránh log(0) -> NaN
        x_abs = torch.abs(x)
        log_x = torch.log(x_abs + self.epsilon)
        
        # Cho qua NAC log
        m_log = self.nac_log(log_x)
        
        # Chuyển ngược lại bằng exp
        m = torch.exp(m_log)

        # --- 3. Gate (g) ---
        # g = sigmoid(x * G^T)
        g = torch.sigmoid(F.linear(x, self.G))

        # --- 4. Combine ---
        # Kết hợp kết quả dựa trên cổng
        y = g * a + (1 - g) * m
        
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)