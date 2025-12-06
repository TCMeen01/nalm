import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nac import NAC

class iNALU(nn.Module):
    """
    Improved Neural Arithmetic Logic Unit (iNALU)
    Paper: iNALU: Improved Neural Arithmetic Logic Unit (Schlör et al., 2020)
    
    Cải tiến so với NALU gốc:
    1. Independent Weights: Tách biệt hoàn toàn trọng số cộng và nhân (đã làm ở NALU).
    2. Input Clipping: Kẹp giá trị trước khi vào log để tránh log(0) hoặc log(số quá nhỏ).
    3. Gradient Stability: Giới hạn giá trị mũ để tránh tràn số (overflow).
    """
    def __init__(self, in_dim, out_dim, epsilon=1e-7, omega=20):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.omega = omega # Giới hạn trên cho lũy thừa (exp) để tránh overflow

        # 1. Arithmetic Path (Cộng/Trừ) - Giống NAC
        self.nac_add = NAC(in_dim, out_dim)

        # 2. Multiplicative Path (Nhân/Chia) - Log Space
        self.nac_mul = NAC(in_dim, out_dim)

        # 3. Gate
        # iNALU thường giữ nguyên cấu trúc gate như NALU
        self.G = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo Xavier
        nn.init.xavier_uniform_(self.G)
        self.nac_add.reset_parameters()
        self.nac_mul.reset_parameters()

    def forward(self, x):
        # --- Path 1: Addition ---
        a = self.nac_add(x)

        # --- Path 2: Multiplication ---
        # Cải tiến quan trọng 1: Safe Log
        # Thay vì chỉ cộng epsilon, iNALU xử lý kỹ hơn
        # Tuy nhiên, để tương thích benchmark, ta dùng công thức log(|x| + eps)
        # và kẹp giá trị x để không quá nhỏ.
        
        # log_input = |x| + epsilon
        log_input = torch.abs(x) + self.epsilon
        log_x = torch.log(log_input)
        
        # Tính toán trong log space
        m_log = self.nac_mul(log_x)
        
        # Cải tiến quan trọng 2: Output Clipping (Safe Exp)
        # Kẹp giá trị m_log trong khoảng [-omega, omega] trước khi exp
        # Điều này ngăn chặn việc exp ra vô cực (Infinity) gây nổ loss.
        m_log_clamped = torch.clamp(m_log, -self.omega, self.omega)
        m = torch.exp(m_log_clamped)
        
        # Một số biến thể iNALU xử lý dấu (sign) riêng ở đây, 
        # nhưng trong benchmark cơ bản thường dùng magnitude như NALU gốc 
        # trừ khi cài đặt đầy đủ cơ chế recover sign.
        # Ở đây ta giữ nguyên logic magnitude để so sánh công bằng với NALU.

        # --- Path 3: Gate ---
        g = torch.sigmoid(F.linear(x, self.G))

        # Combine
        y = g * a + (1 - g) * m
        return y

    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, omega={self.omega}'