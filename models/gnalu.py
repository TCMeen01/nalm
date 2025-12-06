import torch
import torch.nn as nn
from models.nalu import NALU

class GNALU(NALU):
    """
    Golden Ratio NALU (GNALU)
    Paper: The Golden Ratio in the Initialization of Neural Arithmetic Logic Units
    
    Logic:
    Hoạt động y hệt NALU, nhưng thay đổi cách khởi tạo trọng số (Initialization).
    Sử dụng phân phối dựa trên tỷ lệ vàng (Golden Ratio) để tránh Dead Units.
    """
    def __init__(self, in_dim, out_dim, epsilon=1e-7):
        # Gọi init của NALU, nó sẽ tự gọi reset_parameters của class con (GNALU)
        super().__init__(in_dim, out_dim, epsilon)

    def reset_parameters(self):
        # Gọi init đặc biệt: Golden Ratio Initialization
        
        # Định nghĩa các hằng số
        phi = 1.61803398875 # Tỷ lệ vàng
        
        # Tính toán độ lệch chuẩn (std) dựa trên phi
        # Công thức: std = 1 / phi
        # Một số paper đề xuất: std = sqrt(2 / (in + out)) * phi (như Kaiming * phi)
        # Nhưng trong benchmark repo, họ thường dùng cách đơn giản:
        std = 1.0 / phi 

        # 1. Init Gate (G)
        # Gate cần hoạt động ở vùng tuyến tính của Sigmoid (gần 0) nhưng có variance cụ thể
        nn.init.normal_(self.G, mean=0.0, std=std)
        
        # 2. Init Sub-modules (NACs)
        # Chúng ta cần truy cập vào params của nac_add và nac_log
        self._init_nac_golden(self.nac_add, std)
        self._init_nac_golden(self.nac_log, std)

    def _init_nac_golden(self, nac_module, std):
        """Hàm helper để init NAC theo chuẩn Golden"""
        # W_hat và M_hat trong NAC
        nn.init.normal_(nac_module.W_hat, mean=0.0, std=std)
        nn.init.normal_(nac_module.M_hat, mean=0.0, std=std)