import torch
import numpy as np

# --- 1. CÁC HÀM TÍNH LOSS ---

def nau_sparsity_loss(model):
    """NAU: Ép trọng số về {-1, 0, 1}"""
    loss = 0
    count = 0
    for name, param in model.named_parameters():
        if 'W' in name:
            w = param
            dist_1 = (w - 1).pow(2)
            dist_0 = w.pow(2)
            dist_m1 = (w + 1).pow(2)
            loss += torch.min(dist_0, torch.min(dist_1, dist_m1)).sum()
            count += w.numel()
    return loss / (count + 1e-8)

def nmu_sparsity_loss(model):
    """NMU: Ép trọng số về {0, 1}"""
    loss = 0
    count = 0
    for name, param in model.named_parameters():
        if 'W' in name:
            w = param
            dist_0 = w.pow(2)
            dist_1 = (w - 1).pow(2)
            loss += torch.min(dist_0, dist_1).sum()
            count += w.numel()
    return loss / (count + 1e-8)

def npu_regularization_loss(model):
    """
    NPU/RealNPU: Ép trọng số về số nguyên (Integer).
    L = min(|W - round(W)|)
    Mặc dù paper gốc dùng công thức phức tạp hơn, nhưng trong benchmark 
    thường dùng penalty đơn giản là khoảng cách tới số nguyên gần nhất.
    """
    loss = 0
    count = 0
    for name, param in model.named_parameters():
        if 'W' in name:
            w = param
            # Khoảng cách tới số nguyên gần nhất: |w - round(w)|
            # Dùng thủ thuật sin: sin(pi * w)^2 = 0 tại các số nguyên
            # Hoặc đơn giản là (w - w.round()).pow(2)
            loss += (torch.sin(torch.pi * w) ** 2).sum()
            count += w.numel()
    return loss / (count + 1e-8)

def inalu_regularization_loss(model):
    """iNALU: Thường ép gate về 0 hoặc 1, hoặc sparse weights"""
    # Theo Table 8, iNALU có regularisation, thường là L1 hoặc Sparsity trên Gate
    # Ta dùng logic tương tự NMU cho Gate hoặc Weight
    return 0 # Placeholder nếu paper không ghi rõ công thức loss cụ thể cho iNALU

# --- 2. CLASS QUẢN LÝ LỊCH TRÌNH (SCHEDULER) ---

class RegScheduler:
    """
    Quản lý việc thay đổi hệ số Regularization theo số bước (Iterations)
    Dựa trên Table 6 (NPU) và Table 7 (NAU/NMU)
    """
    def __init__(self, model_type, op, max_steps):
        self.model_type = model_type.lower()
        self.op = op
        self.step = 0
        
        # Cấu hình mặc định
        self.weight = 0.0
        
        # --- Cấu hình Table 7 (NAU/NMU) ---
        if self.model_type == 'nau':
            self.target_lambda = 0.01
            self.start = 20000
            self.end = 35000
        elif self.model_type == 'nmu':
            self.target_lambda = 10.0 # Table 7
            self.start = 20000
            self.end = 35000
            
        # --- Cấu hình Table 6 (NPU/RealNPU) ---
        elif self.model_type in ['npu', 'realnpu']:
            # Table 6: Mul vs Div ranges
            if self.op == 'mul':
                self.beta_start = 1e-7
                self.beta_end = 1e-5
            elif self.op == 'div':
                self.beta_start = 1e-9
                self.beta_end = 1e-7
            else: # Add/Sub fallback
                self.beta_start = 1e-9
                self.beta_end = 1e-7
                
            self.beta_growth = 10     # Tăng gấp 10 lần
            self.beta_step_size = 10000 # Mỗi 10k bước
            self.weight = self.beta_start # Giá trị khởi đầu

        # --- Cấu hình Table 8 (iNALU) ---
        elif self.model_type == 'inalu':
            self.start = 10000 # Minimum epochs before regularisation starts
            self.target_lambda = 1.0 # Mặc định nếu không ghi rõ
        else:
            self.weight = 0.0

    def step_update(self):
        self.step += 1
        
        # Logic cho NAU / NMU / iNALU (Linear Ramp-up hoặc Threshold)
        if self.model_type in ['nau', 'nmu']:
            if self.step < self.start:
                self.weight = 0.0
            elif self.step >= self.end:
                self.weight = self.target_lambda
            else:
                # Nội suy tuyến tính từ 0 đến target
                progress = (self.step - self.start) / (self.end - self.start)
                self.weight = progress * self.target_lambda
                
        elif self.model_type == 'inalu':
            if self.step < self.start:
                self.weight = 0.0
            else:
                self.weight = self.target_lambda

        # Logic cho NPU / RealNPU (Step Growth)
        elif self.model_type in ['npu', 'realnpu']:
            # Tăng beta sau mỗi 10000 bước
            num_growths = self.step // 10000
            current_beta = self.beta_start * (10 ** num_growths)
            # Kẹp giá trị max
            self.weight = min(current_beta, self.beta_end)
            
        return self.weight

    def get_loss(self, model):
        if self.weight == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        if self.model_type == 'nau':
            return self.weight * nau_sparsity_loss(model)
        elif self.model_type == 'nmu':
            return self.weight * nmu_sparsity_loss(model)
        elif self.model_type in ['npu', 'realnpu']:
            return self.weight * npu_regularization_loss(model)
        # iNALU có thể thêm loss nếu cần
        return torch.tensor(0.0, device=next(model.parameters()).device)