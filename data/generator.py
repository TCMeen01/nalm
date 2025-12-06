import torch
import numpy as np

# Cấu hình dải dữ liệu theo Table 3 trong paper "A Primer for Neural Arithmetic Logic Modules"
# Format: 'Key': (Interpolation_Range, Extrapolation_Range)
# Nếu Range là list lồng nhau (ví dụ U5 extrapolation), nghĩa là hợp của các khoảng rời rạc.
RANGE_CONFIGS = {
    'U1': ([-20, -10],      [-40, -20]),
    'U2': ([-2, -1],        [-6, -2]),
    'U3': ([-1.2, -1.1],    [-6.1, -1.2]),
    'U4': ([-0.2, -0.1],    [-2, -0.2]),
    'U5': ([-2, 2],         [[-6, -2], [2, 6]]), # U5: Extrapolation là 2 khoảng rời nhau
    'U6': ([0.1, 0.2],      [0.2, 2]),
    'U7': ([1, 2],          [2, 6]),
    'U8': ([1.1, 1.2],      [1.2, 6]),
    'U9': ([10, 20],        [20, 40])
}

def generate_data(op, range_cfg, batch_size=128, input_dim=2, epsilon=1e-7):
    """
    Sinh dữ liệu cho bài toán Single Layer.
    
    Args:
        op (str): 'add', 'sub', 'mul', 'div'
        range_cfg (list): Khoảng dữ liệu, ví dụ [-10, 10] hoặc [[-6, -2], [2, 6]]
        batch_size (int): Số lượng mẫu cần sinh
        input_dim (int): Kích thước vector đầu vào (mặc định 2 cho bài toán cơ bản)
        epsilon (float): Số nhỏ để tránh chia cho 0
        
    Returns:
        X (Tensor): [batch_size, input_dim]
        y (Tensor): [batch_size, 1]
    """
    
    # 1. Sinh Input X
    # Kiểm tra xem range_cfg có phải là hợp của nhiều khoảng không (như U5 extrapolation)
    # Cấu trúc check: Nếu phần tử đầu tiên của range_cfg cũng là một list/tuple/tensor
    is_disjoint = isinstance(range_cfg[0], (list, tuple, np.ndarray))
    
    if is_disjoint:
        # Trường hợp khoảng rời (Ví dụ U5 test: [[-6, -2], [2, 6]])
        # Chia đều batch_size cho các khoảng con
        n_sub_ranges = len(range_cfg)
        samples_per_range = batch_size // n_sub_ranges
        
        x_parts = []
        for sub_range in range_cfg:
            # Sinh dữ liệu cho từng khoảng con
            low, high = sub_range
            part = torch.FloatTensor(samples_per_range, input_dim).uniform_(low, high)
            x_parts.append(part)
            
        # Nếu chia không hết, bù mẫu vào khoảng cuối
        remaining = batch_size - (samples_per_range * n_sub_ranges)
        if remaining > 0:
            low, high = range_cfg[-1]
            last_part = torch.FloatTensor(remaining, input_dim).uniform_(low, high)
            x_parts.append(last_part)
            
        X = torch.cat(x_parts, dim=0)
        
        # Shuffle để trộn các khoảng lại với nhau
        indices = torch.randperm(X.size(0))
        X = X[indices]
        
    else:
        # Trường hợp khoảng liền (Ví dụ U1: [-20, -10])
        low, high = range_cfg
        X = torch.FloatTensor(batch_size, input_dim).uniform_(low, high)

    # 2. Sinh Target y (Labels)
    # Lấy 2 cột đầu tiên để tính toán (theo đúng logic benchmark chuẩn)
    a, b = X[:, 0:1], X[:, 1:2]
    
    if op == 'add':
        y = a + b
    elif op == 'sub':
        y = a - b
    elif op == 'mul':
        y = a * b
    elif op == 'div':
        # Xử lý chia cho 0 hoặc số quá nhỏ
        # Kỹ thuật: giữ nguyên dấu của b, nhưng độ lớn tối thiểu là epsilon
        # sign(b) * max(|b|, eps)
        safe_b = torch.sign(b) * torch.maximum(torch.abs(b), torch.tensor(epsilon))
        # Nếu b = 0, sign(0) = 0 -> vẫn lỗi. Fix triệt để:
        safe_b[torch.abs(safe_b) < epsilon] = epsilon 
        y = a / safe_b
    else:
        raise ValueError(f"Operation '{op}' not supported. Choose add, sub, mul, div.")

    return X, y

# --- Helper function để test nhanh file này ---
if __name__ == "__main__":
    print("Testing Generation U5 Extrapolation:")
    # Lấy config U5 phần Extrapolation (index 1)
    cfg = RANGE_CONFIGS['U5'][1] 
    
    X, y = generate_data('add', cfg, batch_size=10)
    
    print("Range Config:", cfg)
    print("X sample:\n", X)
    print("y sample (add):\n", y)
    
    # Kiểm tra xem X có nằm trong khoảng cấm [-2, 2] không
    in_forbidden = ((X > -2) & (X < 2)).any()
    print("Contains data in forbidden gap [-2, 2]?", in_forbidden.item())