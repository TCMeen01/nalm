import torch
import numpy as np
from scipy.stats import binomtest, bayes_mvs
import math

# =================================================
# Generate Data cho từng phép toán và range config
# =================================================

RANGE_CONFIGS = {
    'U1': ([-20, -10], [[-40, -20]]),
    'U2': ([-2, -1], [[-6, -2]]),
    'U3': ([-1.2, -1.1], [[-6.1, -1.2]]),
    'U4': ([-0.2, -0.1], [[-2, -0.2]]),
    'U5': ([-2, 2], [[-6, -2], [2, 6]]), # TH đặc biệt bị rời
    'U6': ([0.1, 0.2], [[0.2, 2]]),
    'U7': ([1, 2], [[2, 6]]),
    'U8': ([1.1, 1.2], [[1.2, 6]]),
    'U9': ([10, 20], [[20, 40]])
}

def apply_op(op, x1, x2, epsilon):
    if op == 'add':
        return x1 + x2 - epsilon * (abs(x1) + abs(x2))
    elif op == 'sub':
        return x1 - x2 - epsilon * (abs(x1) + abs(x2))
    elif op == 'mul':
        return (x1 * x2) * (1 - epsilon)**2
    elif op == 'div':
        assert torch.all(x2 != 0)
        return x1 / x2 * (1 - epsilon) / (1 + epsilon)
    else:
        raise ValueError(f"Unknown op: {op}")
    
def generate_data(op, range_cfg_id, type='inter', n_samples=10000, seed=0, epsilon=0):
    """
    Truyền vào range_cfg là [key]: 'U1', 'U2',...
    Truyền vào op là mảng gồm add, sub, mul, div.
    Truyền vào type là inter (interpolation) hoặc extra (extrapolation)

    Trả về X, Y
    """
    # Đặt seed để tái lập kết quả
    torch.manual_seed(seed)
    
    if type == 'inter':
        train_range = RANGE_CONFIGS[range_cfg_id][0]
        low, high = train_range
        X = torch.zeros(n_samples, 2).uniform_(low, high)
    elif type == 'extra':
        test_range = RANGE_CONFIGS[range_cfg_id][1]
        num_ranges = len(test_range)
        per_range = n_samples // num_ranges
        
        parts = []
        current_count = 0

        for i, (low, high) in enumerate(test_range):
            count = per_range if i < num_ranges - 1 else n_samples - current_count
            
            part = torch.zeros(count, 2).uniform_(low, high)
            parts.append(part)
            current_count += count
            
        X = torch.cat(parts, dim=0)
    else:
        raise ValueError(f'type {type} is not defined')
    
    if isinstance(op, (list, tuple)):
        temp = []
        for operator in op:
            temp.append(apply_op(operator, X[:, 0], X[:, 1], epsilon).unsqueeze(1))
        Y = torch.cat(temp, dim=1)
    else:
        Y = apply_op(op, X[:, 0], X[:, 1], epsilon).unsqueeze(1)

    return X, Y

def rand_data_train(n_iterations, batch_size, op, rid, dim=2):
    low, high = RANGE_CONFIGS[rid][0]
    data = []
    for _ in range(n_iterations):
        torch.manual_seed(np.random.randint(0, (1<<31) - 1))
        data.append(torch.zeros(batch_size, dim).uniform_(low, high))
    
    X = torch.cat(data, dim=0)
    Y = apply_op(op, X[:, 0], X[:, 1], epsilon=0).unsqueeze(1)

    return X, Y

# =================================================
# helper cho tác vụ huấn luyện và đánh giá
# =================================================

def calc_sparsity_loss(W):
    '''Tính sparsity loss cho tất cả tham số trong model'''
    W_abs = torch.abs(W)
    return torch.max(torch.min(W_abs, torch.abs(1 - W_abs)))


def extract_metrics(history, threshold_inter, threshold_extra, log_interval=1000, n_last_steps=5000):
    '''
    Truyền vào history: Chứa data khi training trên 1 seed (inter/extra loss, sparsity_loss)

    Trả về:
        first_solved_step: Của Extrapolation data
        best_model: Model tốt nhất (cho interpolation data) trong n_last_steps bước cuối
        sparsity_error: Của best_model
    '''
    first_solved_step = None
    best_model = None
    sparsity_error = None

    # Tìm first_solved_step
    for i, extra_loss in enumerate(history['extrapolation_loss']):
        if extra_loss.item() < threshold_extra:
            first_solved_step = (i + 1) * log_interval
            break
    
    # Tìm best_model và sparsity_error
    min_step = max(0, len(history['interpolation_loss']) - (n_last_steps // log_interval))
    best_model = np.argmin(history['interpolation_loss'][min_step:]) + min_step
    sparsity_error = history['sparsity_loss'][best_model]
    
    return first_solved_step, (best_model + 1) * log_interval, sparsity_error

# 1) SUCCESS RATE - Binom
def ci_success_rate(success, n_seeds, confidence=0.95):
    """
    Tính CI cho Success Rate dùng phân phối nhị thức (95%)
    Trả về: (mean, plus, minus)
    """
    mean = success / n_seeds
    low, high = binomtest(success, n_seeds).proportion_ci(confidence_level=confidence)

    return mean * 100, (high - mean) * 100, (mean - low) * 100


# 2) SPEED CONVERGENCE - Gamma
def ci_speed_convergence(solved_at_iters, alpha=0.05):
    """
    Tính CI cho Speed Convergence (Mean) dùng scipy.stats.bayes_mvs.
    Phương pháp này tự động tính Mean, Variance, Std cho mẫu nhỏ.
    Trả về: (mean, plus, minus)
    """
    vals = solved_at_iters
    n = len(vals)
    
    if n == 0:
        return None, None, None
    
    if np.allclose(solved_at_iters, solved_at_iters[0], atol=1e-8):
        return np.mean(solved_at_iters), 0.0, 0.0

    if n == 1:
        return vals[0], 0.0, 0.0

    mean_cntr, (lower, upper) = bayes_mvs(vals, alpha=1-alpha)[0]
    
    mean  = mean_cntr
    plus  = max(0.0, upper - mean)
    minus = max(0.0, mean - lower)
    
    return mean, plus, minus


# 3) SPARSITY ERROR
def ci_sparsity_error(sparsity_list, alpha=0.05):
    """
    sparsity_list: list sparsity_error mỗi seed (>=0)
    return: (mean, plus, minus)
    """
    """
    Tính Mean và CI 95% cho Sparsity Error.
    - Input: List các giá trị sparsity (float).
    - Output: (mean, plus, minus).
    """
    n = len(sparsity_list)
    
    if n == 0:
        return None, None, None

    if np.allclose(sparsity_list, sparsity_list[0], atol=1e-8):
        return np.mean(sparsity_list), 0.0, 0.0

    if n == 1:
        return sparsity_list[0], 0.0, 0.0

    mean_cntr, (lower, upper) = bayes_mvs(sparsity_list, alpha=1-alpha)[0]

    mean  = mean_cntr
    plus  = max(0.0, upper - mean)
    minus = max(0.0, mean - lower)
    
    return mean, plus, minus