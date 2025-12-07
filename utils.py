import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.nalu import NALU
from models.inalu import iNALU
from models.nac import NAC
from models.gnalu import GNALU
from models.nau import NAU
from models.nmu import NMU
from models.npu import NPU
from models.realnpu import RealNPU
from models.inpu import iNPU

RANGE_CONFIGS = {
    'U1': ([-20, -10], [-40, -20]),
    'U2': ([-2, -1], [-6, -2]),
    'U3': ([-1.2, -1.1], [-6.1, -1.2]),
    'U4': ([-0.2, -0.1], [-2, -0.2]),
    'U5': ([-2, 2], [[-6, -2], [2, 6]]), # TH đặc biệt bị rời
    'U6': ([0.1, 0.2], [0.2, 2]),
    'U7': ([1, 2], [2, 6]),
    'U8': ([1.1, 1.2], [1.2, 6]),
    'U9': ([10, 20], [20, 40])
}

# =================================================
# Generate Data cho từng phép toán và range config
# =================================================

def apply_op(op, x1, x2):
    if op == 'add':
        return x1 + x2
    elif op == 'sub':
        return x1 - x2
    elif op == 'mul':
        return x1 * x2
    elif op == 'div':
        assert torch.all(x2 != 0)
        return x1 / x2
    else:
        raise ValueError(f"Unknown op: {op}")
    
def generate_data(op, range_cfg_id, n_train=10000, n_test=10000, seed=0):
    """
    Truyền vào range_cfg là [key]: 'U1', 'U2',...
    Truyền vào op là add, sub, mul, div.

    Trả về X_train, y_train, X_test, y_test (Ma trận Nx2 và Nx1)
    """
    # Đặt seed để tái lập kết quả
    torch.manual_seed(seed)
    
    train_range, test_range = RANGE_CONFIGS[range_cfg_id]

    # --- 1. Training Dataset ---
    low, high = train_range
    X_train = torch.zeros(n_train, 2).uniform_(low, high)
    
    # Tính y_train
    y_train = apply_op(op, X_train[:, 0], X_train[:, 1]).unsqueeze(1) 

    # --- 2. Test Dataset ---
    if isinstance(test_range[0], (list, tuple)):
        num_ranges = len(test_range)
        per_range = n_test // num_ranges
        
        parts = []
        current_count = 0
        
        for i, (low, high) in enumerate(test_range):
            # Xử lý phần dư cho range cuối cùng để đảm bảo đủ n_test mẫu
            count = per_range if i < num_ranges - 1 else n_test - current_count
            
            part = torch.zeros(count, 2).uniform_(low, high)
            parts.append(part)
            current_count += count
            
        X_test = torch.cat(parts, dim=0)
        
    else:
        low, high = test_range
        X_test = torch.zeros(n_test, 2).uniform_(low, high)

    y_test = apply_op(op, X_test[:, 0], X_test[:, 1]).unsqueeze(1)

    return X_train, y_train, X_test, y_test


# =================================================
# helper cho tác vụ huấn luyện và đánh giá
# =================================================

def sparsity_loss(model, device):
    '''Tính sparsity loss cho tất cả tham số trong model'''
    loss = torch.tensor(0.0, device=device) 
    for param in model.parameters():
        param_abs = torch.abs(param)
        loss = torch.max(loss, torch.max(torch.minimum(param_abs, 1 - param_abs)))

    return loss

def mse_threshold_for_op(op, range_cfg_id, device, epsilon=1e-5, n_samples=1e6, seed=42):
    """Tính MSE threshold dựa trên epsilon-perfect model trên tập TEST range"""
    torch.manual_seed(seed)
    _, _, X_test, y_test = generate_data(op, range_cfg_id, n_train=0, n_test=int(n_samples), seed=seed)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    x1 = X_test[:, 0]
    x2 = X_test[:, 1]
    
    if op == 'add':
        y_eps = (x1 + x2) - (torch.abs(x1) + torch.abs(x2)) * epsilon
    elif op == 'sub':
        y_eps = (x1 - x2) - (torch.abs(x1) + torch.abs(x2)) * epsilon
    elif op == 'mul':
        y_eps = (x1 * x2) * (1.0 - epsilon) ** 2
    elif op == 'div':
        y_eps = (x1 / x2) * ((1.0 - epsilon) / (1.0 + epsilon))
    else:
        raise ValueError(f"Unknown op: {op}")
    
    y_eps = y_eps.unsqueeze(1)

    return 0.5 * torch.mean((y_test - y_eps) ** 2).item()


# =============================================================================
# Hàm Benchmark Core
# =============================================================================

def get_model_class(model_name):
    '''Mapping tên model từ string sang class'''
    model_name = model_name.upper()
    mapping = {
        'NALU': NALU,
        'INALU': iNALU,
        'NAC': NAC,
        'GNALU': GNALU,
        'NAU': NAU,
        'NMU': NMU,
        'NPU': NPU,
        'REALNPU': RealNPU,
        'INPU': iNPU,
    }
    if model_name not in mapping:
        raise ValueError(f"Model {model_name} chưa được định nghĩa trong mapping.")
    return mapping[model_name]

def fit_model(modelName, model, op, X_train, Y_train, X_test, Y_test,
              threshold, lr=None, epochs=50000, optimizer_algo='Adam'):
    '''Hàm fit chung cho tất cả model'''
    if modelName in ['NALU', 'iNALU', 'NAC', 'gNALU']:
        return model.fit(
            X_train, Y_train, X_test, Y_test,
            lr=lr if lr is not None else 1e-3, epochs=epochs, each_epoch=None,
            optimizer_algo=optimizer_algo, threshold=threshold,
        )
    elif modelName == 'NAU':
        return model.fit(
            X_train, Y_train, X_test, Y_test,
            lr=lr if lr is not None else 1e-3, epochs=epochs, each_epoch=None,
            optimizer_algo=optimizer_algo, threshold=threshold,
            lambda_base=0.01, lambda_start=20000, lambda_end=35000
        )
    elif modelName == 'NMU':
        return model.fit(
            X_train, Y_train, X_test, Y_test,
            lr=lr if lr is not None else 1e-3, epochs=epochs, each_epoch=None,
            optimizer_algo=optimizer_algo, threshold=threshold,
            lambda_base=0.01, lambda_start=20000, lambda_end=35000
        )
    elif modelName in ['NPU', 'RealNPU']:
        return model.fit(
            X_train, Y_train, X_test, Y_test,
            lr=lr if lr is not None else 5e-3, epochs=epochs, each_epoch=None,
            optimizer_algo=optimizer_algo, threshold=threshold,
            beta_begin=1e-7 if op == 'mul' else 1e-9, beta_end=1e-5 if op == 'mul' else 1e-7,
            beta_growth=10, beta_step=10000
        )
    elif modelName == 'iNPU':
        return model.fit(
            X_train, Y_train, X_test, Y_test,
            lr=lr if lr is not None else 5e-3, epochs=epochs, each_epoch=None,
            optimizer_algo=optimizer_algo, threshold=threshold,
        )

def run_benchmark(args):
    results = []
    device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...")

    # 1. Tính Thresholds cho toàn bộ ranges (tránh tính lại nhiều lần)
    thresholds = {}
    print("Pre-computing thresholds...")
    for rid in [f'U{i}' for i in range(1, 10)]:
        thresholds[rid] = mse_threshold_for_op(args.op, rid, device, epsilon=1e-5)

    ModelClass = get_model_class(args.model)

    # 2. Loop qua các Range
    for rid in tqdm([f'U{i}' for i in range(1, 10)], desc="Ranges"):
        print()
        mse_threshold = thresholds[rid]
        
        epochs_ran_list = []
        success_list = []
        sparsity_list = []

        # 3. Loop qua các Seed
        for seed in range(1, 26): # Seed 1-25
            # Fit Model with Initial Data
            model = ModelClass(in_dim=2, out_dim=1, device=device)

            X_train, y_train, X_test, y_test = generate_data(
                args.op, rid, n_train=10000, n_test=10000, seed=seed
            )

            converged_at, success = fit_model(args.model, model, args.op,
                X_train, y_train, X_test, y_test,
                threshold=mse_threshold,lr=args.lr,
                epochs=int(args.n_epochs), optimizer_algo=args.optimizer)
            
            print(f'DONE | Model: {args.model} | RangeID: {rid} | Seed: {seed} | Result: ({converged_at}, {success})')
            sparsity = sparsity_loss(model, device).item()

            epochs_ran_list.append(converged_at)
            success_list.append(1 if success else 0)
            sparsity_list.append(sparsity)

        # 4. Tổng hợp Metric cho Range này
        n_seeds = 25
        k_success = sum(success_list)
        success_rate = k_success / n_seeds * 100.0
        
        # Speed of convergence (chỉ tính trên seed success)
        success_epochs = [e for e, s in zip(epochs_ran_list, success_list) if s == 1]
        mean_speed = np.mean(success_epochs) if success_epochs else np.nan

        # Sparsity Mean
        mean_sparsity = np.mean(sparsity_list)

        # Lưu kết quả
        row = {
            # Metadata
            'Module': args.model,
            'Operation': args.op,
            'Optimizer': args.optimizer,
            'Batch_Size': args.batch_size,
            'Learning_Rate': args.lr,
            'Device': str(device),
            'Range': rid,
            
            # Metrics
            'Success_Rate': success_rate,
            'Speed_Convergence_Mean': mean_speed,
            'Sparsity_Error_Mean': mean_sparsity,
        }
        results.append(row)

    return pd.DataFrame(results)