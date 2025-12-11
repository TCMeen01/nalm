import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from models.nalu import NALU
from models.inalu import iNALU
from models.nac import *
from models.gnalu import GNALU
from models.nau import NAU
from models.nmu import NMU
from models.npu import NPU
from models.realnpu import RealNPU
from models.inpu import iNPU

from training.utils import *

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục training/
PROJECT_DIR = os.path.dirname(ROOT_DIR)                # thư mục nalm/

# =============================================================================
# Hàm Benchmark Core
# =============================================================================

def get_model_class(model_name):
    '''Mapping tên model từ string sang class'''
    # Việc của Bình: Hiện tại NPU và RealNPU chạy chưa ổn (chưa tìm hiểu vì sao, có thể
    # là initialization, hoặc setup, kiểm tra và tìm hiểu, sửa đổi)
    model_name = model_name.upper()
    mapping = {
        'NALU': NALU,
        'INALU': iNALU,
        'NAC_ADD': NAC_add,
        'NAC_MUL': NAC_mul,
        'GNALU': GNALU,
        'NAU': NAU,             #Done
        'NMU': NMU,             #Done
        'NPU': NPU,
        'REALNPU': RealNPU,
        'INPU': iNPU,           #Done
    }
    if model_name not in mapping:
        raise ValueError(f"Model {model_name} chưa được định nghĩa trong mapping.")
    return mapping[model_name]

def run_benchmark(args):
    results = []
    device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...")

    # Setup param
    modelClass = get_model_class(args.model)
    op_hash = {
        'add': (2, 6),
        'sub': (3, 7),
        'mul': (4, 8),
        'div': (5, 9)
    }
    if args.rangeID is None:
        rangeIDs = [f'U{i}' for i in range(1, 10)]
    else:
        rangeIDs = [rid.strip() for rid in list(args.rangeID.split(','))]

    # Hyperparameter
    lambda_base, lambda_start, lambda_end = 0, 20000, 35000
    beta_start, beta_end, beta_growth, beta_step = 0, 0, 10, 10000
    lr = 1e-3
    batch_size = args.batch_size
    n_iterations = args.n_iterations
    log_interval = args.log_interval
    verbose = args.verbose
    n_seeds = args.n_seeds

    if args.model.upper() == 'NAU': 
        lambda_base = 0.01
    elif args.model.upper() == 'NMU': 
        lambda_base = 10
    elif args.model.upper() in ['NPU', 'REALNPU']:
        lr = 5e-3
        if args.op == 'mul':
            beta_start, beta_end = 1e-7, 1e-5
        elif args.op == 'div':
            beta_start, beta_end = 1e-9, 1e-7
    
    # Training
    for rid in tqdm(rangeIDs, desc='Ranges'):
        val_path  = os.path.join(PROJECT_DIR, f'handle_data/data/range_{rid}_val.npz')
        test_path = os.path.join(PROJECT_DIR, f'handle_data/data/range_{rid}_test.npz')

        data_val  = torch.tensor(np.load(val_path)['data'], device=device)
        data_test = torch.tensor(np.load(test_path)['data'], device=device)

        X_val, Y_val = data_val[:, :2], data_val[:, op_hash[args.op][0]].unsqueeze(1)
        X_test, Y_test = data_test[:, :2], data_test[:, op_hash[args.op][0]].unsqueeze(1)

        mse_val = F.mse_loss(data_val[:, op_hash[args.op][0]], data_val[:, op_hash[args.op][1]])
        mse_test = F.mse_loss(data_test[:, op_hash[args.op][0]], data_test[:, op_hash[args.op][1]])

        print()
        print(f'MSE_val: {mse_val} | MSE_test: {mse_test}')

        solved_at_iters = []
        success = 0
        sparsity_list = []

        # Training
        for _ in tqdm(range(n_seeds), desc='Seeds'):
            print('\n\n')
            # Reinit model
            model = modelClass(in_dim=2, out_dim=1, device=device)

            # Generate new training data
            X_train, Y_train = rand_data_train(n_iterations=n_iterations, batch_size=batch_size,
                                               op=args.op, rid=rid)
            X_train = X_train.to(device=device)
            Y_train = Y_train.to(device=device)

            # Fit model
            history = {
                'interpolation_loss': [],
                'extrapolation_loss': [],
                'sparsity_loss'     : []
            }
            if args.optimizer == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif args.optimizer == 'SGD': optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else: raise ValueError(f'Unknown Optimization Algorithm: {args.optimizer}\n')

            for iter in range(1, n_iterations + 1):
                X = X_train[iter*batch_size : (iter + 1)*batch_size]
                Y = Y_train[iter*batch_size : (iter + 1)*batch_size]

                model.train()
                lambda_current = lambda_base * min(1.0, max(0.0, (iter - lambda_start) / (lambda_end - lambda_start)))
                beta_current   = min(beta_start * (beta_growth ** (iter // beta_step)), beta_end)
                
                if args.model.upper() != 'INALU':
                    total_loss = (F.mse_loss(model(X), Y) + 
                                lambda_current * model.regularization_loss() + 
                                beta_current * model.regularization_loss())
                elif args.model.upper() == 'INALU':
                    # Việc của Bình. Lưu ý ở đoạn Reinit và Khi nào kích hoạt regularization
                    pass


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if iter % log_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        history['interpolation_loss'].append(F.mse_loss(model(X_val), Y_val))
                        history['extrapolation_loss'].append(F.mse_loss(model(X_test), Y_test))
                        history['sparsity_loss'].append(model.sparsity_loss())

                        if verbose is True:
                            print(
                                f"Interation: {iter} | Train Loss: {total_loss} | "
                                f"Validation Loss: {history['interpolation_loss'][-1]} | "
                                f"Extrapolation Loss: {history['extrapolation_loss'][-1]}"
                            )
                            # print(model.W)

                        # Early stopping nếu 5000 iterations gần đây không thay đổi nhiều (interpolation)
                        if len(history['interpolation_loss']) > 5000 // log_interval:
                            recent_losses = history['interpolation_loss'][-(5000 // log_interval):]
                            if max(recent_losses) - min(recent_losses) < min(1e-6, mse_val.item() / 100):
                                if verbose is True:
                                    print(f"Early stopping at iteration {iter} due to convergence.")
                                break

                
            # Post-process
            solved_at_iter, best_model, sparsity_error = extract_metrics(history, threshold_inter=mse_val.item(),
                                                                         threshold_extra=mse_test.item(),
                                                                         log_interval=log_interval)
            if solved_at_iter is not None:
                print(solved_at_iter, best_model, sparsity_error)
                solved_at_iters.append(solved_at_iter)
                success += 1
                sparsity_list.append(sparsity_error)

        # Tính toán kết quả và lưu lại
        sr, sr_plus, sr_minus = ci_success_rate(success, n_seeds)
        sc, sc_plus, sc_minus = ci_speed_convergence(solved_at_iters)
        se, se_plus, se_minus = ci_sparsity_error(sparsity_list)

        results.append({
            'Model': args.model,
            'Operation': args.op,
            'Range': rid,
            'Success_Rate': sr,
            'Success_Rate_Plus': sr_plus,
            'Success_Rate_Minus': sr_minus,
            'Speed_Convergence_Mean': sc,
            'Speed_Convergence_Plus': sc_plus,
            'Speed_Convergence_Minus': sc_minus,
            'Sparsity_Error_Mean': se,
            'Sparsity_Error_Plus': se_plus,
            'Sparsity_Error_Minus': se_minus
        })

    df_results = pd.DataFrame(results)
    return df_results