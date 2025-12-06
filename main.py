import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

# Import Data & Modules
from data.generator import generate_data, RANGE_CONFIGS
from training.regularization import RegScheduler
from models.nac import NAC
from models.nalu import NALU
from models.nau import NAU
from models.nmu import NMU
from models.inalu import iNALU
from models.gnalu import GNALU
from models.npu import NPU
from models.realnpu import RealNPU

# --- CONFIG CHUNG ---
ITERATIONS = 50000
BATCH_SIZE = 128
VAL_TEST_SIZE = 10000
DEFAULT_LR = 1e-3 # NAC, NALU, NAU...

class EarlyStopping:
    """Dừng sớm nếu Validation loss không giảm sau 'patience' lần kiểm tra."""
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_model_instance(model_name, in_dim=2, out_dim=1):
    name = model_name.lower()
    if name == 'nac': return NAC(in_dim, out_dim)
    if name == 'nalu': return NALU(in_dim, out_dim)
    if name == 'nau': return NAU(in_dim, out_dim)
    if name == 'nmu': return NMU(in_dim, out_dim)
    if name == 'inalu': return iNALU(in_dim, out_dim)
    if name == 'gnalu': return GNALU(in_dim, out_dim)
    if name == 'npu': return NPU(in_dim, out_dim)
    if name == 'realnpu': return RealNPU(in_dim, out_dim)
    raise ValueError(f"Model {name} not supported")

def train_job(args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Data Setup
    train_range_cfg = RANGE_CONFIGS[args.range][0]
    test_range_cfg  = RANGE_CONFIGS[args.range][1]
    
    X_val, y_val = generate_data(args.op, train_range_cfg, VAL_TEST_SIZE)
    X_test, y_test = generate_data(args.op, test_range_cfg, VAL_TEST_SIZE)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # 2. Model Setup
    model = get_model_instance(args.model).to(device)
    
    # --- TABLE 6: LR riêng cho NPU/RealNPU ---
    current_lr = 5e-3 if args.model in ['npu', 'realnpu'] else DEFAULT_LR
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    
    criterion = nn.MSELoss()
    
    # --- SCHEDULER & EARLY STOPPING ---
    reg_scheduler = RegScheduler(args.model, args.op, ITERATIONS)
    # Check validation mỗi 1000 bước, patience = 20 checks => chịu đựng được 20k bước ko giảm
    early_stopping = EarlyStopping(patience=20) 
    
    # 3. Training Loop
    pbar = tqdm(range(ITERATIONS), desc=f"Seed {seed:02d}", leave=False)
    final_test_mse = float('inf')
    
    for step in pbar:
        # a. Data
        X_train, y_train = generate_data(args.op, train_range_cfg, BATCH_SIZE)
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # b. Forward
        pred = model(X_train)
        mse_loss = criterion(pred, y_train)
        
        # c. Regularization (Dynamic Weight Update)
        reg_weight = reg_scheduler.step_update()
        reg_loss = reg_scheduler.get_loss(model)
        
        total_loss = mse_loss + reg_loss
        
        # d. Backward
        optimizer.zero_grad()
        if torch.isnan(total_loss):
            pbar.write(f"NaN Loss at step {step}")
            break
        total_loss.backward()
        
        # --- TABLE 8: Gradient Clipping riêng cho iNALU ---
        if args.model == 'inalu':
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1) # Clip range [-0.1, 0.1]
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Default
            
        optimizer.step()
        
        # e. Validation & Early Stopping (Check mỗi 1000 bước)
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_mse = criterion(val_pred, y_val).item()
            model.train()
            
            pbar.set_postfix({'mse': f"{mse_loss.item():.2e}", 'reg_w': f"{reg_weight:.1e}"})
            
            # Check Early Stopping
            early_stopping(val_mse)
            if early_stopping.early_stop:
                pbar.write(f"Early stopping at step {step}")
                break

    # 4. Final Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        final_test_mse = criterion(test_pred, y_test).item()
        
        # Validation cuối cùng
        val_pred = model(X_val)
        final_val_mse = criterion(val_pred, y_val).item()

    return {
        'seed': seed,
        'val_mse': final_val_mse,
        'test_mse': final_test_mse,
        'success': final_test_mse < 1e-5
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['nac', 'nalu', 'nau', 'nmu', 'inalu', 'gnalu', 'npu', 'realnpu'])
    parser.add_argument('--op', type=str, required=True, choices=['add', 'sub', 'mul', 'div'])
    parser.add_argument('--range', type=str, default='U1', choices=RANGE_CONFIGS.keys())
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Running: {args.model.upper()} | {args.op} | {args.range}")
    
    results = []
    for seed in range(25):
        res = train_job(args, seed, device)
        status = "SUCCESS" if res['success'] else "FAIL"
        print(f"Seed {seed:02d}: Test MSE={res['test_mse']:.2e} [{status}]")
        results.append(res)
        
    df = pd.DataFrame(results)
    success_rate = df['success'].mean() * 100
    print(f"Final Success Rate: {success_rate:.1f}%")
    
    os.makedirs('results', exist_ok=True)
    df.to_csv(f"results/{args.model}_{args.op}_{args.range}.csv", index=False)