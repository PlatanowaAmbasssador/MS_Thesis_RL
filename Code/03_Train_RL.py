# 03 — Train RL Agent v2 (LSTM + Attention + Dirichlet)

import os, time
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
else:
    print('Device: CPU')

from functions.data_pipeline import build_dataset
from functions.RL_1.train import train_walk_forward, generate_wfo_folds, count_wfo_folds, plot_wfo_folds

dataset = build_dataset('../Data/Outputs/Filtered/Data')

## 1. Preview & Visualize WFO Folds

wfo_info = count_wfo_folds(dataset['trading_dates'], train_months=24, val_months=1, test_months=1, step_months=1, embargo_days=5)
print(f'Total folds: {wfo_info["n_folds"]}')
if wfo_info['n_folds'] > 0:
    print(f'First test: {wfo_info["first_fold"]["test_start"]} → {wfo_info["first_fold"]["test_end"]}')
    print(f'Last test:  {wfo_info["last_fold"]["test_start"]} → {wfo_info["last_fold"]["test_end"]}')
    print(f'OOS: {wfo_info["total_test_period"][0]} → {wfo_info["total_test_period"][1]}')

folds = generate_wfo_folds(dataset['trading_dates'], train_months=24, val_months=1, test_months=1, step_months=1, embargo_days=5)

## 2. Train (Walk-Forward)
### 5 HP configs → sliding 24m window → checkpoint after each fold

t0 = time.time()

rl_results = train_walk_forward(
    dataset,
    train_months=24,
    val_months=1,
    test_months=1,
    step_months=1,
    embargo_days=5,
    n_epochs=30,
    patience=5,
    min_epochs=10,
    transaction_cost_bps=5.0,
    turnover_penalty=0.001,
    variance_penalty=0.5,
    tc_curriculum_frac=0.3,
    lookback_window=60,
    results_dir='../Results',
    verbose=True,
)

print(f'\n\nTotal time: {(time.time()-t0)/60:.1f} minutes')

## 3. Results

print('=' * 60)
print('OUT-OF-SAMPLE PERFORMANCE')
print('=' * 60)
rl_m = pd.read_csv('../Results/rl_performance_metrics.csv', index_col=0)
print(rl_m.to_string())

fold_log = pd.read_csv('../Results/rl_fold_log.csv')
print(f'Retrains: {fold_log["retrained"].sum()} / {len(fold_log)} folds')
print(fold_log.to_string())

bl_metrics = pd.read_csv('../Results/performance_metrics_full.csv', index_col=0)
rl_row = rl_m.loc[['RL Agent']]
combined = pd.concat([bl_metrics, rl_row]).sort_values('IR2', ascending=False)
print(combined[['ARC (%)', 'ASD (%)', 'Max Drawdown (%)', 'IR1', 'IR2']].to_string())

## 4. Files Saved

print('RL files in ../Results/:')
for f in sorted(os.listdir('../Results')):
    if f.startswith('rl_') or f.startswith('agent_') or f.startswith('wfo_'):
        size = os.path.getsize(f'../Results/{f}') / 1024
        print(f'  {f:<45} {size:>8.1f} KB')


