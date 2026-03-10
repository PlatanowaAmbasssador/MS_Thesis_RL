# 03 — Config 2: Concentrated Momentum Alpha (2x/Day)
# Sortino reward, asymmetric penalty, full equity range, larger scorer
# 4 HP configs → re-tuned at every retrain fold → sliding 24m window

import os, time
import numpy as np
import pandas as pd
import torch

RUN_BASELINES = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
else:
    print('Device: CPU')

# NOTE: Uses intraday pipeline (2x/day sessions)
from functions.data_pipeline_intraday import build_dataset
from functions.RL_1.train import train_walk_forward, generate_wfo_folds, count_wfo_folds
from functions.baseline import run_all_baselines

dataset = build_dataset('../Data/Outputs/Filtered/Data')

## 1. Preview WFO Folds (2-month test/val/step)

wfo_info = count_wfo_folds(
    dataset['trading_dates'],
    train_months=24, val_months=2, test_months=2,
    step_months=2, embargo_days=5,
)
print(f'Total folds: {wfo_info["n_folds"]}')
if wfo_info['n_folds'] > 0:
    print(f'First test: {wfo_info["first_fold"]["test_start"]} → {wfo_info["first_fold"]["test_end"]}')
    print(f'Last test:  {wfo_info["last_fold"]["test_start"]} → {wfo_info["last_fold"]["test_end"]}')
    print(f'OOS: {wfo_info["total_test_period"][0]} → {wfo_info["total_test_period"][1]}')

## 2. Train (Walk-Forward) — 2-month folds

t0 = time.time()

rl_results = train_walk_forward(
    dataset,
    train_months=24,
    val_months=2,
    test_months=2,
    step_months=2,
    embargo_days=5,
    n_epochs=40,
    patience=7,
    min_epochs=15,
    transaction_cost_bps=5.0,
    turnover_penalty=0.001,
    variance_penalty=0.0,
    tc_curriculum_frac=0.0,
    lookback_window=40,     # 40 sessions = 20 trading days
    results_dir='../Results_Config2',
    verbose=True,
)

print(f'\n\nTotal time: {(time.time()-t0)/60:.1f} minutes')

## 3. Results

print('=' * 60)
print('OUT-OF-SAMPLE PERFORMANCE — CONFIG 2 (Concentrated Momentum Alpha)')
print('=' * 60)
rl_m = pd.read_csv('../Results_Config2/rl_performance_metrics.csv', index_col=0)
print(rl_m.to_string())

fold_log = pd.read_csv('../Results_Config2/rl_fold_log.csv')
print(f'Retrains: {fold_log["retrained"].sum()} / {len(fold_log)} folds')
print(fold_log.to_string())

## 4. Baselines

if RUN_BASELINES:
    oos_start = rl_results["rl_equity"].index[0].strftime('%Y-%m-%d %H:%M:%S')
    oos_end = rl_results["rl_equity"].index[-1].strftime('%Y-%m-%d %H:%M:%S')

    print(f'\nRunning baselines on OOS period: {oos_start} → {oos_end}')
    bl_results = run_all_baselines(
        dataset,
        start_date=oos_start,
        end_date=oos_end,
        transaction_cost_bps=5.0,
        results_dir='../Results_Config2',
        tag='oos',
        verbose=True,
    )

    bl_metrics = pd.read_csv('../Results_Config2/performance_metrics_oos.csv', index_col=0)
    rl_row = rl_m.loc[['RL Agent']]
    combined = pd.concat([bl_metrics, rl_row]).sort_values('IR2', ascending=False)
    metric_cols = [
        'Absolute Return (%)', 'ARC (%)', 'ASD (%)', 'Max Drawdown (%)',
        'MLD (years)', 'IR1', 'IR2', 'Sharpe', 'Sortino', 'Calmar', 'N Days',
    ]
    metric_cols = [c for c in metric_cols if c in combined.columns]
    if 'Avg Daily Turnover (%)' in combined.columns:
        metric_cols.append('Avg Daily Turnover (%)')
    print('\n' + '=' * 80)
    print('COMBINED COMPARISON — CONFIG 2 (same OOS period)')
    print('=' * 80)
    print(combined[metric_cols].to_string())

## 5. Files Saved

print('\n2x/Day RL files in ../Results_Config2/:')
for f in sorted(os.listdir('../Results_Config2')):
    if f.startswith('rl_') or f.startswith('agent_') or f.startswith('wfo_'):
        size = os.path.getsize(f'../Results_Config2/{f}') / 1024
        print(f'  {f:<45} {size:>8.1f} KB')
