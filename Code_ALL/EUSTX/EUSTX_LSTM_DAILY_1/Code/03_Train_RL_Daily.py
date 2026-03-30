# 03 — Train RL Agent: DAILY — LSTM Architecture Search
# ======================================================
# 6 configs: 3 LSTM sizes (small/medium/deep) × 2 top_k (20/30)
# 80 epochs, min_epochs=25, patience=8, gradient_steps=2
# WFO: 5yr train, 6mo val, 6mo test, 6mo step
# Force retrain every fold, non-learning detection
# annualization=252 (daily)

import os, time
import numpy as np
import pandas as pd
import torch

RUN_BASELINES = True
ANNUALIZATION = 252           # ← DAILY (was 504 for 2x/day)
REWARD_TYPE = "log_return"
TURNOVER_PENALTY = 0.003
TRANSACTION_COST_BPS = 2.0
VARIANCE_PENALTY = 0.5

N_EPOCHS = 50
MIN_EPOCHS = 20
PATIENCE = 8

# WFO windows (daily)
TRAIN_MONTHS = 60            # 5 years
VAL_MONTHS = 12              # 1 year
TEST_MONTHS = 12             # 1 year
STEP_MONTHS = 12             # 1 year

# Lookback window for LSTM
LOOKBACK_WINDOW = 60         # 3 months of daily history (was 40 for 2x/day)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
else:
    print('Device: CPU')

from functions.data_pipeline_daily import build_dataset
from functions.RL_1.train import train_walk_forward, count_wfo_folds
from functions.baseline import run_all_baselines

# NOTE: Data path adjusted for new directory structure
# Data/Outputs/Data/ contains: close_prices.csv, tradable_mask.csv, QQQ.csv, VIX.csv, risk_free_data.csv
dataset = build_dataset('../Data/Outputs/Data')

## 1. Preview WFO Folds

wfo_info = count_wfo_folds(
    dataset['trading_dates'],
    train_months=TRAIN_MONTHS, val_months=VAL_MONTHS,
    test_months=TEST_MONTHS, step_months=STEP_MONTHS,
    embargo_days=0,
)
print(f'Total folds: {wfo_info["n_folds"]}')
if wfo_info['n_folds'] > 0:
    print(f'First test: {wfo_info["first_fold"]["test_start"]} → {wfo_info["first_fold"]["test_end"]}')
    print(f'Last test:  {wfo_info["last_fold"]["test_start"]} → {wfo_info["last_fold"]["test_end"]}')
    print(f'OOS: {wfo_info["total_test_period"][0]} → {wfo_info["total_test_period"][1]}')

print(f'\nDaily Run: 6 LSTM-varied configs | {N_EPOCHS} epochs | gradient_steps=2')
print(f'  WFO: {TRAIN_MONTHS}mo train / {VAL_MONTHS}mo val / {TEST_MONTHS}mo test / {STEP_MONTHS}mo step')
print(f'  Reward: {REWARD_TYPE} | TC: {TRANSACTION_COST_BPS}bps | Lookback: {LOOKBACK_WINDOW}')
print(f'  Annualization: {ANNUALIZATION}')

## 2. Train

t0 = time.time()

rl_results = train_walk_forward(
    dataset,
    train_months=TRAIN_MONTHS,
    val_months=VAL_MONTHS,
    test_months=TEST_MONTHS,
    step_months=STEP_MONTHS,
    embargo_days=0,
    n_epochs=N_EPOCHS,
    patience=PATIENCE,
    min_epochs=MIN_EPOCHS,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    turnover_penalty=TURNOVER_PENALTY,
    variance_penalty=VARIANCE_PENALTY,
    tc_curriculum_frac=0.0,
    lookback_window=LOOKBACK_WINDOW,
    results_dir='../Results_Daily',
    verbose=True,
    annualization=ANNUALIZATION,
    reward_type=REWARD_TYPE,
)

elapsed_min = (time.time()-t0)/60
print(f'\n\nTotal time: {elapsed_min:.1f} minutes ({elapsed_min/60:.1f} hours)')

## 3. Results

print('=' * 60)
print(f'OUT-OF-SAMPLE PERFORMANCE (Daily, {TRANSACTION_COST_BPS}bps)')
print('=' * 60)
rl_m = pd.read_csv('../Results_Daily/rl_performance_metrics.csv', index_col=0)
print(rl_m.to_string())

fold_log = pd.read_csv('../Results_Daily/rl_fold_log.csv')
print(f'Retrains: {fold_log["retrained"].sum()} / {len(fold_log)} folds')
print(fold_log.to_string())

## 4. Baselines

if RUN_BASELINES:
    oos_start = rl_results["rl_equity"].index[0].strftime('%Y-%m-%d')
    oos_end = rl_results["rl_equity"].index[-1].strftime('%Y-%m-%d')

    print(f'\nRunning baselines on OOS period: {oos_start} → {oos_end}')
    bl_results = run_all_baselines(
        dataset,
        start_date=oos_start,
        end_date=oos_end,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        results_dir='../Results_Daily',
        tag='oos',
        verbose=True,
        annualization=ANNUALIZATION,
    )

    bl_metrics = pd.read_csv('../Results_Daily/performance_metrics_oos.csv', index_col=0)
    rl_row = rl_m.loc[['RL Agent']]
    combined = pd.concat([bl_metrics, rl_row]).sort_values('IR2', ascending=False)
    metric_cols = [c for c in [
        'Absolute Return (%)', 'ARC (%)', 'ASD (%)', 'Max Drawdown (%)',
        'MLD (years)', 'IR1', 'IR2', 'Sharpe', 'Sortino', 'Calmar', 'N Days',
        'Avg Daily Turnover (%)',
    ] if c in combined.columns]
    print('\n' + '=' * 80)
    print(f'COMBINED COMPARISON — Daily ({TRANSACTION_COST_BPS}bps)')
    print('=' * 80)
    print(combined[metric_cols].to_string())

## 5. Files Saved

print('\nRL files in ../Results_Daily/:')
for f in sorted(os.listdir('../Results_Daily')):
    if f.startswith('rl_') or f.startswith('agent_') or f.startswith('wfo_'):
        size = os.path.getsize(f'../Results_Daily/{f}') / 1024
        print(f'  {f:<45} {size:>8.1f} KB')
