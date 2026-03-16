# 03 — Train HRA-SAC: Run 8 — 14-CONFIG HP GRID
# ================================================
# 7 investment styles × 2 RL variants (v1=exploratory, v2=exploitative)
# Per-config: top_k ∈ {10,20,30}, min_equity ∈ {0.50,0.70,0.85}
# RL tuning: lr_critic, batch_size, ent_multiplier, gamma, dropout
# Reward = (port_ret - rf) × 100 / max(equity_frac, 0.3)
# turnover_penalty = 0.003 (up from 0.001)
# Annualization = 504 (correct for 2x/day)

import os, time
import numpy as np
import pandas as pd
import torch

RUN_BASELINES = True
ANNUALIZATION = 504
REWARD_TYPE = "excess_return"
TURNOVER_PENALTY = 0.003

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
else:
    print('Device: CPU')

from functions.data_pipeline_intraday import build_dataset
from functions.RL_1.train import train_walk_forward, count_wfo_folds
from functions.baseline import run_all_baselines

dataset = build_dataset('../Data/Outputs/Filtered/Data')

## 1. Preview WFO Folds

wfo_info = count_wfo_folds(
    dataset['trading_dates'],
    train_months=24, val_months=2, test_months=2,
    step_months=2, embargo_days=0,
)
print(f'Total folds: {wfo_info["n_folds"]}')
if wfo_info['n_folds'] > 0:
    print(f'First test: {wfo_info["first_fold"]["test_start"]} → {wfo_info["first_fold"]["test_end"]}')
    print(f'Last test:  {wfo_info["last_fold"]["test_start"]} → {wfo_info["last_fold"]["test_end"]}')
    print(f'OOS: {wfo_info["total_test_period"][0]} → {wfo_info["total_test_period"][1]}')

print(f'\nRun 8: 14 HP configs | Reward: {REWARD_TYPE} (equity-scaled)')
print(f'       turnover_penalty={TURNOVER_PENALTY} | annualization={ANNUALIZATION}')

## 2. Train

t0 = time.time()

rl_results = train_walk_forward(
    dataset,
    train_months=24,
    val_months=2,
    test_months=2,
    step_months=2,
    embargo_days=0,
    n_epochs=30,
    patience=5,
    min_epochs=10,
    transaction_cost_bps=5.0,
    turnover_penalty=TURNOVER_PENALTY,
    variance_penalty=0.0,
    tc_curriculum_frac=0.0,
    lookback_window=40,
    results_dir='../Results_Intraday',
    verbose=True,
    annualization=ANNUALIZATION,
    reward_type=REWARD_TYPE,
)

print(f'\n\nTotal time: {(time.time()-t0)/60:.1f} minutes')

## 3. Results

print('=' * 60)
print('OUT-OF-SAMPLE PERFORMANCE (Run 8: 14-config grid)')
print('=' * 60)
rl_m = pd.read_csv('../Results_Intraday/rl_performance_metrics.csv', index_col=0)
print(rl_m.to_string())

fold_log = pd.read_csv('../Results_Intraday/rl_fold_log.csv')
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
        results_dir='../Results_Intraday',
        tag='oos',
        verbose=True,
        annualization=ANNUALIZATION,
    )

    bl_metrics = pd.read_csv('../Results_Intraday/performance_metrics_oos.csv', index_col=0)
    rl_row = rl_m.loc[['RL Agent']]
    combined = pd.concat([bl_metrics, rl_row]).sort_values('IR2', ascending=False)
    metric_cols = [c for c in [
        'Absolute Return (%)', 'ARC (%)', 'ASD (%)', 'Max Drawdown (%)',
        'MLD (years)', 'IR1', 'IR2', 'Sharpe', 'Sortino', 'Calmar', 'N Days',
        'Avg Daily Turnover (%)',
    ] if c in combined.columns]
    print('\n' + '=' * 80)
    print('COMBINED COMPARISON — Run 8 (14 configs, equity-scaled reward)')
    print('=' * 80)
    print(combined[metric_cols].to_string())

## 5. Files Saved

print('\nRL files in ../Results_Intraday/:')
for f in sorted(os.listdir('../Results_Intraday')):
    if f.startswith('rl_') or f.startswith('agent_') or f.startswith('wfo_'):
        size = os.path.getsize(f'../Results_Intraday/{f}') / 1024
        print(f'  {f:<45} {size:>8.1f} KB')
