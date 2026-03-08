"""Run baselines on the same OOS period as the already-completed RL run."""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from functions.data_pipeline import build_dataset
from functions.baseline import run_all_baselines

dataset = build_dataset('../Data/Outputs/Filtered/Data')

rl_equity = pd.read_csv('../Results/rl_equity_oos.csv', index_col=0, parse_dates=True)
oos_start = rl_equity.index[0].strftime('%Y-%m-%d')
oos_end = rl_equity.index[-1].strftime('%Y-%m-%d')

print(f'Running baselines on OOS period: {oos_start} → {oos_end}')
bl_results = run_all_baselines(
    dataset,
    start_date=oos_start,
    end_date=oos_end,
    transaction_cost_bps=5.0,
    results_dir='../Results',
    tag='oos',
    verbose=True,
)

rl_m = pd.read_csv('../Results/rl_performance_metrics.csv', index_col=0)
bl_metrics = pd.read_csv('../Results/performance_metrics_oos.csv', index_col=0)
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
print('COMBINED COMPARISON (same OOS period)')
print('=' * 80)
print(combined[metric_cols].to_string())
print('\nDone.')
