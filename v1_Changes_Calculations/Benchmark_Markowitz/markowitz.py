"""
markowitz.py — Classic Markowitz Mean-Variance Optimization with WFO
=====================================================================
Walk-Forward Markowitz baseline for comparison with RL portfolio allocation.

Two variants:
  1. Max Sharpe Portfolio — maximize risk-adjusted returns
  2. Min Variance Portfolio — minimize portfolio volatility (risk-parity-like)

Both use same WFO structure as RL runs:
  5yr train / 1yr val / 1yr test / 1yr step → 16 folds, OOS 2009-2025

Rebalancing: monthly within each test window.
Estimation: sample covariance + Ledoit-Wolf shrinkage on training window.
Constraints: long-only, fully invested, max 10% per stock.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


# =============================================================================
# PORTFOLIO OPTIMIZATION SOLVERS
# =============================================================================

def _max_sharpe_weights(mu, cov, max_weight=0.10, rf=0.0):
    """
    Find portfolio weights that maximize the Sharpe ratio.
    Constraints: long-only, fully invested, max weight per asset.
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    def neg_sharpe(w):
        port_ret = w @ mu - rf
        port_vol = np.sqrt(w @ cov @ w + 1e-10)
        return -port_ret / port_vol

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    try:
        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if result.success:
            w = np.maximum(result.x, 0)
            return w / w.sum()
    except Exception:
        pass

    return w0  # fallback to equal weight


def _min_variance_weights(cov, max_weight=0.10):
    """
    Find minimum variance portfolio weights.
    Constraints: long-only, fully invested, max weight per asset.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])

    def port_var(w):
        return w @ cov @ w

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    try:
        result = minimize(port_var, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if result.success:
            w = np.maximum(result.x, 0)
            return w / w.sum()
    except Exception:
        pass

    return w0


# =============================================================================
# WALK-FORWARD MARKOWITZ
# =============================================================================

def run_markowitz_wfo(dataset, variant="max_sharpe",
                      train_months=60, val_months=12,
                      test_months=12, step_months=12,
                      rebalance_freq_days=21,
                      max_weight=0.10, top_k=20,
                      transaction_cost_bps=2.0,
                      annualization=252,
                      verbose=True):
    """
    Walk-Forward Markowitz optimization.

    Args:
        dataset: dict from build_dataset()
        variant: "max_sharpe" or "min_variance"
        train_months: training window in months
        val_months: validation window (used for embargo only)
        test_months: test window
        step_months: step between folds
        rebalance_freq_days: rebalance every N trading days within test
        max_weight: maximum weight per stock
        top_k: number of stocks to select (by momentum)
        transaction_cost_bps: TC in basis points
        annualization: 252 for daily

    Returns:
        dict with equity curve, metrics, fold details, daily returns
    """
    dates = dataset["trading_dates"]
    close = dataset["daily_close"]
    mask = dataset["daily_mask"]
    qqq = dataset["qqq"]
    tickers = dataset["tickers"]
    n_tickers = len(tickers)

    tc_rate = transaction_cost_bps / 10_000

    # Build folds (same as RL WFO)
    from dateutil.relativedelta import relativedelta
    folds = []
    first_date = dates[0]
    fold_id = 0
    while True:
        train_start = first_date + relativedelta(months=fold_id * step_months)
        train_end = train_start + relativedelta(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)

        # Snap to actual trading dates
        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]

        if len(train_dates) < 252 or len(test_dates) < 10:
            break
        if test_dates[-1] > dates[-1]:
            break

        folds.append({
            "fold_id": fold_id + 1,
            "train_dates": train_dates,
            "test_dates": test_dates,
        })
        fold_id += 1
        if fold_id > 50:  # safety
            break

    if verbose:
        print(f"\n{'='*70}")
        print(f"MARKOWITZ {variant.upper()} — Walk-Forward Optimization")
        print(f"{'='*70}")
        print(f"  Folds: {len(folds)}")
        print(f"  Train: {train_months}mo | Test: {test_months}mo | Step: {step_months}mo")
        print(f"  Rebalance: every {rebalance_freq_days} trading days")
        print(f"  Max weight: {max_weight:.0%} | Top-k: {top_k}")
        print(f"  TC: {transaction_cost_bps}bps | Annualization: {annualization}")

    # Run walk-forward
    all_test_returns = []
    all_qqq_returns = []
    fold_log = []

    for fold in folds:
        fid = fold["fold_id"]
        train_dates = fold["train_dates"]
        test_dates = fold["test_dates"]

        if verbose:
            print(f"\n  Fold {fid}/{len(folds)}: "
                  f"Train {train_dates[0].strftime('%Y-%m-%d')}→{train_dates[-1].strftime('%Y-%m-%d')} | "
                  f"Test {test_dates[0].strftime('%Y-%m-%d')}→{test_dates[-1].strftime('%Y-%m-%d')}")

        # ----- ESTIMATION on training window -----
        train_close = close.loc[train_dates]
        train_mask = mask.loc[train_dates]

        # Find stocks tradable for >80% of training window
        tradable_pct = train_mask.mean()
        eligible = tradable_pct[tradable_pct > 0.80].index.tolist()

        # Filter to those with price data
        train_rets = train_close[eligible].pct_change().dropna(how="all")
        valid_stocks = train_rets.columns[train_rets.notna().mean() > 0.80].tolist()

        if len(valid_stocks) < 5:
            if verbose:
                print(f"    ⚠ Only {len(valid_stocks)} valid stocks, using equal weight")
            # Equal weight fallback for this fold
            test_rets_all = close.loc[test_dates].pct_change().dropna(how="all")
            qqq_rets = (qqq.loc[test_dates, "qqq_close"].pct_change().dropna())
            common_idx = test_rets_all.index.intersection(qqq_rets.index)
            port_rets = test_rets_all.loc[common_idx].mean(axis=1)
            all_test_returns.append(port_rets)
            all_qqq_returns.append(qqq_rets.loc[common_idx])
            fold_log.append({"fold_id": fid, "n_stocks": 0, "variant": variant})
            continue

        # Top-k by momentum (last 60 days of training)
        mom_period = min(60, len(train_close) - 1)
        momentum = (train_close[valid_stocks].iloc[-1] /
                     train_close[valid_stocks].iloc[-mom_period - 1] - 1)
        momentum = momentum.dropna().sort_values(ascending=False)
        selected = momentum.head(top_k).index.tolist()
        n_sel = len(selected)

        if n_sel < 3:
            selected = valid_stocks[:top_k]
            n_sel = len(selected)

        # Compute expected returns and covariance (with shrinkage)
        sel_rets = train_rets[selected].dropna()
        if len(sel_rets) < 60:
            sel_rets = train_rets[selected].fillna(0)

        mu = sel_rets.mean().values * annualization
        cov = sel_rets.cov().values * annualization

        # Ledoit-Wolf shrinkage for covariance stability
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(sel_rets.fillna(0).values)
            cov = lw.covariance_ * annualization
        except Exception:
            pass  # use sample covariance

        # Solve optimization
        if variant == "max_sharpe":
            weights = _max_sharpe_weights(mu, cov, max_weight=max_weight)
        else:
            weights = _min_variance_weights(cov, max_weight=max_weight)

        if verbose:
            top3_idx = np.argsort(weights)[-3:][::-1]
            top3 = [(selected[i], weights[i]) for i in top3_idx]
            print(f"    {n_sel} stocks | Top-3: " +
                  ", ".join([f"{t} {w:.1%}" for t, w in top3]))

        # ----- TEST: simulate daily returns with rebalancing -----
        test_close = close.loc[test_dates]
        test_rets_sel = test_close[selected].pct_change().fillna(0)
        qqq_test = qqq.loc[test_dates, "qqq_close"].pct_change().fillna(0)

        # Align indices
        common_idx = test_rets_sel.index[1:]  # skip first NaN day
        test_rets_sel = test_rets_sel.loc[common_idx]
        qqq_test = qqq_test.loc[qqq_test.index.isin(common_idx)]
        common_idx = test_rets_sel.index.intersection(qqq_test.index)
        test_rets_sel = test_rets_sel.loc[common_idx]
        qqq_test = qqq_test.loc[common_idx]

        # Simulate with position drift and periodic rebalancing
        current_w = weights.copy()
        port_daily_rets = []
        days_since_rebal = 0

        for day_idx in range(len(common_idx)):
            day_rets = test_rets_sel.iloc[day_idx].values

            # Portfolio return
            port_ret = np.dot(current_w, day_rets)

            # Transaction cost on rebalance days
            tc = 0.0
            if days_since_rebal >= rebalance_freq_days:
                # Rebalance back to target
                turnover = np.abs(current_w - weights).sum()
                tc = turnover * tc_rate
                current_w = weights.copy()
                days_since_rebal = 0

            port_ret_net = port_ret - tc
            port_daily_rets.append(port_ret_net)

            # Drift weights
            new_w = current_w * (1 + day_rets)
            w_sum = new_w.sum()
            if w_sum > 1e-8:
                current_w = new_w / w_sum
            days_since_rebal += 1

        port_rets = pd.Series(port_daily_rets, index=common_idx)
        all_test_returns.append(port_rets)
        all_qqq_returns.append(qqq_test)

        # Fold metrics
        eq = np.array([1.0] + list((1 + port_rets).cumprod().values))
        fold_arc = (eq[-1] ** (annualization / len(port_rets)) - 1) * 100
        qqq_eq = np.array([1.0] + list((1 + qqq_test).cumprod().values))
        qqq_arc = (qqq_eq[-1] ** (annualization / len(qqq_test)) - 1) * 100

        fold_log.append({
            "fold_id": fid, "n_stocks": n_sel, "variant": variant,
            "rl_arc": round(fold_arc, 1), "qqq_arc": round(qqq_arc, 1),
        })

        if verbose:
            marker = "✓" if fold_arc > qqq_arc else "✗"
            print(f"    → MKZ ARC: {fold_arc:+.1f}% | QQQ ARC: {qqq_arc:+.1f}% {marker}")

    # ----- STITCH OOS returns -----
    if not all_test_returns:
        print("  ERROR: No valid folds.")
        return {}

    stitched_rets = pd.concat(all_test_returns)
    stitched_qqq = pd.concat(all_qqq_returns)

    # Remove duplicates (overlapping fold boundaries)
    stitched_rets = stitched_rets[~stitched_rets.index.duplicated(keep='first')]
    stitched_qqq = stitched_qqq[~stitched_qqq.index.duplicated(keep='first')]

    # Equity curve
    equity = np.array([1.0] + list((1 + stitched_rets).cumprod().values))
    qqq_equity = np.array([1.0] + list((1 + stitched_qqq).cumprod().values))

    # Metrics
    from functions.baseline import compute_all_metrics
    mkz_metrics = compute_all_metrics(equity, annualization=annualization)
    qqq_metrics = compute_all_metrics(qqq_equity, annualization=annualization)

    if verbose:
        print(f"\n{'='*70}")
        print(f"MARKOWITZ {variant.upper()} — OOS Results")
        print(f"{'='*70}")
        print(f"  OOS: {stitched_rets.index[0].strftime('%Y-%m-%d')} → "
              f"{stitched_rets.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Days: {len(stitched_rets)}")
        print(f"\n  {'METRIC':<30} {'MKZ':>10} {'QQQ':>10}")
        print(f"  {'-'*50}")
        for key in ["ARC (%)", "ASD (%)", "Max Drawdown (%)", "IR2", "Sharpe", "Sortino", "Calmar"]:
            print(f"  {key:<30} {mkz_metrics[key]:>10.4f} {qqq_metrics[key]:>10.4f}")

    return {
        "equity": equity,
        "qqq_equity": qqq_equity,
        "metrics": mkz_metrics,
        "qqq_metrics": qqq_metrics,
        "daily_returns": stitched_rets,
        "qqq_returns": stitched_qqq,
        "fold_log": fold_log,
        "variant": variant,
    }
