"""
baselines.py — Baseline Strategies + Performance Evaluation
=============================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Five baselines:
    1. QQQ Buy-and-Hold
    2. Equal-Weight Monthly Rebalance
    3. Inverse-Volatility Weighted (daily rebalance)
    4. Momentum (top quintile by 20d returns, daily rebalance)
    5. Supervised Learning + Mean-Variance Optimization

Outputs saved to Results/ folder:
    - equity_curves_{tag}.csv
    - daily_returns_{tag}.csv
    - performance_metrics_{tag}.csv
    - turnover_{tag}.csv

Usage (from notebook):
    from functions.data_pipeline import build_dataset
    from functions.baseline import run_all_baselines

    dataset = build_dataset("../Data/Outputs/Filtered/Data")
    all_results = run_all_baselines(dataset, results_dir="../Results")
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from .environment import PortfolioEnv


# =============================================================================
# PERFORMANCE METRICS (user's framework)
# =============================================================================

def equity_to_returns(tab):
    """Convert equity curve (list/array) to simple returns."""
    tab = np.array(tab, dtype=np.float64)
    return (tab[1:] / tab[:-1]) - 1


def absolute_return(tab):
    """Total return from equity curve, in percent."""
    ret = equity_to_returns(tab)
    if len(ret) == 0:
        return 0.0
    return (np.prod(1 + ret) - 1.0) * 100


def ARC(tab):
    """Annualized Rate of Return (%), assuming 252 trading days/year."""
    tab = np.array(tab, dtype=np.float64)
    ret = equity_to_returns(tab)
    length = len(tab)
    if length <= 1:
        return 0.0
    a_rtn = np.prod(1 + ret[:-1])
    if a_rtn <= 0:
        return 0.0
    return 100 * (math.pow(a_rtn, 252 / length) - 1)


def MaximumDrawdown(tab):
    """Maximum Drawdown (%), from equity curve."""
    eqr = equity_to_returns(tab)
    if len(eqr) == 0:
        return 0.0
    cum_returns = np.cumprod(1 + eqr)
    cum_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_max - cum_returns) / cum_max
    return np.max(drawdowns) * 100


def ASD(tab):
    """Annualized Standard Deviation (%), assuming 252 trading days/year."""
    ret = equity_to_returns(tab)
    if len(ret) == 0:
        return 0.0
    return (math.sqrt(252) * np.std(ret)) * 100


def sgn(x):
    if x == 0:
        return 0
    return int(abs(x) / x)


def MLD(tab):
    """Maximum Loss Duration in years (252.03 days/year)."""
    temp = np.array(tab, dtype=np.float64)
    if len(temp) == 0:
        return 1.0
    i = np.argmax(np.maximum.accumulate(temp) - temp)
    if i == 0:
        return len(temp) / 252.03
    j = np.argmax(temp[:i])
    MLD_end = -1
    for k in range(i, len(temp)):
        if (temp[k - 1] < temp[j]) and (temp[j] < temp[k]):
            MLD_end = k
            break
    if MLD_end == -1:
        MLD_end = len(temp)
    return abs(MLD_end - j) / 252.03


def IR1(tab):
    """Information Ratio 1: ARC / ASD (≈ annualized Sharpe)."""
    asd = ASD(tab)
    arc = ARC(tab)
    if asd == 0:
        return 0.0
    return max(arc / asd, 0)


def IR2(tab):
    """
    Information Ratio 2: (ARC^2 * sgn(ARC)) / (ASD * MDD).
    PRIMARY OPTIMIZATION TARGET.
    """
    asd = ASD(tab)
    arc = ARC(tab)
    mdd = MaximumDrawdown(tab)
    denom = asd * mdd
    if denom == 0:
        return 0.0
    numer = (arc ** 2) * sgn(arc)
    return max(numer / denom, 0)


def compute_all_metrics(equity_curve) -> dict:
    """Compute all performance metrics from an equity curve (starting at 1.0)."""
    tab = np.array(equity_curve, dtype=np.float64)
    ret = equity_to_returns(tab)

    # Annualized Sharpe (daily returns → annualized)
    if len(ret) > 1 and np.std(ret) > 0:
        sharpe_ann = (np.mean(ret) / np.std(ret)) * np.sqrt(252)
    else:
        sharpe_ann = 0.0

    # Sortino (penalizes only downside vol)
    downside = np.array([r for r in ret if r < 0])
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(ret) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = 0.0

    # Calmar (ARC / MaxDD)
    arc_val = ARC(tab)
    mdd_val = MaximumDrawdown(tab)
    calmar = arc_val / mdd_val if mdd_val > 0 else 0.0

    # Number of trades (position changes in the equity curve)
    # This counts sign changes in daily returns as a proxy when positions aren't available
    n_days = len(ret)

    return {
        "Absolute Return (%)": round(absolute_return(tab), 4),
        "ARC (%)": round(arc_val, 4),
        "ASD (%)": round(ASD(tab), 4),
        "Max Drawdown (%)": round(mdd_val, 4),
        "MLD (years)": round(MLD(tab), 4),
        "IR1": round(IR1(tab), 4),
        "IR2": round(IR2(tab), 4),
        "Sharpe": round(sharpe_ann, 4),
        "Sortino": round(sortino, 4),
        "Calmar": round(calmar, 4),
        "N Days": n_days,
    }


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class BaselineStrategy:
    def __init__(self, name: str):
        self.name = name

    def get_target_weights(self, env: PortfolioEnv, step: int) -> np.ndarray:
        raise NotImplementedError

    def run(
        self,
        dataset: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        transaction_cost_bps: float = 5.0,
    ) -> dict:
        env = PortfolioEnv(
            dataset,
            start_date=start_date,
            end_date=end_date,
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty=0.0,
            reward_type="return",
        )
        state = env.reset()
        step = 0
        while not env.done:
            action = self.get_target_weights(env, step)
            state, reward, done, info = env.step(action)
            step += 1

        results = env.get_results()
        equity = np.array([1.0] + list(results["portfolio_value"].values))
        metrics = compute_all_metrics(equity)
        metrics["Avg Daily Turnover (%)"] = round(results["turnover"].mean() * 100, 4)
        metrics["Total TC (%)"] = round(results["transaction_cost"].sum() * 100, 4)
        return {"results": results, "metrics": metrics, "equity": equity, "name": self.name}


# =============================================================================
# BASELINE 1: QQQ BUY-AND-HOLD
# =============================================================================

class QQQBuyHold:
    def __init__(self):
        self.name = "QQQ Buy-and-Hold"

    def run(self, dataset, start_date=None, end_date=None, **kwargs):
        dates = dataset["trading_dates"]
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date:
            dates = dates[dates <= pd.Timestamp(end_date)]

        qqq = dataset["qqq"].loc[dates, "qqq_close"]
        qqq_ret = qqq.pct_change().dropna()
        cum = (1 + qqq_ret).cumprod()
        equity = np.array([1.0] + list(cum.values))
        metrics = compute_all_metrics(equity)
        metrics["Avg Daily Turnover (%)"] = 0.0
        metrics["Total TC (%)"] = 0.0

        results = pd.DataFrame({
            "portfolio_return_net": qqq_ret.values,
            "qqq_return": qqq_ret.values,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "portfolio_value": cum.values,
            "qqq_value": cum.values,
        }, index=qqq_ret.index)
        return {"results": results, "metrics": metrics, "equity": equity, "name": self.name}


# =============================================================================
# BASELINE 2: EQUAL-WEIGHT MONTHLY REBALANCE
# =============================================================================

class EqualWeightMonthly(BaselineStrategy):
    def __init__(self):
        super().__init__("Equal-Weight Monthly")
        self._last_rebalance_month = None

    def get_target_weights(self, env, step):
        date = env.dates[step]
        current_month = (date.year, date.month)
        tradable = env._get_tradable_mask(date)
        n = tradable.sum()
        if self._last_rebalance_month != current_month:
            self._last_rebalance_month = current_month
            w = np.ones(n, dtype=np.float32) / n
            return np.concatenate([w, [0.0]])
        else:
            current_stocks = env.weights[:env.n_tickers][tradable]
            current_cash = env.weights[-1]
            return np.concatenate([current_stocks, [current_cash]]).astype(np.float32)

    def run(self, dataset, start_date=None, end_date=None, transaction_cost_bps=5.0):
        self._last_rebalance_month = None
        return super().run(dataset, start_date, end_date, transaction_cost_bps)


# =============================================================================
# BASELINE 3: INVERSE-VOLATILITY WEIGHTED
# =============================================================================

class InverseVolatility(BaselineStrategy):
    def __init__(self, vol_lookback: int = 20):
        super().__init__("Inverse Volatility")
        self.vol_lookback = vol_lookback

    def get_target_weights(self, env, step):
        date = env.dates[step]
        tradable = env._get_tradable_mask(date)
        tickers = [t for t, m in zip(env.all_tickers, tradable) if m]
        n = len(tickers)

        date_idx = env.dates.get_loc(date)
        start_idx = max(0, date_idx - self.vol_lookback)
        hist_dates = env.dates[start_idx:date_idx + 1]
        hist_close = env.daily_close.loc[hist_dates, tickers]

        log_ret = np.log(hist_close / hist_close.shift(1)).dropna()
        if len(log_ret) < 5:
            w = np.ones(n, dtype=np.float32) / n
            return np.concatenate([w, [0.0]])

        vol = log_ret.std()
        vol = vol.replace(0, np.nan).fillna(vol.median())
        inv_vol = 1.0 / vol
        target_w = (inv_vol / inv_vol.sum()).values.astype(np.float32)
        return np.concatenate([target_w, [0.0]])


# =============================================================================
# BASELINE 4: MOMENTUM (TOP QUINTILE)
# =============================================================================

class MomentumTopQuintile(BaselineStrategy):
    def __init__(self, lookback: int = 20, top_pct: float = 0.20):
        super().__init__(f"Momentum Top-{int(top_pct * 100)}%")
        self.lookback = lookback
        self.top_pct = top_pct

    def get_target_weights(self, env, step):
        date = env.dates[step]
        tradable = env._get_tradable_mask(date)
        tickers = [t for t, m in zip(env.all_tickers, tradable) if m]
        n = len(tickers)

        date_idx = env.dates.get_loc(date)
        if date_idx < self.lookback:
            w = np.ones(n, dtype=np.float32) / n
            return np.concatenate([w, [0.0]])

        past_date = env.dates[date_idx - self.lookback]
        close_now = env.daily_close.loc[date, tickers]
        close_past = env.daily_close.loc[past_date, tickers]
        momentum = (close_now / close_past - 1).fillna(-999)

        n_top = max(1, int(n * self.top_pct))
        top_tickers = set(momentum.nlargest(n_top).index.tolist())

        target_w = np.zeros(n, dtype=np.float32)
        for i, t in enumerate(tickers):
            if t in top_tickers:
                target_w[i] = 1.0 / n_top
        return np.concatenate([target_w, [0.0]])


# =============================================================================
# BASELINE 5: SUPERVISED LEARNING + MEAN-VARIANCE OPTIMIZATION
# =============================================================================

class SupervisedMVO(BaselineStrategy):
    """
    1. Predict next-day returns with HistGradientBoosting (sklearn)
    2. Score-weighted portfolio with max weight constraint
    Re-trains every retrain_freq days on rolling 252-day window.
    """

    def __init__(self, retrain_freq: int = 20, risk_aversion: float = 1.0, max_weight: float = 0.10):
        super().__init__("Supervised + MVO")
        self.retrain_freq = retrain_freq
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.model = None
        self._step_count = 0
        self._dataset = None

    def _build_training_data(self, dataset, end_date_idx, dates):
        pa = dataset["per_asset_features"]
        mask = dataset["daily_mask"]
        close = dataset["daily_close"]
        tickers = dataset["tickers"]
        feat_names = sorted(pa.columns.get_level_values("feature").unique())

        train_dates = dates[:end_date_idx]
        if len(train_dates) < 30:
            return None, None

        log_ret_1d = np.log(close / close.shift(1))
        train_start = max(0, len(train_dates) - 252)
        train_slice = train_dates[train_start:]

        X_list, y_list = [], []
        for date in train_slice:
            date_idx = dates.get_loc(date)
            if date_idx + 1 >= len(dates):
                continue
            next_date = dates[date_idx + 1]
            tradable = mask.loc[date] == 1
            for ticker in tickers:
                if not tradable[ticker]:
                    continue
                feat_vals = []
                for feat in feat_names:
                    val = pa.loc[date].get((ticker, feat), np.nan)
                    feat_vals.append(val if not np.isnan(val) else 0.0)
                target = log_ret_1d.loc[next_date, ticker]
                if np.isnan(target):
                    continue
                X_list.append(feat_vals)
                y_list.append(target)

        if len(X_list) < 100:
            return None, None
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def _predict_returns(self, dataset, date):
        pa = dataset["per_asset_features"]
        mask = dataset["daily_mask"]
        tickers = dataset["tickers"]
        feat_names = sorted(pa.columns.get_level_values("feature").unique())
        tradable = mask.loc[date] == 1
        predictions = {}
        for ticker in tickers:
            if not tradable[ticker]:
                continue
            feat_vals = []
            for feat in feat_names:
                val = pa.loc[date].get((ticker, feat), np.nan)
                feat_vals.append(val if not np.isnan(val) else 0.0)
            X = np.array(feat_vals, dtype=np.float32).reshape(1, -1)
            predictions[ticker] = self.model.predict(X)[0]
        return predictions

    def get_target_weights(self, env, step):
        date = env.dates[step]
        tradable = env._get_tradable_mask(date)
        tickers = [t for t, m in zip(env.all_tickers, tradable) if m]
        n = len(tickers)

        if self._step_count % self.retrain_freq == 0:
            date_idx = env.dates.get_loc(date)
            X, y = self._build_training_data(self._dataset, date_idx, env.dates)
            if X is not None:
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(
                    max_iter=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=20, random_state=42,
                )
                self.model.fit(X, y)
        self._step_count += 1

        if self.model is None:
            w = np.ones(n, dtype=np.float32) / n
            return np.concatenate([w, [0.0]])

        predictions = self._predict_returns(self._dataset, date)
        mu = np.array([predictions.get(t, 0.0) for t in tickers])
        mu_centered = mu - np.median(mu)
        scores = mu_centered / (np.std(mu_centered) + 1e-8) / self.risk_aversion
        scores = np.clip(scores, -3, 3)
        exp_scores = np.exp(scores)
        target_w = exp_scores / exp_scores.sum()
        target_w = np.clip(target_w, 0, self.max_weight)
        target_w = (target_w / target_w.sum()).astype(np.float32)
        return np.concatenate([target_w, [0.0]])

    def run(self, dataset, start_date=None, end_date=None, transaction_cost_bps=5.0):
        self._dataset = dataset
        self._step_count = 0
        self.model = None
        return super().run(dataset, start_date, end_date, transaction_cost_bps)


# =============================================================================
# RUN ALL BASELINES + SAVE RESULTS
# =============================================================================

def run_all_baselines(
    dataset: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_cost_bps: float = 5.0,
    results_dir: str = "../Results",
    tag: str = "full",
    verbose: bool = True,
) -> dict:
    """
    Run all 5 baselines, print comparison table, save CSVs.

    Saves to results_dir/:
        - equity_curves_{tag}.csv
        - daily_returns_{tag}.csv
        - performance_metrics_{tag}.csv
        - turnover_{tag}.csv
    """
    strategies = [
        QQQBuyHold(),
        EqualWeightMonthly(),
        InverseVolatility(vol_lookback=20),
        MomentumTopQuintile(lookback=20, top_pct=0.20),
        SupervisedMVO(retrain_freq=20, risk_aversion=1.0, max_weight=0.10),
    ]

    all_results = {}
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING ALL BASELINES")
        print(f"  Period: {start_date or 'start'} → {end_date or 'end'}")
        print(f"  Transaction cost: {transaction_cost_bps} bps one-way")
        print(f"  Results dir: {out_dir.resolve()}")
        print("=" * 70)

    for strategy in strategies:
        if verbose:
            print(f"\n  ▶ {strategy.name}...", end=" ", flush=True)
        result = strategy.run(
            dataset, start_date=start_date, end_date=end_date,
            transaction_cost_bps=transaction_cost_bps,
        )
        all_results[strategy.name] = result
        if verbose:
            m = result["metrics"]
            print(f"Done — IR2: {m['IR2']:.4f}, ARC: {m['ARC (%)']:.1f}%, "
                  f"MDD: {m['Max Drawdown (%)']:.1f}%")

    # --- Save CSVs ---
    _save_results(all_results, out_dir, tag, verbose)

    if verbose:
        print_comparison_table(all_results)

    return all_results


def _save_results(all_results: dict, out_dir: Path, tag: str, verbose: bool):
    """Save equity curves, returns, turnover, and metrics to CSV."""

    # 1. Equity curves
    equity_df = pd.DataFrame()
    for name, res in all_results.items():
        equity_df[name] = res["results"]["portfolio_value"]
    equity_df.index.name = "date"
    path = out_dir / f"equity_curves_{tag}.csv"
    equity_df.to_csv(path)
    if verbose:
        print(f"\n  💾 Saved: {path}")

    # 2. Daily returns
    returns_df = pd.DataFrame()
    for name, res in all_results.items():
        returns_df[name] = res["results"]["portfolio_return_net"]
    returns_df.index.name = "date"
    path = out_dir / f"daily_returns_{tag}.csv"
    returns_df.to_csv(path)
    if verbose:
        print(f"  💾 Saved: {path}")

    # 3. Turnover
    turnover_df = pd.DataFrame()
    for name, res in all_results.items():
        turnover_df[name] = res["results"]["turnover"]
    turnover_df.index.name = "date"
    path = out_dir / f"turnover_{tag}.csv"
    turnover_df.to_csv(path)
    if verbose:
        print(f"  💾 Saved: {path}")

    # 4. Performance metrics
    metrics_rows = []
    for name, res in all_results.items():
        row = {"Strategy": name}
        row.update(res["metrics"])
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")
    path = out_dir / f"performance_metrics_{tag}.csv"
    metrics_df.to_csv(path)
    if verbose:
        print(f"  💾 Saved: {path}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(all_results: dict) -> None:
    """Print formatted comparison table sorted by IR2."""
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"]["IR2"],
        reverse=True,
    )

    print("\n" + "=" * 130)
    print(f"{'STRATEGY':<25} {'ARC%':>7} {'ASD%':>7} {'MDD%':>7} "
          f"{'MLD':>6} {'IR1':>7} {'IR2':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'TO%':>7}")
    print("=" * 130)

    for name, res in sorted_results:
        m = res["metrics"]
        print(
            f"  {name:<23} "
            f"{m['ARC (%)']:>6.1f} "
            f"{m['ASD (%)']:>6.1f} "
            f"{m['Max Drawdown (%)']:>6.1f} "
            f"{m['MLD (years)']:>5.2f} "
            f"{m['IR1']:>6.3f} "
            f"{m['IR2']:>6.4f} "
            f"{m.get('Sharpe', 0):>6.3f} "
            f"{m.get('Sortino', 0):>7.3f} "
            f"{m.get('Calmar', 0):>6.3f} "
            f"{m.get('Avg Daily Turnover (%)', 0):>6.2f}"
        )

    print("=" * 130)
    print("  (Sorted by IR2 — primary optimization target)")