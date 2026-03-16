"""
environment.py — RL Environment for Portfolio Allocation (v2.1)
==============================================================
Fixes from v2:
    - VECTORIZED _get_state() — precomputes feature tensor, no Python loops
    - 50-100x faster state construction (2s vs 200s per epoch)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DifferentialSharpe:
    """Online differential Sharpe ratio estimator."""
    def __init__(self, eta: float = 0.005):
        self.eta = eta
        self.A = 0.0
        self.B = 0.0
        self._initialized = False

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self._initialized = False

    def compute(self, portfolio_return: float) -> float:
        R = portfolio_return
        if not self._initialized:
            self.A = R
            self.B = R ** 2
            self._initialized = True
            return 0.0
        denom = (self.B - self.A ** 2) ** 1.5
        dS = 0.0 if abs(denom) < 1e-12 else (self.B * (R - self.A) - 0.5 * self.A * (R ** 2 - self.B)) / denom
        self.A += self.eta * (R - self.A)
        self.B += self.eta * (R ** 2 - self.B)
        return dS


class PortfolioEnv:
    def __init__(self, dataset, start_date=None, end_date=None,
                 transaction_cost_bps=5.0, turnover_penalty=0.001,
                 reward_type="sharpe", sharpe_eta=0.005,
                 lookback_window=60, variance_penalty=0.0,
                 tc_curriculum_frac=0.0):
        self.all_tickers = dataset["tickers"]
        self.n_tickers = len(self.all_tickers)
        self.per_asset_features = dataset["per_asset_features"]
        self.global_features = dataset["global_features"]
        self.daily_close = dataset["daily_close"]
        self.daily_mask = dataset["daily_mask"]
        self.qqq = dataset["qqq"]

        # Risk-free rate for cash returns
        rf_data = dataset.get("rf_rate")
        if rf_data is not None and "rf_daily" in rf_data.columns:
            self.rf_rate = rf_data["rf_daily"]
        else:
            self.rf_rate = None  # cash earns 0%

        all_dates = dataset["trading_dates"]
        if start_date:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]
        self.dates = all_dates
        self.n_steps = len(self.dates) - 1

        self.tc_rate_target = transaction_cost_bps / 10_000
        self.tc_rate = self.tc_rate_target
        self.turnover_penalty = turnover_penalty
        self.reward_type = reward_type
        self.lookback_window = lookback_window
        self.variance_penalty = variance_penalty
        self.tc_curriculum_frac = tc_curriculum_frac

        self.daily_returns = self.daily_close.loc[self.dates] / self.daily_close.loc[self.dates].shift(1) - 1
        self.feature_names = sorted(self.per_asset_features.columns.get_level_values("feature").unique())
        self.n_asset_features = len(self.feature_names)
        self.n_global_features = self.global_features.shape[1]
        self.all_trading_dates = dataset["trading_dates"]
        self.diff_sharpe = DifferentialSharpe(eta=sharpe_eta)

        # === PRECOMPUTE feature tensor for fast lookback ===
        self._precompute_feature_arrays()

        self.current_step = 0
        self.weights = None
        self.portfolio_value = 1.0
        self.done = False
        self.history = {k: [] for k in [
            "date", "portfolio_return", "portfolio_return_net", "turnover",
            "transaction_cost", "reward", "portfolio_value", "weights",
            "qqq_return", "cash_weight", "equity_fraction", "rf_earned",
        ]}

    def _precompute_feature_arrays(self):
        """
        Precompute per-asset features as a numpy array indexed by
        (date_position, ticker_position, feature_position).
        This eliminates all Python loops from _get_state().
        """
        pa = self.per_asset_features
        all_dates = self.all_trading_dates
        tickers = self.all_tickers
        feat_names = self.feature_names

        n_dates = len(all_dates)
        n_tickers = len(tickers)
        n_feats = len(feat_names)

        # Build (n_dates, n_tickers, n_features) array
        self._feature_array = np.zeros((n_dates, n_tickers, n_feats), dtype=np.float32)

        for f_i, feat in enumerate(feat_names):
            for t_i, ticker in enumerate(tickers):
                if (ticker, feat) in pa.columns:
                    col = pa[(ticker, feat)]
                    # Align to all_trading_dates
                    vals = col.reindex(all_dates).values.astype(np.float32)
                    self._feature_array[:, t_i, f_i] = np.nan_to_num(vals, 0.0)

        # Also precompute global features array
        self._global_array = np.nan_to_num(
            self.global_features.reindex(all_dates).values.astype(np.float32), 0.0
        )

        # Precompute tradable masks
        self._mask_array = self.daily_mask.reindex(all_dates).values.astype(bool)

        # Map dates to integer indices for fast lookup
        self._date_to_idx = {d: i for i, d in enumerate(all_dates)}

    def _get_tradable_mask(self, date):
        idx = self._date_to_idx.get(date)
        if idx is not None:
            return self._mask_array[idx]
        return self.daily_mask.loc[date].values.astype(bool)

    def _get_state(self):
        """Vectorized state construction — no Python loops."""
        date = self.dates[self.current_step]
        date_idx = self._date_to_idx[date]
        tradable = self._mask_array[date_idx]
        n_tradable = tradable.sum()

        # Lookback window indices
        start_idx = max(0, date_idx - self.lookback_window + 1)
        W_actual = date_idx - start_idx + 1

        # Extract (W_actual, n_tickers, n_features) then select tradable
        raw_window = self._feature_array[start_idx:date_idx + 1, tradable, :]  # (W_actual, n_tradable, F)

        # Right-align into fixed-size window
        asset_window = np.zeros((n_tradable, self.lookback_window, self.n_asset_features), dtype=np.float32)
        offset = self.lookback_window - W_actual
        # raw_window is (W, n_tradable, F), we need (n_tradable, W, F)
        asset_window[:, offset:, :] = raw_window.transpose(1, 0, 2)

        # Global features (single day)
        global_feats = self._global_array[date_idx]

        # Weights
        if self.weights is None:
            w = np.ones(n_tradable + 1, dtype=np.float32) / (n_tradable + 1)
        else:
            w_stocks = self.weights[:self.n_tickers][tradable].astype(np.float32)
            w_cash = np.float32(self.weights[-1])
            w = np.concatenate([w_stocks, [w_cash]])

        return {"asset_features": asset_window, "global_features": global_feats,
                "weights": w, "n_tradable": n_tradable}

    def _action_to_weights(self, action, tradable):
        n_tradable = tradable.sum()
        stock_w = np.clip(action[:n_tradable], 0, 1)
        cash_w = np.clip(action[n_tradable], 0, 1)
        total = stock_w.sum() + cash_w
        if total > 0:
            stock_w /= total
            cash_w /= total
        full_w = np.zeros(self.n_tickers, dtype=np.float64)
        full_w[tradable] = stock_w
        return full_w, float(cash_w)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.portfolio_value = 1.0
        tradable = self._get_tradable_mask(self.dates[0])
        n_t = tradable.sum()
        self.weights = np.zeros(self.n_tickers + 1, dtype=np.float64)
        eq_w = 1.0 / (n_t + 1)
        self.weights[:self.n_tickers][tradable] = eq_w
        self.weights[-1] = eq_w
        self.diff_sharpe.reset()
        for k in self.history:
            self.history[k] = []
        self.tc_rate = 0.0 if self.tc_curriculum_frac > 0 else self.tc_rate_target
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done.")
        date_t = self.dates[self.current_step]
        date_t1 = self.dates[self.current_step + 1]
        tradable_t = self._get_tradable_mask(date_t)

        if self.tc_curriculum_frac > 0 and self.n_steps > 0:
            progress = self.current_step / self.n_steps
            self.tc_rate = self.tc_rate_target * min(progress / self.tc_curriculum_frac, 1.0)

        stock_w, cash_w = self._action_to_weights(action, tradable_t)
        old_stocks = self.weights[:self.n_tickers] if self.weights is not None else np.zeros(self.n_tickers)
        old_cash = self.weights[-1] if self.weights is not None else 0.0
        turnover = np.abs(stock_w - old_stocks).sum() + abs(cash_w - old_cash)
        tc = turnover * self.tc_rate

        returns_t1 = np.nan_to_num(self.daily_returns.loc[date_t1].values.copy(), nan=0.0)
        # Cash earns risk-free rate (if available)
        rf_daily = 0.0
        if self.rf_rate is not None and date_t1 in self.rf_rate.index:
            rf_daily = float(self.rf_rate.loc[date_t1])
            if np.isnan(rf_daily):
                rf_daily = 0.0
        port_ret_gross = np.dot(stock_w, returns_t1) + cash_w * rf_daily
        port_ret_net = port_ret_gross - tc
        self.portfolio_value *= (1 + port_ret_net)

        new_stock = stock_w * (1 + returns_t1)
        new_cash = cash_w * (1 + rf_daily)
        total = new_stock.sum() + new_cash
        if total > 0:
            drifted_stock = new_stock / total
            drifted_cash = new_cash / total
        else:
            drifted_stock = stock_w
            drifted_cash = cash_w
        self.weights = np.zeros(self.n_tickers + 1, dtype=np.float64)
        self.weights[:self.n_tickers] = drifted_stock
        self.weights[-1] = drifted_cash

        tradable_t1 = self._get_tradable_mask(date_t1)
        exiting = (~tradable_t1) & (self.weights[:self.n_tickers] > 0)
        if exiting.any():
            ew = self.weights[:self.n_tickers][exiting].sum()
            self.weights[:self.n_tickers][exiting] = 0.0
            remaining = tradable_t1 & (self.weights[:self.n_tickers] > 0)
            if remaining.any():
                self.weights[:self.n_tickers][remaining] += ew * self.weights[:self.n_tickers][remaining] / self.weights[:self.n_tickers][remaining].sum()

        qqq_ret = (self.qqq.loc[date_t1, "qqq_close"] / self.qqq.loc[date_t, "qqq_close"]) - 1

        if self.reward_type == "sharpe":
            # Excess return over equal-weight tradable stocks
            ew_ret = np.mean(returns_t1[tradable_t]) if tradable_t.any() else 0.0
            excess_ret = port_ret_net - ew_ret
            reward = self.diff_sharpe.compute(excess_ret)
            reward -= self.turnover_penalty * turnover
        else:
            reward = port_ret_net - self.turnover_penalty * turnover
        if self.variance_penalty > 0:
            recent = self.history["portfolio_return_net"][-20:]
            if len(recent) >= 5:
                downside = [r for r in recent if r < 0]
                if len(downside) >= 2:
                    reward -= self.variance_penalty * np.var(downside)
        reward *= 100.0

        self.history["date"].append(date_t1)
        self.history["portfolio_return"].append(port_ret_gross)
        self.history["portfolio_return_net"].append(port_ret_net)
        self.history["turnover"].append(turnover)
        self.history["transaction_cost"].append(tc)
        self.history["reward"].append(reward)
        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["weights"].append(stock_w.copy())
        self.history["qqq_return"].append(qqq_ret)
        self.history["cash_weight"].append(cash_w)
        self.history["equity_fraction"].append(1.0 - cash_w)
        self.history["rf_earned"].append(cash_w * rf_daily)

        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True

        if not self.done:
            next_state = self._get_state()
        else:
            n_t = self._get_tradable_mask(date_t1).sum()
            next_state = {"asset_features": np.zeros((n_t, self.lookback_window, self.n_asset_features), dtype=np.float32),
                          "global_features": np.zeros(self.n_global_features, dtype=np.float32),
                          "weights": np.ones(n_t + 1, dtype=np.float32) / (n_t + 1), "n_tradable": n_t}

        return next_state, reward, self.done, {
            "date": date_t1, "portfolio_return_gross": port_ret_gross,
            "portfolio_return_net": port_ret_net, "turnover": turnover,
            "transaction_cost": tc, "portfolio_value": self.portfolio_value,
            "qqq_return": qqq_ret, "n_tradable": tradable_t.sum(), "cash_weight": cash_w,
        }

    @property
    def action_dim(self):
        return self._get_tradable_mask(self.dates[self.current_step]).sum() + 1

    def get_results(self):
        df = pd.DataFrame({k: self.history[k] for k in self.history if k != "weights"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["qqq_value"] = (1 + df["qqq_return"]).cumprod()
        return df


def compute_metrics(results, annualization=252, risk_free_rate=0.0):
    n = len(results)
    rf = risk_free_rate / annualization
    m = {}
    for pfx, col in [("portfolio", "portfolio_return_net"), ("qqq", "qqq_return")]:
        r = results[col]
        cum = (1 + r).cumprod()
        tr = cum.iloc[-1] - 1
        ar = (1 + tr) ** (annualization / n) - 1
        av = r.std() * np.sqrt(annualization)
        ex = r - rf
        sh = ex.mean() / ex.std() * np.sqrt(annualization) if ex.std() > 0 else 0.0
        rm = cum.cummax()
        dd = (cum - rm) / rm
        md = dd.min()
        cal = ar / abs(md) if abs(md) > 0 else 0.0
        m[f"{pfx}_total_return"] = tr
        m[f"{pfx}_ann_return"] = ar
        m[f"{pfx}_ann_vol"] = av
        m[f"{pfx}_sharpe"] = sh
        m[f"{pfx}_max_drawdown"] = md
        m[f"{pfx}_calmar"] = cal
    m["avg_daily_turnover"] = results["turnover"].mean()
    m["total_transaction_costs"] = results["transaction_cost"].sum()
    m["avg_cash_weight"] = results["cash_weight"].mean() if "cash_weight" in results else 0.0
    m["n_days"] = n
    return m


def print_metrics(metrics):
    print("\n" + "=" * 55)
    print(f"{'METRIC':<30} {'PORTFOLIO':>10} {'QQQ':>10}")
    print("=" * 55)
    for label, key, fmt in [("Total Return", "total_return", "{:.2%}"), ("Ann Return", "ann_return", "{:.2%}"),
                             ("Ann Vol", "ann_vol", "{:.2%}"), ("Sharpe", "sharpe", "{:.3f}"),
                             ("Max DD", "max_drawdown", "{:.2%}"), ("Calmar", "calmar", "{:.3f}")]:
        print(f"  {label:<28} {fmt.format(metrics[f'portfolio_{key}']):>10} {fmt.format(metrics[f'qqq_{key}']):>10}")
    print("-" * 55)
    print(f"  {'Avg Turnover':<28} {metrics['avg_daily_turnover']:>10.2%}")
    print(f"  {'Avg Cash':<28} {metrics.get('avg_cash_weight', 0):>10.2%}")
    print(f"  {'Days':<28} {metrics['n_days']:>10d}")
    print("=" * 55)
