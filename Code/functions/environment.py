"""
environment.py — RL Environment for Portfolio Allocation (v2)
==============================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Changes from v1:
    - State is a DICT with 60-day lookback window (not flat vector)
    - Cash position (N+1 weights)
    - Action = Dirichlet portfolio weights (not deviations)
    - Variance penalty in reward
    - TC curriculum (ramp from 0 to target over training)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DifferentialSharpe:
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

        self.current_step = 0
        self.weights = None
        self.portfolio_value = 1.0
        self.done = False
        self.history = {k: [] for k in [
            "date", "portfolio_return", "portfolio_return_net", "turnover",
            "transaction_cost", "reward", "portfolio_value", "weights",
            "qqq_return", "cash_weight",
        ]}

    def _get_tradable_mask(self, date):
        return self.daily_mask.loc[date].values.astype(bool)

    def _get_state(self):
        date = self.dates[self.current_step]
        tradable = self._get_tradable_mask(date)
        n_tradable = tradable.sum()
        tradable_tickers = [t for t, m in zip(self.all_tickers, tradable) if m]

        date_idx = self.all_trading_dates.get_loc(date)
        start_idx = max(0, date_idx - self.lookback_window + 1)
        lookback_dates = self.all_trading_dates[start_idx:date_idx + 1]
        W_actual = len(lookback_dates)

        asset_window = np.zeros((n_tradable, self.lookback_window, self.n_asset_features), dtype=np.float32)
        for d_i, d in enumerate(lookback_dates):
            offset = self.lookback_window - W_actual + d_i
            try:
                pa_row = self.per_asset_features.loc[d]
                for a_i, ticker in enumerate(tradable_tickers):
                    for f_i, feat in enumerate(self.feature_names):
                        val = pa_row.get((ticker, feat), np.nan)
                        asset_window[a_i, offset, f_i] = val if not np.isnan(val) else 0.0
            except KeyError:
                pass

        global_feats = np.nan_to_num(self.global_features.loc[date].values.astype(np.float32), 0.0)

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
        port_ret_gross = np.dot(stock_w, returns_t1)
        port_ret_net = port_ret_gross - tc
        self.portfolio_value *= (1 + port_ret_net)

        # Drift
        new_stock = stock_w * (1 + returns_t1)
        total = new_stock.sum() + cash_w
        if total > 0:
            drifted_stock = new_stock / total * (1 - cash_w / total)
            drifted_cash = cash_w / total
        else:
            drifted_stock = stock_w
            drifted_cash = cash_w
        self.weights = np.zeros(self.n_tickers + 1, dtype=np.float64)
        self.weights[:self.n_tickers] = drifted_stock
        self.weights[-1] = drifted_cash

        # Membership changes
        tradable_t1 = self._get_tradable_mask(date_t1)
        exiting = (~tradable_t1) & (self.weights[:self.n_tickers] > 0)
        if exiting.any():
            ew = self.weights[:self.n_tickers][exiting].sum()
            self.weights[:self.n_tickers][exiting] = 0.0
            remaining = tradable_t1 & (self.weights[:self.n_tickers] > 0)
            if remaining.any():
                self.weights[:self.n_tickers][remaining] += ew * self.weights[:self.n_tickers][remaining] / self.weights[:self.n_tickers][remaining].sum()

        # Reward
        if self.reward_type == "sharpe":
            reward = self.diff_sharpe.compute(port_ret_net)
            reward -= self.turnover_penalty * turnover
        else:
            reward = port_ret_net - self.turnover_penalty * turnover
        if self.variance_penalty > 0:
            recent = self.history["portfolio_return_net"][-20:]
            if len(recent) >= 5:
                reward -= self.variance_penalty * np.var(recent)

        qqq_ret = (self.qqq.loc[date_t1, "qqq_close"] / self.qqq.loc[date_t, "qqq_close"]) - 1

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
