"""
train.py — Walk-Forward Optimization for HRA-SAC Portfolio Agent (v4)
======================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Key features:
    - Full HP re-selection at EVERY retrain fold (4 configs)
    - Sliding (non-anchored) WFO with Sharpe-based retrain trigger
    - Hierarchical Risk-Aware SAC with cash timing + stock selection
    - 4 HP configs covering LR, model capacity, and cash timing aggressiveness
"""

import numpy as np
import pandas as pd
import time
import io
import json
import gc
from pathlib import Path
from typing import Optional, Dict, List

import torch
from ..environment import PortfolioEnv
from ..baseline import compute_all_metrics
from .sac_agent import SACAgent


# =============================================================================
# SLIDING (NON-ANCHORED) WFO FOLD GENERATOR
# =============================================================================

def generate_wfo_folds(
    trading_dates: pd.DatetimeIndex,
    train_months: int = 24,
    val_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    embargo_days: int = 5,
) -> List[Dict]:
    """
    Generate SLIDING walk-forward folds. Training window is FIXED width
    (not expanding). Includes embargo gap between train and val.
    """
    dates = trading_dates.sort_values()
    data_start = dates[0]
    data_end = dates[-1]

    # First fold: train starts at data_start
    train_start = data_start
    folds = []
    fold_id = 1

    while True:
        train_end_raw = train_start + pd.DateOffset(months=train_months)
        # Embargo: skip 5 trading days after train end
        embargo_end = train_end_raw + pd.DateOffset(days=embargo_days + 2)  # buffer for weekends
        val_start_raw = embargo_end
        val_end_raw = val_start_raw + pd.DateOffset(months=val_months)
        test_start_raw = val_end_raw
        test_end_raw = test_start_raw + pd.DateOffset(months=test_months)

        if test_end_raw > data_end:
            break

        # Map to actual trading dates
        train_d = dates[(dates >= train_start) & (dates < train_end_raw)]
        val_d = dates[(dates >= val_start_raw) & (dates < val_end_raw)]
        test_d = dates[(dates >= test_start_raw) & (dates < test_end_raw)]

        if len(train_d) < 60 or len(val_d) == 0 or len(test_d) == 0:
            break

        folds.append({
            "fold_id": fold_id,
            "train_start": str(train_d[0].date()),
            "train_end": str(train_d[-1].date()),
            "val_start": str(val_d[0].date()),
            "val_end": str(val_d[-1].date()),
            "test_start": str(test_d[0].date()),
            "test_end": str(test_d[-1].date()),
            "n_train": len(train_d),
            "n_val": len(val_d),
            "n_test": len(test_d),
        })

        fold_id += 1
        # SLIDE forward (non-anchored)
        train_start = train_start + pd.DateOffset(months=step_months)

    return folds


def count_wfo_folds(trading_dates, train_months=24, val_months=1,
                    test_months=1, step_months=1, embargo_days=5):
    folds = generate_wfo_folds(trading_dates, train_months, val_months,
                                test_months, step_months, embargo_days)
    if not folds:
        return {"n_folds": 0}
    return {
        "n_folds": len(folds),
        "first_fold": folds[0],
        "last_fold": folds[-1],
        "total_test_period": (folds[0]["test_start"], folds[-1]["test_end"]),
    }


# =============================================================================
# FOLD VISUALIZATION (Gantt-style)
# =============================================================================

def plot_wfo_folds(folds: List[Dict]):
    """Plot WFO folds as horizontal Gantt bars with concatenated test bar. Returns plotly figure."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping fold plot")
        return None

    fig = go.Figure()
    n = len(folds)

    # Add invisible trace to establish datetime x-axis
    all_dates = []
    for fold in folds:
        all_dates.extend([fold["train_start"], fold["test_end"]])
    fig.add_trace(go.Scatter(
        x=[min(all_dates), max(all_dates)], y=[-1, n + 1],
        mode="markers", marker=dict(size=0, opacity=0),
        showlegend=False, hoverinfo="skip"))

    for i, fold in enumerate(folds):
        y = n - i
        for phase, color, key_s, key_e in [
            ("Train", "rgba(0,128,0,0.7)", "train_start", "train_end"),
            ("Val", "rgba(255,255,0,0.7)", "val_start", "val_end"),
            ("Test", "rgba(255,0,0,0.7)", "test_start", "test_end"),
        ]:
            fig.add_shape(type="rect",
                          x0=fold[key_s], x1=fold[key_e],
                          y0=y - 0.4, y1=y + 0.4, fillcolor=color,
                          line=dict(color="black", width=0.5))
            # Hover trace
            fig.add_trace(go.Scatter(
                x=[fold[key_s]], y=[y], mode="markers",
                marker=dict(size=0, opacity=0), showlegend=False,
                hovertemplate=f"<b>Fold {fold['fold_id']} — {phase}</b><br>"
                              f"{fold[key_s]} → {fold[key_e]}<extra></extra>"))

    # Concatenated test bar at bottom (y=0)
    if folds:
        fig.add_shape(type="rect",
                      x0=folds[0]["test_start"], x1=folds[-1]["test_end"],
                      y0=-0.4, y1=0.4,
                      fillcolor="rgba(139,0,0,0.8)",
                      line=dict(color="white", width=0.5))
        fig.add_trace(go.Scatter(
            x=[folds[0]["test_start"]], y=[0], mode="markers",
            marker=dict(size=0, opacity=0), showlegend=False,
            hovertemplate=f"<b>Full OOS Test Period</b><br>"
                          f"{folds[0]['test_start']} → {folds[-1]['test_end']}<extra></extra>"))

    fig.update_layout(
        title=f"Walk-Forward Folds ({n} folds, sliding window)",
        xaxis_title="Date", yaxis_title="Fold",
        height=max(400, n * 30 + 100), template="plotly_white",
        yaxis=dict(tickmode="linear", tick0=0, dtick=1,
                   range=[-1, n + 1]),
        xaxis=dict(type="date"),
    )
    fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
                       text="<b>Legend:</b><br>🟢 Train<br>🟡 Val<br>🔴 Test<br>🟤 Full OOS",
                       showarrow=False, align="right",
                       bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    return fig


# =============================================================================
# EVALUATE AGENT
# =============================================================================

def evaluate_agent(agent, dataset, start_date, end_date,
                   transaction_cost_bps=5.0, lookback_window=20):
    env = PortfolioEnv(
        dataset, start_date=start_date, end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
        turnover_penalty=0.0, reward_type="return",
        lookback_window=lookback_window,
    )
    state = env.reset()
    while not env.done:
        action = agent.select_action(state, deterministic=True)
        state, _, _, _ = env.step(action)

    results = env.get_results()
    equity = np.array([1.0] + list(results["portfolio_value"].values))
    metrics = compute_all_metrics(equity)
    metrics["Avg Daily Turnover (%)"] = round(results["turnover"].mean() * 100, 4)
    metrics["Avg Cash (%)"] = round(results["cash_weight"].mean() * 100, 2) if "cash_weight" in results else 0
    # Trade counting: a trade = any day where weights change meaningfully
    turnover_series = results["turnover"]
    metrics["N Trades"] = int((turnover_series > 0.01).sum())  # days with >1% turnover
    metrics["Total TC (%)"] = round(results["transaction_cost"].sum() * 100, 4)
    if "equity_fraction" in results:
        metrics["Avg Equity (%)"] = round(results["equity_fraction"].mean() * 100, 2)
    if "rf_earned" in results:
        metrics["Total RF Earned (%)"] = round(results["rf_earned"].sum() * 100, 4)
    return {"results": results, "metrics": metrics, "equity": equity}


# =============================================================================
# TRAIN AGENT (single fold)
# =============================================================================

def train_agent(agent, dataset, train_start, train_end, val_start, val_end,
                n_epochs=40, patience=7, min_epochs=15,
                transaction_cost_bps=5.0, turnover_penalty=0.001,
                variance_penalty=0.0, tc_curriculum_frac=0.0,
                lookback_window=20, verbose=True):
    train_env = PortfolioEnv(
        dataset, start_date=train_start, end_date=train_end,
        transaction_cost_bps=transaction_cost_bps,
        turnover_penalty=turnover_penalty, reward_type="sharpe",
        lookback_window=lookback_window,
        variance_penalty=variance_penalty,
        tc_curriculum_frac=tc_curriculum_frac,
    )

    best_val_score = -np.inf
    best_val_ir2 = 0.0
    best_val_sharpe = -np.inf
    best_epoch = 0
    best_state_bytes = None
    patience_counter = 0

    update_every = 4
    for epoch in range(n_epochs):
        t0 = time.time()
        state = train_env.reset()
        step_count = 0
        while not train_env.done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = train_env.step(action)
            agent.store_transition(state, action, reward, next_state, done,
                                   info["n_tradable"])
            step_count += 1
            if step_count % update_every == 0:
                for _ in range(agent.config["gradient_steps"]):
                    agent.update()
            state = next_state

        # Train metrics
        train_results = train_env.get_results()
        train_eq = np.array([1.0] + list(train_results["portfolio_value"].values))
        train_m = compute_all_metrics(train_eq)

        # Validate
        val_r = evaluate_agent(agent, dataset, val_start, val_end,
                               transaction_cost_bps, lookback_window)
        val_ir2 = val_r["metrics"]["IR2"]
        val_arc = val_r["metrics"]["ARC (%)"]
        val_rets = val_r["results"]["portfolio_return_net"]
        val_std = val_rets.std()
        val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(252), -10.0, 10.0)) if val_std > 1e-4 else 0.0

        elapsed = time.time() - t0
        if verbose:
            print(f"    Ep {epoch:2d} | Train IR2: {train_m['IR2']:.4f} "
                  f"Val Sharpe: {val_sharpe:.3f} | Val ARC: {val_arc:+.1f}% | "
                  f"α: {agent.alpha.item():.3f} | {elapsed:.1f}s")

        # Early stopping on Val Sharpe (more stable than IR2 on 1-month windows)
        score = val_sharpe
        if score > best_val_score:
            best_val_score = score
            best_val_ir2 = val_ir2
            best_val_sharpe = val_sharpe
            best_epoch = epoch
            buf = io.BytesIO()
            torch.save({"actor": agent.actor.state_dict(),
                         "critic": agent.critic.state_dict(),
                         "critic_target": agent.critic_target.state_dict()}, buf)
            best_state_bytes = buf.getvalue()
            patience_counter = 0
        else:
            if epoch >= min_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stop at epoch {epoch} (best: {best_epoch})")
                    break

    if best_state_bytes:
        ckpt = torch.load(io.BytesIO(best_state_bytes), map_location=agent.device, weights_only=False)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])

    return {"best_val_ir2": best_val_ir2, "best_val_sharpe": best_val_sharpe, "best_epoch": best_epoch}


# =============================================================================
# HP CONFIGS (10 configs)
# =============================================================================

DEFAULT_HP_CONFIGS = [
    {"name": "standard",
     "lr_actor": 3e-4, "lr_critic": 3e-4, "lstm_hidden": 64, "n_attn_heads": 4,
     "scorer_hidden": 256, "cash_head_hidden": 64, "hierarchical": True,
     "min_equity": 0.0, "max_equity": 1.0, "variance_penalty": 0.0},
    {"name": "conservative_lr",
     "lr_actor": 1e-4, "lr_critic": 3e-4, "lstm_hidden": 64, "n_attn_heads": 4,
     "scorer_hidden": 256, "cash_head_hidden": 64, "hierarchical": True,
     "min_equity": 0.0, "max_equity": 1.0, "variance_penalty": 0.0},
    {"name": "large_capacity",
     "lr_actor": 3e-4, "lr_critic": 3e-4, "lstm_hidden": 128, "n_attn_heads": 8,
     "scorer_hidden": 256, "cash_head_hidden": 128, "hierarchical": True,
     "min_equity": 0.0, "max_equity": 1.0, "variance_penalty": 0.0},
    {"name": "aggressive_timing",
     "lr_actor": 3e-4, "lr_critic": 3e-4, "lstm_hidden": 64, "n_attn_heads": 4,
     "scorer_hidden": 256, "cash_head_hidden": 128, "hierarchical": True,
     "min_equity": 0.0, "max_equity": 1.0, "variance_penalty": 0.0},
]


def _compute_monthly_sharpes(returns_series, annualization=252):
    """
    Split a return series into calendar-month chunks and compute
    annualized Sharpe for each month. Returns list of monthly Sharpes.
    Sharpe is clamped to [-10, 10] to avoid blow-up from low-vol months.
    """
    if returns_series.empty:
        return []
    monthly_groups = returns_series.groupby(returns_series.index.to_period("M"))
    sharpes = []
    for _, month_rets in monthly_groups:
        if len(month_rets) < 5:
            continue
        mu = month_rets.mean()
        sigma = month_rets.std()
        if sigma > 1e-4:
            s = mu / sigma * np.sqrt(annualization)
            sharpes.append(float(np.clip(s, -10.0, 10.0)))
        else:
            sharpes.append(0.0)
    return sharpes


def select_hyperparameters(dataset, fold, hp_configs, n_epochs=25,
                           patience=7, min_epochs=12, transaction_cost_bps=5.0,
                           turnover_penalty=0.001, lookback_window=20,
                           variance_penalty=0.0, tc_curriculum_frac=0.0,
                           verbose=True):
    """
    Run all HP configs on a fold, select best using monthly-Sharpe consistency.

    Selection (3-tier):
      Tier 1: median(train monthly Sharpes) > 2 AND max(val monthly Sharpes) > 2
              → pick config with smallest |median_train - max_val| (consistency)
      Tier 2: both median_train > 0 and max_val > 0
              → pick config with highest max_val
      Tier 3: fallback → pick config with highest max_val
    """
    fold_id = fold.get("fold_id", "?")
    print(f"\n  HP Selection on fold {fold_id}:")
    print(f"    Train: {fold['train_start']} → {fold['train_end']} ({fold['n_train']}d)")
    print(f"    Val:   {fold['val_start']} → {fold['val_end']} ({fold['n_val']}d)")

    candidates = []
    trained_agents = {}

    for hp in hp_configs:
        hp_copy = hp.copy()
        hp_name = hp_copy["name"]
        print(f"\n    --- Config: {hp_name} ---")
        vp = hp_copy.pop("variance_penalty", variance_penalty)
        config = {
            "n_asset_features": dataset["metadata"]["n_per_asset_features"],
            "n_global_features": dataset["metadata"]["n_global_features"],
            **{k: v for k, v in hp_copy.items() if k != "name"},
        }
        agent = SACAgent(config)
        result = train_agent(
            agent, dataset,
            fold["train_start"], fold["train_end"],
            fold["val_start"], fold["val_end"],
            n_epochs=n_epochs, patience=patience, min_epochs=min_epochs,
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty=turnover_penalty,
            variance_penalty=vp, tc_curriculum_frac=tc_curriculum_frac,
            lookback_window=lookback_window, verbose=verbose,
        )

        # Evaluate trained agent on full train and val windows
        train_r = evaluate_agent(agent, dataset, fold["train_start"],
                                 fold["train_end"], transaction_cost_bps,
                                 lookback_window)
        val_r = evaluate_agent(agent, dataset, fold["val_start"],
                               fold["val_end"], transaction_cost_bps,
                               lookback_window)

        train_monthly = _compute_monthly_sharpes(train_r["results"]["portfolio_return_net"])
        val_monthly = _compute_monthly_sharpes(val_r["results"]["portfolio_return_net"])

        median_train = float(np.median(train_monthly)) if train_monthly else -np.inf
        max_val = float(np.max(val_monthly)) if val_monthly else -np.inf

        val_ir2 = val_r["metrics"]["IR2"]
        val_rets = val_r["results"]["portfolio_return_net"]
        val_std = val_rets.std()
        val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(252), -10.0, 10.0)) if val_std > 1e-4 else 0.0

        entry = {
            "name": hp_name, "config": hp, "val_ir2": val_ir2,
            "val_sharpe": val_sharpe, "variance_penalty": vp,
            "median_train_sharpe": median_train, "max_val_sharpe": max_val,
            "n_train_months": len(train_monthly), "n_val_months": len(val_monthly),
        }
        candidates.append(entry)
        trained_agents[hp_name] = agent

        print(f"    → Med-Train Sharpe: {median_train:.3f} ({len(train_monthly)} months) | "
              f"Max-Val Sharpe: {max_val:.3f} ({len(val_monthly)} months)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 3-tier selection ===
    tier1 = [c for c in candidates
             if c["median_train_sharpe"] > 2.0 and c["max_val_sharpe"] > 2.0]
    if tier1:
        best = min(tier1, key=lambda c: abs(c["median_train_sharpe"] - c["max_val_sharpe"]))
        tier_label = "Tier-1 (both > 2, closest gap)"
    else:
        tier2 = [c for c in candidates
                 if c["median_train_sharpe"] > 0 and c["max_val_sharpe"] > 0]
        if tier2:
            best = max(tier2, key=lambda c: c["max_val_sharpe"])
            tier_label = "Tier-2 (both positive, best max-val)"
        else:
            best = max(candidates, key=lambda c: c["max_val_sharpe"])
            tier_label = "Tier-3 (fallback, best max-val)"

    best_agent = trained_agents[best["name"]]
    for name, ag in trained_agents.items():
        if name != best["name"]:
            del ag

    print(f"\n  ★ Selected: '{best['name']}' [{tier_label}]")
    print(f"    Med-Train: {best['median_train_sharpe']:.3f} | Max-Val: {best['max_val_sharpe']:.3f} | "
          f"Val Sharpe: {best['val_sharpe']:.4f} | IR2: {best['val_ir2']:.4f}")

    return best, candidates, best_agent


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def _save_checkpoint(out_dir, fold_id, agent, fold_log, all_test_returns,
                     all_test_qqq_returns, val_sharpe_history, selected_config):
    ckpt = {
        "fold_id": fold_id,
        "fold_log": fold_log,
        "val_sharpe_history": val_sharpe_history,
        "selected_config": selected_config,
        "n_test_returns": len(all_test_returns),
    }
    with open(out_dir / "wfo_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2, default=str)
    agent.save(str(out_dir / "agent_checkpoint.pt"))


def _load_checkpoint(out_dir):
    ckpt_path = out_dir / "wfo_checkpoint.json"
    if not ckpt_path.exists():
        return None
    with open(ckpt_path) as f:
        return json.load(f)


# =============================================================================
# MAIN: WALK-FORWARD TRAINING
# =============================================================================

def train_walk_forward(
    dataset: dict,
    train_months: int = 24,
    val_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    embargo_days: int = 5,
    hp_configs: Optional[List[Dict]] = None,
    n_epochs: int = 40,
    patience: int = 7,
    min_epochs: int = 15,
    transaction_cost_bps: float = 5.0,
    turnover_penalty: float = 0.001,
    variance_penalty: float = 0.0,
    tc_curriculum_frac: float = 0.0,
    lookback_window: int = 20,
    results_dir: str = "../Results",
    verbose: bool = True,
) -> Dict:
    if hp_configs is None:
        hp_configs = [c.copy() for c in DEFAULT_HP_CONFIGS]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate folds
    folds = generate_wfo_folds(
        dataset["trading_dates"], train_months, val_months,
        test_months, step_months, embargo_days,
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD RL TRAINING (SLIDING WINDOW)")
    print("=" * 70)
    print(f"  WFO: train={train_months}m (sliding), val={val_months}m, "
          f"test={test_months}m, step={step_months}m, embargo={embargo_days}d")
    print(f"  Folds: {len(folds)}")
    if folds:
        print(f"  OOS test: {folds[0]['test_start']} → {folds[-1]['test_end']}")
    print(f"  HP configs: {len(hp_configs)}")
    print(f"  Lookback window: {lookback_window} days")
    print(f"  Variance penalty: {variance_penalty}")
    print(f"  TC curriculum: {tc_curriculum_frac*100:.0f}% of episode")
    print(f"  Policy mode: {'Hierarchical (HRA-SAC)' if hp_configs[0].get('hierarchical', True) else 'Flat Dirichlet'}")
    print("=" * 70)

    if not folds:
        print("  ERROR: No valid folds.")
        return {}

    # --- Plot folds ---
    fig = plot_wfo_folds(folds)
    if fig:
        fig.write_html(str(out_dir / "wfo_folds_plot.html"))
        print(f"  Fold plot saved to {out_dir / 'wfo_folds_plot.html'}")

    # --- Check for checkpoint (resume) ---
    ckpt = _load_checkpoint(out_dir)
    start_fold = 0
    all_test_returns = []
    all_test_qqq_returns = []
    all_test_turnover = []
    fold_log = []
    val_sharpe_history = []
    selected_config = None
    agent = None

    if ckpt:
        start_fold = ckpt["fold_id"]
        fold_log = ckpt["fold_log"]
        val_sharpe_history = ckpt.get("val_sharpe_history", [])
        selected_config = ckpt["selected_config"]
        for i in range(ckpt["n_test_returns"]):
            p = out_dir / f"rl_fold_{i+1}_test_returns.csv"
            if p.exists():
                s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
                all_test_returns.append(s)
            p2 = out_dir / f"rl_fold_{i+1}_qqq_returns.csv"
            if p2.exists():
                all_test_qqq_returns.append(pd.read_csv(p2, index_col=0, parse_dates=True).iloc[:, 0])
        # Rebuild agent from checkpoint
        agent_config = {
            "n_asset_features": dataset["metadata"]["n_per_asset_features"],
            "n_global_features": dataset["metadata"]["n_global_features"],
            **{k: v for k, v in selected_config.items() if k not in ("name", "variance_penalty")},
        }
        agent = SACAgent(agent_config)
        agent_ckpt_path = out_dir / "agent_checkpoint.pt"
        if agent_ckpt_path.exists():
            agent.load(str(agent_ckpt_path))
        print(f"\n  *** RESUMING from fold {start_fold + 1} (completed {start_fold} folds) ***\n")

    # --- Walk-forward loop ---
    print(f"\n{'='*70}")
    print(f"WALKING FORWARD — {len(folds)} folds")
    print(f"{'='*70}")

    n_retrains = sum(1 for f in fold_log if f.get("retrained", False))

    for i, fold in enumerate(folds):
        if i < start_fold:
            continue

        fid = fold["fold_id"]

        # --- Retrain decision ---
        need_retrain = False
        current_val_sharpe = 0.0
        current_val_ir2 = 0.0

        if i == 0 and not ckpt:
            need_retrain = True
            reason = "initial"
        else:
            val_r = evaluate_agent(agent, dataset, fold["val_start"], fold["val_end"],
                                   transaction_cost_bps, lookback_window)
            current_val_ir2 = val_r["metrics"]["IR2"]
            val_rets = val_r["results"]["portfolio_return_net"]
            val_std = val_rets.std()
            current_val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(252), -10.0, 10.0)) if val_std > 1e-4 else 0.0

            # Mandatory retrain every 4 folds to prevent stale models
            folds_since_retrain = 0
            for fl in reversed(fold_log):
                if fl.get("retrained", False):
                    break
                folds_since_retrain += 1
            if folds_since_retrain >= 3:
                need_retrain = True
                reason = f"mandatory (>{folds_since_retrain} folds since retrain)"
            elif current_val_sharpe < 0:
                need_retrain = True
                reason = f"Sharpe {current_val_sharpe:.3f} < 0"
            elif len(val_sharpe_history) >= 3:
                recent = val_sharpe_history[-5:]
                med = np.median(recent)
                std = np.std(recent) if len(recent) > 1 else 0.0
                threshold = med - 0.5 * std
                if current_val_sharpe < threshold:
                    need_retrain = True
                    reason = f"Sharpe {current_val_sharpe:.3f} < {threshold:.3f} (med={med:.3f} - 0.5*std={std:.3f})"

        if need_retrain:
            if verbose:
                print(f"\n  Fold {fid:2d}/{len(folds)} | RETRAIN ({reason})")

            best_hp, _, agent = select_hyperparameters(
                dataset, fold, hp_configs,
                n_epochs=min(n_epochs, 25), patience=patience,
                min_epochs=min(min_epochs, 12),
                transaction_cost_bps=transaction_cost_bps,
                turnover_penalty=turnover_penalty,
                lookback_window=lookback_window,
                variance_penalty=variance_penalty,
                tc_curriculum_frac=tc_curriculum_frac,
                verbose=verbose,
            )
            selected_config = best_hp["config"]
            current_val_ir2 = best_hp["val_ir2"]

            val_r_post = evaluate_agent(agent, dataset, fold["val_start"], fold["val_end"],
                                        transaction_cost_bps, lookback_window)
            post_rets = val_r_post["results"]["portfolio_return_net"]
            post_std = post_rets.std()
            current_val_sharpe = float(np.clip(post_rets.mean() / post_std * np.sqrt(252), -10.0, 10.0)) if post_std > 1e-4 else 0.0
            n_retrains += 1
        else:
            if verbose:
                print(f"  Fold {fid:2d}/{len(folds)} | CARRY (Sharpe: {current_val_sharpe:.3f})", end="")

        val_sharpe_history.append(current_val_sharpe)

        # Test
        test_r = evaluate_agent(agent, dataset, fold["test_start"], fold["test_end"],
                                transaction_cost_bps, lookback_window)
        test_ir2 = test_r["metrics"]["IR2"]
        test_arc = test_r["metrics"]["ARC (%)"]

        # QQQ buy & hold for this test window
        qqq_rets = test_r["results"]["qqq_return"]
        qqq_eq = np.array([1.0] + list((1 + qqq_rets).cumprod().values))
        qqq_test_m = compute_all_metrics(qqq_eq)
        qqq_test_arc = qqq_test_m["ARC (%)"]

        if verbose and not need_retrain:
            print(f" → RL ARC: {test_arc:+.1f}% | QQQ ARC: {qqq_test_arc:+.1f}%")
        elif verbose:
            print(f"    Test: {fold['test_start']}→{fold['test_end']} → "
                  f"RL ARC: {test_arc:+.1f}% | QQQ ARC: {qqq_test_arc:+.1f}%")

        # Collect test returns
        all_test_returns.append(test_r["results"]["portfolio_return_net"])
        all_test_qqq_returns.append(test_r["results"]["qqq_return"])
        if "turnover" in test_r["results"]:
            all_test_turnover.append(test_r["results"]["turnover"])

        # Save per-fold returns
        test_r["results"]["portfolio_return_net"].to_csv(
            out_dir / f"rl_fold_{fid}_test_returns.csv")
        test_r["results"]["qqq_return"].to_csv(
            out_dir / f"rl_fold_{fid}_qqq_returns.csv")

        fold_log.append({
            "fold_id": fid,
            "train_start": fold["train_start"], "train_end": fold["train_end"],
            "val_start": fold["val_start"], "val_end": fold["val_end"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "n_train": fold["n_train"], "n_test": fold["n_test"],
            "retrained": need_retrain,
            "selected_config": selected_config.get("name", "unknown") if selected_config else "unknown",
            "val_ir2": round(current_val_ir2, 4),
            "val_sharpe": round(current_val_sharpe, 4),
            "test_ir2": round(test_ir2, 4),
            "test_arc": round(test_arc, 2),
            "qqq_test_arc": round(qqq_test_arc, 2),
            "test_mdd": round(test_r["metrics"]["Max Drawdown (%)"], 2),
            "test_sharpe": round(test_r["metrics"].get("Sharpe", 0), 4),
            "test_sortino": round(test_r["metrics"].get("Sortino", 0), 4),
            "test_n_trades": test_r["metrics"].get("N Trades", 0),
            "test_total_tc": round(test_r["metrics"].get("Total TC (%)", 0), 4),
            "test_cash": round(test_r["metrics"].get("Avg Cash (%)", 0), 2),
        })

        # Checkpoint
        _save_checkpoint(out_dir, i + 1, agent, fold_log,
                         all_test_returns, all_test_qqq_returns,
                         val_sharpe_history, selected_config)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === STITCH ===
    print(f"\n{'='*70}")
    print("STITCHING TEST RETURNS")
    print(f"{'='*70}")

    stitched_rl = pd.concat(all_test_returns)
    stitched_qqq = pd.concat(all_test_qqq_returns)
    stitched_rl = stitched_rl[~stitched_rl.index.duplicated(keep="first")].sort_index()
    stitched_qqq = stitched_qqq[~stitched_qqq.index.duplicated(keep="first")].sort_index()

    rl_equity = (1 + stitched_rl).cumprod()
    qqq_equity = (1 + stitched_qqq).cumprod()

    rl_eq_arr = np.array([1.0] + list(rl_equity.values))
    qqq_eq_arr = np.array([1.0] + list(qqq_equity.values))
    rl_m = compute_all_metrics(rl_eq_arr)
    qqq_m = compute_all_metrics(qqq_eq_arr)
    if all_test_turnover:
        stitched_to = pd.concat(all_test_turnover)
        rl_m["Avg Daily Turnover (%)"] = round(stitched_to.mean() * 100, 4)

    print(f"\n  OOS: {stitched_rl.index[0].date()} → {stitched_rl.index[-1].date()}")
    print(f"  Days: {len(stitched_rl)}, Retrains: {n_retrains}/{len(folds)}")
    print(f"\n  {'METRIC':<25} {'RL':>12} {'QQQ':>12}")
    print(f"  {'-'*51}")
    # All metrics from compute_all_metrics (baseline.py)
    metric_keys = [
        "Absolute Return (%)", "ARC (%)", "ASD (%)", "Max Drawdown (%)",
        "MLD (years)", "IR1", "IR2", "Sharpe", "Sortino", "Calmar", "N Days",
    ]
    for k in metric_keys:
        rv = rl_m.get(k, "N/A")
        qv = qqq_m.get(k, "N/A")
        if isinstance(rv, (int, float)) and isinstance(qv, (int, float)):
            print(f"  {k:<25} {rv:>12.4f} {qv:>12.4f}")
        else:
            print(f"  {k:<25} {str(rv):>12} {str(qv):>12}")
    if "Avg Daily Turnover (%)" in rl_m:
        print(f"  {'Avg Daily Turnover (%)':<25} {rl_m['Avg Daily Turnover (%)']:>12.4f} {'N/A':>12}")

    # Save
    equity_df = pd.DataFrame({"RL Agent": rl_equity, "QQQ": qqq_equity})
    equity_df.to_csv(out_dir / "rl_equity_oos.csv")
    pd.DataFrame({"RL Agent": stitched_rl, "QQQ": stitched_qqq}).to_csv(
        out_dir / "rl_daily_returns_oos.csv")
    pd.DataFrame({"RL Agent": rl_m, "QQQ": qqq_m}).T.to_csv(
        out_dir / "rl_performance_metrics.csv")
    pd.DataFrame(fold_log).to_csv(out_dir / "rl_fold_log.csv", index=False)
    agent.save(str(out_dir / "agent_final.pt"))

    wfo_cfg = {"train_months": train_months, "val_months": val_months,
               "test_months": test_months, "step_months": step_months,
               "embargo_days": embargo_days, "n_folds": len(folds),
               "n_retrains": n_retrains, "lookback_window": lookback_window,
               "variance_penalty": variance_penalty,
               "tc_curriculum_frac": tc_curriculum_frac,
               "hp_configs": len(hp_configs),
               "window_type": "SLIDING (non-anchored)",
               "hierarchical": hp_configs[0].get("hierarchical", True)}
    with open(out_dir / "rl_wfo_config.json", "w") as f:
        json.dump(wfo_cfg, f, indent=2)

    print(f"\n  All saved to {out_dir}")
    return {"rl_equity": rl_equity, "qqq_equity": qqq_equity,
            "rl_oos_metrics": rl_m, "fold_log": pd.DataFrame(fold_log),
            "agent": agent}
