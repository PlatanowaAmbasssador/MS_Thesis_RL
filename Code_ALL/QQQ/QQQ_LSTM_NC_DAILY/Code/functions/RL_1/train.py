"""
train.py — Walk-Forward Optimization for Portfolio Agent (Daily Run)
======================================================================
Daily pipeline with LSTM HP variation:
    - 6 configs varying: lstm_hidden, lstm_layers, dropout, batch_size, LR
    - 80 epochs, min_epochs=25, patience=8, gradient_steps=2
    - Force retrain every fold, non-learning detection
    - 5yr train / 6mo val / 6mo test / 6mo step (annualization=252)
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
                   transaction_cost_bps=5.0, lookback_window=40,
                   top_k=0, annualization=504, allow_short=False, force_fully_invested=False):
    env = PortfolioEnv(
        dataset, start_date=start_date, end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
        turnover_penalty=0.0, reward_type="return",
        lookback_window=lookback_window, top_k=top_k,
        allow_short=allow_short, force_fully_invested=force_fully_invested,
    )
    state = env.reset()
    while not env.done:
        action = agent.select_action(state, deterministic=True)
        state, _, _, _ = env.step(action)

    results = env.get_results()
    equity = np.array([1.0] + list(results["portfolio_value"].values))
    metrics = compute_all_metrics(equity, annualization=annualization)
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
                n_epochs=30, patience=5, min_epochs=10,
                transaction_cost_bps=5.0, turnover_penalty=0.003,
                variance_penalty=0.0, tc_curriculum_frac=0.0,
                lookback_window=40, verbose=True,
                top_k=0, annualization=504, reward_type="excess_return",
                allow_short=False, force_fully_invested=False):
    train_env = PortfolioEnv(
        dataset, start_date=train_start, end_date=train_end,
        transaction_cost_bps=transaction_cost_bps,
        turnover_penalty=turnover_penalty, reward_type=reward_type,
        lookback_window=lookback_window,
        variance_penalty=variance_penalty,
        tc_curriculum_frac=tc_curriculum_frac,
        top_k=top_k, allow_short=allow_short, force_fully_invested=force_fully_invested,
    )

    best_val_score = -np.inf
    best_val_ir2 = 0.0
    best_val_sharpe = -np.inf
    best_epoch = 0
    best_state_bytes = None
    patience_counter = 0
    max_train_ir2 = 0.0  # Track if agent ever learns

    update_every = 4
    for epoch in range(n_epochs):
        t0 = time.time()
        state = train_env.reset()
        step_count = 0
        epoch_update_info = {}
        while not train_env.done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = train_env.step(action)
            agent.store_transition(state, action, reward, next_state, done,
                                   state["n_tradable"])
            step_count += 1
            if step_count % update_every == 0:
                for _ in range(agent.config["gradient_steps"]):
                    epoch_update_info = agent.update()
            state = next_state

        # Train metrics
        train_results = train_env.get_results()
        train_eq = np.array([1.0] + list(train_results["portfolio_value"].values))
        train_m = compute_all_metrics(train_eq, annualization=annualization)
        train_ir2 = train_m["IR2"]
        max_train_ir2 = max(max_train_ir2, train_ir2)

        # Train absolute return and cash
        train_abs_ret = train_m.get("Absolute Return (%)", 0)
        train_cash = train_results["cash_weight"].mean() * 100 if "cash_weight" in train_results else 0
        train_turnover = train_results["turnover"].mean() * 100 if "turnover" in train_results else 0

        # Validate
        val_r = evaluate_agent(agent, dataset, val_start, val_end,
                               transaction_cost_bps, lookback_window,
                               top_k=top_k, annualization=annualization,
                               allow_short=allow_short, force_fully_invested=force_fully_invested)
        val_ir2 = val_r["metrics"]["IR2"]
        val_arc = val_r["metrics"]["ARC (%)"]
        val_abs_ret = val_r["metrics"].get("Absolute Return (%)", 0)
        val_rets = val_r["results"]["portfolio_return_net"]
        val_std = val_rets.std()
        val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(annualization), -10.0, 10.0)) if val_std > 1e-4 else 0.0

        elapsed = time.time() - t0
        if verbose:
            # Enhanced logging (Run 11)
            grad_info = ""
            if epoch_update_info:
                ag = epoch_update_info.get("actor_grad_norm", 0)
                cg = epoch_update_info.get("critic_grad_norm", 0)
                grad_info = f" | ag={ag:.2f} cg={cg:.2f}"
            print(f"    Ep {epoch:2d} | Train IR2: {train_m['IR2']:.4f} AbsR: {train_abs_ret:+.1f}% "
                  f"Cash: {train_cash:.0f}% TO: {train_turnover:.1f}% | "
                  f"Val Sh: {val_sharpe:.3f} ARC: {val_arc:+.1f}% AbsR: {val_abs_ret:+.1f}% | "
                  f"α: {agent.alpha.item():.3f}{grad_info} | {elapsed:.1f}s")

        # Early stopping on Val Sharpe
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

    return {
        "best_val_ir2": best_val_ir2, "best_val_sharpe": best_val_sharpe,
        "best_epoch": best_epoch, "max_train_ir2": max_train_ir2,
        "learned": max_train_ir2 > 0.01,  # Flag for non-learning detection
    }


# =============================================================================
# HP CONFIGS — 6 configs: LSTM architecture variation (Daily Run)
# =============================================================================
# HP CONFIGS — FINAL: Benchmark-relative, no cash drag, no HHI penalty
# =============================================================================
# Changes that address the structural gap to QQQ:
# 1. variance_penalty=0.0 → agent CAN concentrate into mega-caps
# 2. force_fully_invested=True → no cash drag (was costing 1% ARC/year)
# 3. reward_type="benchmark_relative" → directly optimizes vs QQQ
# 4. top_k=20 → proven to work best across all runs

_SHARED_FINAL = {
    "lr_critic": 5e-4, "lr_alpha": 3e-4,
    "n_attn_heads": 4,
    "scorer_hidden": 128, "cash_head_hidden": 64, "critic_hidden": 256,
    "gamma": 0.99,
    "auto_alpha": False, "alpha_init": 0.2,
    "warmup_steps": 500,
    "variance_penalty": 0.0, "hierarchical": False,
    "weight_smooth_beta": 0.3,
    "buffer_capacity": 20000,
    "force_fully_invested": True,
}

DEFAULT_HP_CONFIGS = [
    # A) No penalty, fully invested, k20 (main experiment)
    {"name": "bmk_nopen_k20",
     "lstm_hidden": 64, "lstm_layers": 1, "dropout": 0.0,
     "lr_actor": 3e-4, "batch_size": 128, "top_k": 20,
     **_SHARED_FINAL},

    # B) Same but k10 (more concentrated = closer to QQQ)
    {"name": "bmk_nopen_k10",
     "lstm_hidden": 64, "lstm_layers": 1, "dropout": 0.0,
     "lr_actor": 3e-4, "batch_size": 128, "top_k": 10,
     **_SHARED_FINAL},

    # C) Medium LSTM, k20
    {"name": "bmk_nopen_med_k20",
     "lstm_hidden": 128, "lstm_layers": 1, "dropout": 0.1,
     "lr_actor": 3e-4, "batch_size": 128, "top_k": 20,
     **_SHARED_FINAL},
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
                           patience=5, min_epochs=10, transaction_cost_bps=5.0,
                           turnover_penalty=0.003, lookback_window=40,
                           variance_penalty=0.0, tc_curriculum_frac=0.0,
                           verbose=True, annualization=504,
                           reward_type="excess_return"):
    """
    Run all HP configs on a fold, select best using monthly-Sharpe consistency.
    Per-config top_k, batch_size, gamma, ent_multiplier, dropout are extracted
    from each HP config dict and passed to SACAgent and environment.
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
        # Extract per-config top_k (used by environment, not SACAgent)
        config_top_k = hp_copy.pop("top_k", 20)
        config_allow_short = hp_copy.pop("allow_short", False)
        config_force_fi = hp_copy.pop("force_fully_invested", False)
        # All remaining keys (except name) go to SACAgent config
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
            top_k=config_top_k, annualization=annualization,
            reward_type=reward_type,
            allow_short=config_allow_short, force_fully_invested=config_force_fi,
        )

        # Evaluate on full train and val windows
        train_r = evaluate_agent(agent, dataset, fold["train_start"],
                                 fold["train_end"], transaction_cost_bps,
                                 lookback_window, top_k=config_top_k,
                                 annualization=annualization,
                                 allow_short=config_allow_short, force_fully_invested=config_force_fi)
        val_r = evaluate_agent(agent, dataset, fold["val_start"],
                               fold["val_end"], transaction_cost_bps,
                               lookback_window, top_k=config_top_k,
                               annualization=annualization,
                               allow_short=config_allow_short, force_fully_invested=config_force_fi)

        train_monthly = _compute_monthly_sharpes(train_r["results"]["portfolio_return_net"], annualization)
        val_monthly = _compute_monthly_sharpes(val_r["results"]["portfolio_return_net"], annualization)

        median_train = float(np.median(train_monthly)) if train_monthly else -np.inf
        max_val = float(np.max(val_monthly)) if val_monthly else -np.inf

        val_ir2 = val_r["metrics"]["IR2"]
        val_rets = val_r["results"]["portfolio_return_net"]
        val_std = val_rets.std()
        val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(annualization), -10.0, 10.0)) if val_std > 1e-4 else 0.0

        entry = {
            "name": hp_name, "config": hp, "val_ir2": val_ir2,
            "val_sharpe": val_sharpe, "variance_penalty": vp,
            "median_train_sharpe": median_train, "max_val_sharpe": max_val,
            "n_train_months": len(train_monthly), "n_val_months": len(val_monthly),
            "learned": result.get("learned", False),
            "max_train_ir2": result.get("max_train_ir2", 0),
        }
        candidates.append(entry)
        trained_agents[hp_name] = agent

        learned_tag = "✓ LEARNED" if result.get("learned", False) else "✗ NO LEARNING"
        print(f"    → {learned_tag} | max Train IR2: {result.get('max_train_ir2', 0):.4f}")
        print(f"    → Med-Train Sharpe: {median_train:.3f} ({len(train_monthly)} months) | "
              f"Max-Val Sharpe: {max_val:.3f} ({len(val_monthly)} months)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === NON-LEARNING FILTER (Run 11) ===
    # Only consider configs where the agent actually learned (Train IR2 > 0.01)
    learned_candidates = [c for c in candidates if c["learned"]]
    if learned_candidates:
        active = learned_candidates
        n_learned = len(learned_candidates)
        n_failed = len(candidates) - n_learned
        if n_failed > 0:
            print(f"\n  ⚠ {n_failed}/{len(candidates)} configs failed to learn (Train IR2=0), excluded")
    else:
        # ALL configs failed to learn → fall back to all candidates with warning
        active = candidates
        print(f"\n  ⚠⚠ ALL {len(candidates)} configs failed to learn! Using best-of-bad.")

    # === 3-tier selection (on active candidates only) ===
    tier1 = [c for c in active
             if c["median_train_sharpe"] > 2.0 and c["max_val_sharpe"] > 2.0]
    if tier1:
        best = min(tier1, key=lambda c: abs(c["median_train_sharpe"] - c["max_val_sharpe"]))
        tier_label = "Tier-1 (both > 2, closest gap)"
    else:
        tier2 = [c for c in active
                 if c["median_train_sharpe"] > 0 and c["max_val_sharpe"] > 0]
        if tier2:
            best = max(tier2, key=lambda c: c["max_val_sharpe"])
            tier_label = "Tier-2 (both positive, best max-val)"
        else:
            best = max(active, key=lambda c: c["max_val_sharpe"])
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
    val_months: int = 2,
    test_months: int = 2,
    step_months: int = 2,
    embargo_days: int = 0,
    hp_configs: Optional[List[Dict]] = None,
    n_epochs: int = 30,
    patience: int = 5,
    min_epochs: int = 10,
    transaction_cost_bps: float = 5.0,
    turnover_penalty: float = 0.003,
    variance_penalty: float = 0.0,
    tc_curriculum_frac: float = 0.0,
    lookback_window: int = 40,
    results_dir: str = "../Results",
    verbose: bool = True,
    annualization: int = 504,
    reward_type: str = "excess_return",
) -> Dict:
    if hp_configs is None:
        hp_configs = [c.copy() for c in DEFAULT_HP_CONFIGS]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Lookback: {lookback_window} | Annualization: {annualization}")
    print(f"  Reward: {reward_type} | Turnover penalty: {turnover_penalty}")
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
            # Carry logic: evaluate current model on new val window
            carry_top_k = selected_config.get("top_k", 20) if selected_config else 20
            carry_allow_short = selected_config.get("allow_short", False) if selected_config else False
            carry_force_fi = selected_config.get("force_fully_invested", False) if selected_config else False
            val_r = evaluate_agent(agent, dataset, fold["val_start"], fold["val_end"],
                                   transaction_cost_bps, lookback_window,
                                   top_k=carry_top_k, annualization=annualization,
                                   allow_short=carry_allow_short, force_fully_invested=carry_force_fi)
            current_val_ir2 = val_r["metrics"]["IR2"]
            val_rets = val_r["results"]["portfolio_return_net"]
            val_std = val_rets.std()
            current_val_sharpe = float(np.clip(val_rets.mean() / val_std * np.sqrt(annualization), -10.0, 10.0)) if val_std > 1e-4 else 0.0

            # Retrain triggers:
            # 1) Mandatory every 4 folds
            folds_since_retrain = 0
            for fl in reversed(fold_log):
                if fl.get("retrained", False):
                    break
                folds_since_retrain += 1
            if folds_since_retrain >= 3:
                need_retrain = True
                reason = f"mandatory (>{folds_since_retrain} folds since retrain)"
            # 2) Val Sharpe negative
            elif current_val_sharpe < 0:
                need_retrain = True
                reason = f"Sharpe {current_val_sharpe:.3f} < 0"
            # 3) Sharpe degraded vs recent history
            elif len(val_sharpe_history) >= 3:
                recent = val_sharpe_history[-5:]
                med = np.median(recent)
                std = np.std(recent) if len(recent) > 1 else 0.0
                threshold = med - 0.5 * std
                if current_val_sharpe < threshold:
                    need_retrain = True
                    reason = f"Sharpe {current_val_sharpe:.3f} < {threshold:.3f}"

        if need_retrain:
            if verbose:
                print(f"\n  Fold {fid:2d}/{len(folds)} | RETRAIN ({reason})")

            best_hp, _, agent = select_hyperparameters(
                dataset, fold, hp_configs,
                n_epochs=n_epochs, patience=patience,
                min_epochs=min_epochs,
                transaction_cost_bps=transaction_cost_bps,
                turnover_penalty=turnover_penalty,
                lookback_window=lookback_window,
                variance_penalty=variance_penalty,
                tc_curriculum_frac=tc_curriculum_frac,
                verbose=verbose,
                annualization=annualization,
                reward_type=reward_type,
            )
            selected_config = best_hp["config"]
            current_val_ir2 = best_hp["val_ir2"]

            sel_top_k = selected_config.get("top_k", 20)
            sel_allow_short = selected_config.get("allow_short", False)
            sel_force_fi = selected_config.get("force_fully_invested", False)
            val_r_post = evaluate_agent(agent, dataset, fold["val_start"], fold["val_end"],
                                        transaction_cost_bps, lookback_window,
                                        top_k=sel_top_k, annualization=annualization,
                                        allow_short=sel_allow_short, force_fully_invested=sel_force_fi)
            post_rets = val_r_post["results"]["portfolio_return_net"]
            post_std = post_rets.std()
            current_val_sharpe = float(np.clip(post_rets.mean() / post_std * np.sqrt(annualization), -10.0, 10.0)) if post_std > 1e-4 else 0.0
            n_retrains += 1
        else:
            if verbose:
                print(f"  Fold {fid:2d}/{len(folds)} | CARRY (Sharpe: {current_val_sharpe:.3f})", end="")

        val_sharpe_history.append(current_val_sharpe)

        # Test — use top_k from selected config
        test_top_k = selected_config.get("top_k", 20) if selected_config else 20
        test_allow_short = selected_config.get("allow_short", False) if selected_config else False
        test_force_fi = selected_config.get("force_fully_invested", False) if selected_config else False
        test_r = evaluate_agent(agent, dataset, fold["test_start"], fold["test_end"],
                                transaction_cost_bps, lookback_window,
                                top_k=test_top_k, annualization=annualization,
                                allow_short=test_allow_short, force_fully_invested=test_force_fi)
        test_ir2 = test_r["metrics"]["IR2"]
        test_arc = test_r["metrics"]["ARC (%)"]
        test_abs_ret = test_r["metrics"].get("Absolute Return (%)", 0)

        # QQQ buy & hold for this test window
        qqq_rets = test_r["results"]["qqq_return"]
        qqq_eq = np.array([1.0] + list((1 + qqq_rets).cumprod().values))
        qqq_test_m = compute_all_metrics(qqq_eq, annualization=annualization)
        qqq_test_arc = qqq_test_m["ARC (%)"]

        if verbose and not need_retrain:
            print(f" → RL ARC: {test_arc:+.1f}% | QQQ ARC: {qqq_test_arc:+.1f}%")
        elif verbose:
            print(f"    Test: {fold['test_start']}→{fold['test_end']} → "
                  f"RL ARC: {test_arc:+.1f}% AbsR: {test_abs_ret:+.1f}% | "
                  f"QQQ ARC: {qqq_test_arc:+.1f}%")

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
    rl_m = compute_all_metrics(rl_eq_arr, annualization=annualization)
    qqq_m = compute_all_metrics(qqq_eq_arr, annualization=annualization)
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
               "hierarchical": hp_configs[0].get("hierarchical", True),
               "annualization": annualization, "reward_type": reward_type,
               "turnover_penalty": turnover_penalty}
    with open(out_dir / "rl_wfo_config.json", "w") as f:
        json.dump(wfo_cfg, f, indent=2)

    print(f"\n  All saved to {out_dir}")
    return {"rl_equity": rl_equity, "qqq_equity": qqq_equity,
            "rl_oos_metrics": rl_m, "fold_log": pd.DataFrame(fold_log),
            "agent": agent}
