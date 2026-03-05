"""
train.py — Walk-Forward Optimization for SAC Portfolio Agent (v2)
==================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Key changes from v1:
    - SLIDING (non-anchored) training window — fixed width, moves forward
    - Checkpoint-resume: saves state after each fold, auto-resumes on restart
    - 5 HP configs for fold-1 selection
    - 5-day embargo between train end and val start
    - Fine-tune on retrain (not fresh agent)
    - Dirichlet policy + lookback state integration
"""

import numpy as np
import pandas as pd
import time
import io
import json
import gc
from pathlib import Path
from typing import Optional, Dict, List

import sys, os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import torch
from environment import PortfolioEnv
from baseline import compute_all_metrics
from RL_1.sac_agent import SACAgent


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
    """Plot WFO folds as horizontal Gantt bars. Returns plotly figure."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping fold plot")
        return None

    fig = go.Figure()
    n = len(folds)
    for i, fold in enumerate(folds):
        y = n - i
        for phase, color, key_s, key_e in [
            ("Train", "rgba(0,128,0,0.7)", "train_start", "train_end"),
            ("Val", "rgba(255,255,0,0.7)", "val_start", "val_end"),
            ("Test", "rgba(255,0,0,0.7)", "test_start", "test_end"),
        ]:
            fig.add_shape(type="rect", x0=fold[key_s], x1=fold[key_e],
                          y0=y - 0.4, y1=y + 0.4, fillcolor=color,
                          line=dict(color="black", width=0.5))

    fig.update_layout(
        title=f"Walk-Forward Folds ({n} folds, sliding window)",
        xaxis_title="Date", yaxis_title="Fold",
        height=max(400, n * 35), template="plotly_white",
        yaxis=dict(tickmode="linear", tick0=1, dtick=1, range=[0, n + 1]),
        xaxis=dict(type="date"),
    )
    fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
                       text="<b>Legend:</b><br>🟢 Train<br>🟡 Val<br>🔴 Test",
                       showarrow=False, align="right",
                       bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    return fig


# =============================================================================
# EVALUATE AGENT
# =============================================================================

def evaluate_agent(agent, dataset, start_date, end_date,
                   transaction_cost_bps=5.0, lookback_window=60):
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
    return {"results": results, "metrics": metrics, "equity": equity}


# =============================================================================
# TRAIN AGENT (single fold)
# =============================================================================

def train_agent(agent, dataset, train_start, train_end, val_start, val_end,
                n_epochs=30, patience=5, min_epochs=10,
                transaction_cost_bps=5.0, turnover_penalty=0.001,
                variance_penalty=0.5, tc_curriculum_frac=0.3,
                lookback_window=60, verbose=True):
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
    best_epoch = 0
    best_state_bytes = None
    patience_counter = 0

    for epoch in range(n_epochs):
        t0 = time.time()
        state = train_env.reset()
        while not train_env.done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = train_env.step(action)
            agent.store_transition(state, action, reward, next_state, done,
                                   info["n_tradable"])
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

        elapsed = time.time() - t0
        if verbose:
            print(f"    Ep {epoch:2d} | Train IR2: {train_m['IR2']:.4f} "
                  f"Val IR2: {val_ir2:.4f} | Val ARC: {val_arc:+.1f}% | "
                  f"α: {agent.alpha.item():.3f} | {elapsed:.1f}s")

        # Early stopping (IR2 when > 0, else ARC)
        score = val_ir2 if val_ir2 > 0 else val_arc / 100.0
        if score > best_val_score:
            best_val_score = score
            best_val_ir2 = val_ir2
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

    return {"best_val_ir2": best_val_ir2, "best_epoch": best_epoch}


# =============================================================================
# HP CONFIGS (5 configs)
# =============================================================================

DEFAULT_HP_CONFIGS = [
    {"name": "baseline",     "lr_actor": 3e-4, "lr_critic": 3e-4, "lstm_hidden": 64,  "n_attn_heads": 4, "variance_penalty": 0.5},
    {"name": "conservative", "lr_actor": 1e-4, "lr_critic": 3e-4, "lstm_hidden": 64,  "n_attn_heads": 4, "variance_penalty": 1.0},
    {"name": "large_model",  "lr_actor": 3e-4, "lr_critic": 3e-4, "lstm_hidden": 128, "n_attn_heads": 8, "variance_penalty": 0.5},
    {"name": "aggressive",   "lr_actor": 5e-4, "lr_critic": 5e-4, "lstm_hidden": 64,  "n_attn_heads": 2, "variance_penalty": 0.2},
    {"name": "high_cap_con", "lr_actor": 1e-4, "lr_critic": 3e-4, "lstm_hidden": 128, "n_attn_heads": 4, "variance_penalty": 1.0},
]


def select_hyperparameters(dataset, first_fold, hp_configs, n_epochs=20,
                           patience=5, min_epochs=8, transaction_cost_bps=5.0,
                           turnover_penalty=0.001, lookback_window=60, verbose=True):
    print(f"\n  HP Selection on fold 1:")
    print(f"    Train: {first_fold['train_start']} → {first_fold['train_end']} ({first_fold['n_train']}d)")
    print(f"    Val:   {first_fold['val_start']} → {first_fold['val_end']} ({first_fold['n_val']}d)")

    results = []
    for hp in hp_configs:
        print(f"\n    --- Config: {hp['name']} ---")
        vp = hp.pop("variance_penalty", 0.5)
        config = {
            "n_asset_features": dataset["metadata"]["n_per_asset_features"],
            "n_global_features": dataset["metadata"]["n_global_features"],
            **{k: v for k, v in hp.items() if k != "name"},
        }
        agent = SACAgent(config)
        result = train_agent(
            agent, dataset,
            first_fold["train_start"], first_fold["train_end"],
            first_fold["val_start"], first_fold["val_end"],
            n_epochs=n_epochs, patience=patience, min_epochs=min_epochs,
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty=turnover_penalty,
            variance_penalty=vp, lookback_window=lookback_window, verbose=verbose,
        )
        hp["variance_penalty"] = vp  # restore
        results.append({"name": hp["name"], "config": hp, "val_ir2": result["best_val_ir2"],
                         "variance_penalty": vp})
        print(f"    → Val IR2: {result['best_val_ir2']:.4f}")
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(results, key=lambda x: x["val_ir2"])
    print(f"\n  ★ Selected: '{best['name']}' (Val IR2: {best['val_ir2']:.4f})")
    return best, results


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def _save_checkpoint(out_dir, fold_id, agent, fold_log, all_test_returns,
                     all_test_qqq_returns, val_sharpe_history, last_retrain_fold,
                     selected_config):
    ckpt = {
        "fold_id": fold_id,
        "fold_log": fold_log,
        "val_sharpe_history": val_sharpe_history,
        "last_retrain_fold": last_retrain_fold,
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
    n_epochs: int = 30,
    patience: int = 5,
    min_epochs: int = 10,
    transaction_cost_bps: float = 5.0,
    turnover_penalty: float = 0.001,
    variance_penalty: float = 0.5,
    tc_curriculum_frac: float = 0.3,
    retrain_cooldown: int = 3,
    lookback_window: int = 60,
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
    val_sharpe_history = []  # rolling window for retrain decision
    last_retrain_fold = -999  # track cooldown
    selected_config = None
    agent = None

    if ckpt:
        start_fold = ckpt["fold_id"]  # resume from NEXT fold
        fold_log = ckpt["fold_log"]
        val_sharpe_history = ckpt.get("val_sharpe_history", [])
        last_retrain_fold = ckpt.get("last_retrain_fold", -999)
        selected_config = ckpt["selected_config"]
        # Load saved test returns
        for i in range(ckpt["n_test_returns"]):
            p = out_dir / f"rl_fold_{i+1}_test_returns.csv"
            if p.exists():
                s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
                all_test_returns.append(s)
            p2 = out_dir / f"rl_fold_{i+1}_qqq_returns.csv"
            if p2.exists():
                all_test_qqq_returns.append(pd.read_csv(p2, index_col=0, parse_dates=True).iloc[:, 0])
        print(f"\n  *** RESUMING from fold {start_fold + 1} (completed {start_fold} folds) ***\n")

    # --- HP selection on fold 1 ---
    if selected_config is None:
        best_hp, hp_results = select_hyperparameters(
            dataset, folds[0], hp_configs, n_epochs=min(n_epochs, 20),
            patience=patience, min_epochs=min(min_epochs, 8),
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty=turnover_penalty,
            lookback_window=lookback_window, verbose=verbose,
        )
        selected_config = best_hp["config"]
        variance_penalty = best_hp.get("variance_penalty", variance_penalty)

    # Build agent config
    agent_config = {
        "n_asset_features": dataset["metadata"]["n_per_asset_features"],
        "n_global_features": dataset["metadata"]["n_global_features"],
        **{k: v for k, v in selected_config.items() if k not in ("name", "variance_penalty")},
    }

    # --- Walk-forward loop ---
    print(f"\n{'='*70}")
    print(f"WALKING FORWARD — {len(folds)} folds")
    print(f"{'='*70}")

    if agent is None:
        agent = SACAgent(agent_config)
    # Load checkpoint weights if resuming
    agent_ckpt_path = out_dir / "agent_checkpoint.pt"
    if ckpt and agent_ckpt_path.exists():
        agent.load(str(agent_ckpt_path))
        print(f"  Loaded agent checkpoint from {agent_ckpt_path}")

    n_retrains = sum(1 for f in fold_log if f.get("retrained", False))

    for i, fold in enumerate(folds):
        if i < start_fold:
            continue

        fid = fold["fold_id"]

        # Evaluate current agent on val
        val_r = evaluate_agent(agent, dataset, fold["val_start"], fold["val_end"],
                               transaction_cost_bps, lookback_window)
        current_val_ir2 = val_r["metrics"]["IR2"]
        # Compute val Sharpe for retrain decision (more stable on short windows)
        val_rets = val_r["results"]["portfolio_return_net"]
        current_val_sharpe = (val_rets.mean() / val_rets.std() * np.sqrt(252)) if val_rets.std() > 0 else 0.0

        # --- Retrain decision: rolling Sharpe (no cooldown) ---
        need_retrain = False

        if i == 0 and not ckpt:
            need_retrain = True  # always train first fold
            reason = "initial"
        elif len(val_sharpe_history) >= 3:
            # Retrain when current Sharpe < median(last 5) - 1*std(last 5)
            recent = val_sharpe_history[-5:]
            med = np.median(recent)
            std = np.std(recent) if len(recent) > 1 else 0.0
            threshold = med - 1.0 * std
            if current_val_sharpe < threshold:
                need_retrain = True
                reason = f"Sharpe {current_val_sharpe:.3f} < {threshold:.3f} (med={med:.3f} - 1*std={std:.3f})"
        # else: not enough history yet, carry forward

        val_sharpe_history.append(current_val_sharpe)

        if need_retrain:
            if verbose:
                print(f"\n  Fold {fid:2d}/{len(folds)} | RETRAIN ({reason})")
                print(f"    Train: {fold['train_start']}→{fold['train_end']} ({fold['n_train']}d) "
                      f"Val: {fold['val_start']}→{fold['val_end']} ({fold['n_val']}d)")

            if i == 0 and not ckpt:
                agent = SACAgent(agent_config)
            else:
                agent.reset_for_fine_tune()

            train_result = train_agent(
                agent, dataset,
                fold["train_start"], fold["train_end"],
                fold["val_start"], fold["val_end"],
                n_epochs=n_epochs, patience=patience, min_epochs=min_epochs,
                transaction_cost_bps=transaction_cost_bps,
                turnover_penalty=turnover_penalty,
                variance_penalty=variance_penalty,
                tc_curriculum_frac=tc_curriculum_frac,
                lookback_window=lookback_window, verbose=verbose,
            )
            current_val_ir2 = train_result["best_val_ir2"]
            n_retrains += 1
            last_retrain_fold = i
        else:
            if verbose:
                print(f"  Fold {fid:2d}/{len(folds)} | CARRY (Sharpe: {current_val_sharpe:.3f})", end="")

        # Test
        test_r = evaluate_agent(agent, dataset, fold["test_start"], fold["test_end"],
                                transaction_cost_bps, lookback_window)
        test_ir2 = test_r["metrics"]["IR2"]
        test_arc = test_r["metrics"]["ARC (%)"]

        if verbose and not need_retrain:
            print(f" → Test IR2: {test_ir2:.4f}, ARC: {test_arc:+.1f}%")
        elif verbose:
            print(f"    Test: {fold['test_start']}→{fold['test_end']} → "
                  f"IR2: {test_ir2:.4f}, ARC: {test_arc:+.1f}%")

        # Collect test returns
        all_test_returns.append(test_r["results"]["portfolio_return_net"])
        all_test_qqq_returns.append(test_r["results"]["qqq_return"])
        if "turnover" in test_r["results"]:
            all_test_turnover.append(test_r["results"]["turnover"])

        # Save per-fold returns (for checkpoint resume)
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
            "val_ir2": round(current_val_ir2, 4),
            "val_sharpe": round(current_val_sharpe, 4),
            "test_ir2": round(test_ir2, 4),
            "test_arc": round(test_arc, 2),
            "test_mdd": round(test_r["metrics"]["Max Drawdown (%)"], 2),
            "test_cash": round(test_r["metrics"].get("Avg Cash (%)", 0), 2),
        })

        # Checkpoint
        _save_checkpoint(out_dir, i + 1, agent, fold_log,
                         all_test_returns, all_test_qqq_returns,
                         val_sharpe_history, last_retrain_fold, selected_config)

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
    print(f"\n  {'METRIC':<25} {'RL':>10} {'QQQ':>10}")
    print(f"  {'-'*47}")
    for k in ["ARC (%)", "ASD (%)", "Max Drawdown (%)", "IR1", "IR2"]:
        print(f"  {k:<25} {rl_m[k]:>10.4f} {qqq_m[k]:>10.4f}")

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
               "window_type": "SLIDING (non-anchored)"}
    with open(out_dir / "rl_wfo_config.json", "w") as f:
        json.dump(wfo_cfg, f, indent=2)

    print(f"\n  All saved to {out_dir}")
    return {"rl_equity": rl_equity, "qqq_equity": qqq_equity,
            "rl_oos_metrics": rl_m, "fold_log": pd.DataFrame(fold_log),
            "agent": agent}
