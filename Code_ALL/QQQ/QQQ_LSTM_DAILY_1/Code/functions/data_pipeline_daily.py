"""
data_pipeline_daily.py — Feature Pipeline for DAILY Rebalancing
================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Adapted from intraday pipeline for daily close data (2003-2026).
Key differences:
    - No hourly/session splitting — each row is one trading day
    - Technical indicators use standard daily periods (RSI-14, MACD 12/26/9, etc.)
    - No intraday_ret / intraday_rvol features (no hourly data)
    - 13 per-asset features (was 15 for intraday)
    - 9 global features (same)
    - Annualization: 252 (not 504)

Compatible with: environment.py, train.py, baseline.py, networks.py, sac_agent.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_raw_data(data_dir: str) -> dict:
    data_dir = Path(data_dir)

    # Daily close prices: index=date, columns=tickers
    prices = pd.read_csv(
        data_dir / "close_prices.csv",
        index_col=0, parse_dates=True
    )
    prices.index.name = "date"

    # Tradable mask (NASDAQ-100 membership)
    mask = pd.read_csv(
        data_dir / "tradable_mask.csv",
        index_col=0, parse_dates=True
    )
    mask.index.name = "date"

    # QQQ — try yfinance multi-row header format first, fall back to simple
    try:
        qqq = pd.read_csv(data_dir / "QQQ.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
        qqq = qqq[["Close"]].rename(columns={"Close": "qqq_close"})
    except Exception:
        qqq = pd.read_csv(data_dir / "QQQ.csv", index_col=0, parse_dates=True)
        close_col = [c for c in qqq.columns if "close" in c.lower() or "Close" in c]
        qqq = qqq[[close_col[0]]].rename(columns={close_col[0]: "qqq_close"})
    qqq.index.name = "date"
    qqq["qqq_close"] = qqq["qqq_close"].astype(float)

    # VIX
    try:
        vix = pd.read_csv(data_dir / "VIX.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
        vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
    except Exception:
        vix = pd.read_csv(data_dir / "VIX.csv", index_col=0, parse_dates=True)
        close_col = [c for c in vix.columns if "close" in c.lower() or "Close" in c]
        vix = vix[[close_col[0]]].rename(columns={close_col[0]: "vix_close"})
    vix.index.name = "date"
    vix["vix_close"] = vix["vix_close"].astype(float)

    # Risk-free rate
    rf_rate = None
    rf_candidates = [data_dir / "risk_free_data.csv", data_dir.parent / "risk_free_data.csv"]
    for rf_path in rf_candidates:
        if rf_path.exists():
            try:
                rf_rate = pd.read_csv(rf_path, skiprows=[1, 2], index_col=0, parse_dates=True)
                rf_rate = rf_rate[["Close"]].rename(columns={"Close": "rf_annualized_pct"})
            except Exception:
                rf_rate = pd.read_csv(rf_path, index_col=0, parse_dates=True)
                close_col = [c for c in rf_rate.columns if "close" in c.lower() or "Close" in c]
                rf_rate = rf_rate[[close_col[0]]].rename(columns={close_col[0]: "rf_annualized_pct"})
            rf_rate.index.name = "date"
            rf_rate["rf_annualized_pct"] = rf_rate["rf_annualized_pct"].astype(float)
            rf_rate["rf_daily"] = (1 + rf_rate["rf_annualized_pct"] / 100) ** (1/252) - 1
            print(f"  Risk-free rate loaded from: {rf_path}")
            break
    if rf_rate is None:
        print("  WARNING: risk_free_data.csv not found — cash will earn 0%")

    return {"prices": prices, "mask": mask, "qqq": qqq, "vix": vix, "rf_rate": rf_rate}


# =============================================================================
# 2. CLEANING & ALIGNMENT (daily — no hourly filtering)
# =============================================================================

JUNK_TICKERS = {"9210611D", "9218611D", "9996651D", "ALNUW", "MDBUQ", "LEND", "MPWRUW", "File"}


def clean_and_align(raw: dict) -> dict:
    prices = raw["prices"].copy()
    mask = raw["mask"].copy()
    qqq = raw["qqq"].copy()
    vix = raw["vix"].copy()

    # Common tickers (drop junk)
    tickers = sorted((set(prices.columns) & set(mask.columns)) - JUNK_TICKERS)
    prices = prices[tickers]
    mask = mask.reindex(columns=tickers).fillna(0).astype(int)

    # Align indices
    common_dates = prices.index.intersection(mask.index)
    prices = prices.loc[common_dates]
    mask = mask.loc[common_dates]

    # Forward-fill prices (max 1 day for rare gaps)
    prices = prices.ffill(limit=1)

    # Trading dates
    trading_dates = prices.index

    # Align QQQ, VIX, risk-free to trading dates
    qqq = qqq.reindex(trading_dates).ffill().bfill()
    vix = vix.reindex(trading_dates).ffill().bfill()

    rf_rate = raw.get("rf_rate")
    if rf_rate is not None:
        rf_rate = rf_rate.reindex(trading_dates).ffill().bfill()
    else:
        rf_rate = pd.DataFrame({"rf_annualized_pct": 0.0, "rf_daily": 0.0}, index=trading_dates)

    return {
        "daily_close": prices,
        "daily_mask": mask,
        "qqq": qqq,
        "vix": vix,
        "rf_rate": rf_rate,
        "tickers": tickers,
        "trading_dates": trading_dates,
    }


# =============================================================================
# 3. PER-ASSET FEATURES (daily)
# =============================================================================

def cross_sectional_rank(df, mask):
    """Rank across assets, normalize to [-1, 1]. Non-members get NaN."""
    masked = df.where(mask == 1)
    ranked = masked.rank(axis=1, pct=True)
    return 2 * ranked - 1


def zscore_timeseries(df, period=252):
    """Per-asset expanding z-score (preserves absolute magnitude info)."""
    mean = df.expanding(min_periods=max(20, period // 4)).mean()
    std = df.expanding(min_periods=max(20, period // 4)).std()
    return (df - mean) / (std + 1e-10)


def compute_rsi(close, period=14):
    """Standard RSI, scaled to [-1, 1]."""
    log_ret = np.log(close.replace(0, np.nan)).diff()
    gain = log_ret.clip(lower=0)
    loss = (-log_ret).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period // 2).mean()
    avg_loss = loss.ewm(span=period, min_periods=period // 2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 50 - 1  # scale to [-1, 1]


def compute_macd_histogram(close, fast=12, slow=26, signal=9):
    """Standard daily MACD histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast // 2).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow // 2).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal // 2).mean()
    return macd_line - signal_line


def compute_bollinger_pctb(close, period=20, n_std=2):
    """Bollinger Band %B: (price - lower) / (upper - lower)."""
    sma = close.rolling(period, min_periods=period // 2).mean()
    std = close.rolling(period, min_periods=period // 2).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return (close - lower) / (upper - lower + 1e-10)


def compute_dist_from_high(close, period=20):
    """(price / rolling_max - 1). Always <= 0."""
    rolling_max = close.rolling(period, min_periods=period // 2).max()
    return close / (rolling_max + 1e-10) - 1


def compute_mean_reversion(close, period=20):
    """price / SMA(period) - 1."""
    sma = close.rolling(period, min_periods=period // 2).mean()
    return close / (sma + 1e-10) - 1


def compute_rolling_beta(close, qqq_series, period=60):
    """Rolling beta of each asset vs QQQ."""
    log_ret = np.log(close.replace(0, np.nan)).diff()
    qqq_ret = np.log(qqq_series.replace(0, np.nan)).diff()
    qqq_var = qqq_ret.rolling(period, min_periods=period // 2).var()
    betas = {}
    for col in close.columns:
        cov = log_ret[col].rolling(period, min_periods=period // 2).cov(qqq_ret)
        betas[col] = cov / (qqq_var + 1e-10)
    return pd.DataFrame(betas, index=close.index)


def build_per_asset_features(clean):
    """
    Build 13 per-asset features from daily close data.
    7 ranked technical + 5 ranked momentum/vol + 1 z-scored absolute.
    """
    close = clean["daily_close"]
    mask = clean["daily_mask"]
    tickers = clean["tickers"]
    qqq = clean["qqq"]["qqq_close"]

    features_raw = {}
    log_close = np.log(close.replace(0, np.nan))

    # --- Momentum (4 features) ---
    for lb in [1, 5, 20, 60]:
        features_raw[f"ret_{lb}d"] = log_close.diff(lb)

    # --- Realized volatility (2 features) ---
    log_ret = log_close.diff()
    for w in [5, 20]:
        features_raw[f"rvol_{w}d"] = log_ret.rolling(w, min_periods=max(2, w // 2)).std()

    # --- Technical indicators (5 features) ---
    features_raw["rsi_14"] = compute_rsi(close, period=14)
    features_raw["macd_hist"] = compute_macd_histogram(close, fast=12, slow=26, signal=9)
    features_raw["bb_pctb"] = compute_bollinger_pctb(close, period=20, n_std=2)
    features_raw["dist_high_20"] = compute_dist_from_high(close, period=20)
    features_raw["mean_rev_20"] = compute_mean_reversion(close, period=20)

    # --- Market sensitivity (1 feature) ---
    features_raw["beta_qqq"] = compute_rolling_beta(close, qqq, period=60)

    # --- Absolute momentum z-scored (1 feature, preserves magnitude) ---
    ret_20d_raw = log_close.diff(20)
    features_raw["raw_ret_20d"] = zscore_timeseries(ret_20d_raw, period=252)

    # --- Encoding: cross-sectional rank for most, z-scored for raw ---
    ranked_names = ["ret_1d", "ret_5d", "ret_20d", "ret_60d",
                    "rvol_5d", "rvol_20d", "rsi_14", "macd_hist",
                    "bb_pctb", "dist_high_20", "mean_rev_20", "beta_qqq"]
    zscore_names = ["raw_ret_20d"]  # already z-scored above

    features_encoded = {}
    for name in ranked_names:
        df = features_raw[name].reindex(columns=tickers)
        features_encoded[name] = cross_sectional_rank(df, mask)
    for name in zscore_names:
        features_encoded[name] = features_raw[name].reindex(columns=tickers)

    # Stack into MultiIndex DataFrame: (ticker, feature)
    panels = []
    for feat_name in sorted(features_encoded.keys()):
        df = features_encoded[feat_name]
        df.columns = pd.MultiIndex.from_product(
            [[feat_name], df.columns], names=["feature", "ticker"]
        )
        panels.append(df)

    per_asset = pd.concat(panels, axis=1)
    per_asset = per_asset.swaplevel(axis=1).sort_index(axis=1)
    return per_asset


# =============================================================================
# 4. GLOBAL FEATURES
# =============================================================================

def build_global_features(clean):
    """9 global features: VIX, market stats, QQQ momentum, cross-section dispersion."""
    vix = clean["vix"]["vix_close"]
    close = clean["daily_close"]
    mask = clean["daily_mask"]
    trading_dates = clean["trading_dates"]
    qqq = clean["qqq"]["qqq_close"]

    features = pd.DataFrame(index=trading_dates)

    # --- VIX ---
    vix_mean = vix.expanding(min_periods=20).mean()
    vix_std = vix.expanding(min_periods=20).std()
    features["vix_level"] = (vix - vix_mean) / (vix_std + 1e-10)
    features["vix_change_5d"] = vix.pct_change(5)

    # --- Market return & vol ---
    log_ret = np.log(close.replace(0, np.nan)).diff()
    member_ret = log_ret.where(mask == 1)
    features["market_ret_1d"] = member_ret.mean(axis=1)
    features["market_rvol_5d"] = features["market_ret_1d"].rolling(5).std()

    # --- Day of week ---
    features["dow"] = trading_dates.dayofweek / 4.0

    # --- Market breadth ---
    ret_5d = np.log(close.replace(0, np.nan)).diff(5)
    tradable_count = (mask == 1).sum(axis=1).clip(lower=1)
    breadth = (ret_5d.where(mask == 1) > 0).sum(axis=1) / tradable_count
    features["market_breadth"] = breadth * 2 - 1

    # --- QQQ momentum ---
    qqq_log = np.log(qqq.replace(0, np.nan))
    qqq_r5 = qqq_log.diff(5)
    qqq_r20 = qqq_log.diff(20)
    features["qqq_ret_5d"] = (qqq_r5 - qqq_r5.expanding(20).mean()) / (qqq_r5.expanding(20).std() + 1e-10)
    features["qqq_ret_20d"] = (qqq_r20 - qqq_r20.expanding(20).mean()) / (qqq_r20.expanding(20).std() + 1e-10)

    # --- Cross-sectional dispersion ---
    cross_disp = log_ret.where(mask == 1).std(axis=1)
    features["cross_disp"] = (cross_disp - cross_disp.expanding(20).mean()) / (cross_disp.expanding(20).std() + 1e-10)

    return features


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def build_dataset(data_dir: str) -> dict:
    print("=" * 60)
    print("DATA PIPELINE — RL Portfolio (DAILY REBALANCING)")
    print("=" * 60)

    print("\n[1/5] Loading raw data...")
    raw = load_raw_data(data_dir)
    print(f"  Prices: {raw['prices'].shape}")
    print(f"  Mask:   {raw['mask'].shape}")
    print(f"  QQQ:    {raw['qqq'].shape}")
    print(f"  VIX:    {raw['vix'].shape}")

    print("\n[2/5] Cleaning & aligning (daily)...")
    clean = clean_and_align(raw)
    print(f"  Tickers:       {len(clean['tickers'])}")
    print(f"  Trading days:  {len(clean['trading_dates'])}")
    print(f"  Date range:    {clean['trading_dates'][0].date()} → "
          f"{clean['trading_dates'][-1].date()}")

    member_coverage = (
        (clean["daily_mask"] == 1) & clean["daily_close"].notna()
    ).sum(axis=1) / (clean["daily_mask"] == 1).sum(axis=1).clip(lower=1)
    print(f"  Close coverage: {member_coverage.mean():.1%} mean, {member_coverage.min():.1%} min")

    print("\n[3/5] Building per-asset features (13 features)...")
    per_asset = build_per_asset_features(clean)
    n_features = len(per_asset.columns.get_level_values("feature").unique())
    print(f"  Features per asset:  {n_features}")
    print(f"  Feature names:       {sorted(per_asset.columns.get_level_values('feature').unique())}")
    print(f"  Total columns:       {per_asset.shape[1]}")

    print("\n[4/5] Building global features (9 features)...")
    global_feat = build_global_features(clean)
    print(f"  Global features: {list(global_feat.columns)}")

    # Warmup: 60 days for rolling window features
    print("\n[5/5] Applying warmup period (60 days)...")
    warmup = 60
    valid_dates = clean["trading_dates"][warmup:]

    per_asset = per_asset.loc[valid_dates]
    global_feat = global_feat.loc[valid_dates]
    daily_close = clean["daily_close"].loc[valid_dates]
    daily_mask = clean["daily_mask"].loc[valid_dates]
    qqq = clean["qqq"].loc[valid_dates]
    rf_rate = clean["rf_rate"].loc[valid_dates]

    pa_nan = per_asset.isna().mean().mean()
    gf_nan = global_feat.isna().mean().mean()
    print(f"  Valid range:    {valid_dates[0].date()} → {valid_dates[-1].date()}")
    print(f"  Valid days:     {len(valid_dates)}")
    print(f"  Per-asset NaN:  {pa_nan:.3%}")
    print(f"  Global NaN:     {gf_nan:.3%}")
    rf_mean = rf_rate["rf_annualized_pct"].mean() if "rf_annualized_pct" in rf_rate.columns else 0.0
    print(f"  Avg risk-free:  {rf_mean:.2f}% annualized")

    metadata = {
        "warmup_days": warmup,
        "n_tickers": len(clean["tickers"]),
        "n_per_asset_features": n_features,
        "n_global_features": len(global_feat.columns),
        "n_trading_days": len(valid_dates),
        "date_range": (valid_dates[0].date(), valid_dates[-1].date()),
        "per_asset_nan_rate": pa_nan,
        "global_nan_rate": gf_nan,
        "rebalancing": "daily",
    }

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (DAILY MODE)")
    print("=" * 60)

    return {
        "per_asset_features": per_asset,
        "global_features": global_feat,
        "daily_close": daily_close,
        "daily_mask": daily_mask,
        "qqq": qqq,
        "rf_rate": rf_rate,
        "tickers": clean["tickers"],
        "trading_dates": valid_dates,
        "metadata": metadata,
    }
