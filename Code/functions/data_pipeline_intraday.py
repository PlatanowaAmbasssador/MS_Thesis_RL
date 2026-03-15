"""
data_pipeline_intraday.py — Feature Pipeline for 2x/Day Rebalancing
=====================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Key difference from daily pipeline:
    - Each trading day is split into 2 sessions:
        AM session: 14:00-16:00 (3 hourly bars)
        PM session: 17:00-19:00 (3 hourly bars)
    - Agent rebalances at end of each session (2x per day)
    - Features computed at session resolution
    - 15 per-asset features (7 ranked + 8 new: RSI, MACD, BB, etc.)
    - 9 global features (original 5 + breadth, QQQ momentum, dispersion)
    - trading_dates has 2 entries per day (session-end timestamps)

Compatible with: environment.py, train.py, baseline.py, networks.py, sac_agent.py
(all unchanged — they just see 2x more "trading dates")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# =============================================================================
# 1. LOADING (identical to daily pipeline)
# =============================================================================

def load_raw_data(data_dir: str) -> dict:
    data_dir = Path(data_dir)

    prices = pd.read_csv(
        data_dir / "close_prices.csv",
        index_col=0, parse_dates=True
    )
    prices.index.name = "datetime"

    mask = pd.read_csv(
        data_dir / "tradable_mask.csv",
        index_col=0, parse_dates=True
    )
    mask.index.name = "datetime"

    qqq = pd.read_csv(
        data_dir / "QQQ.csv",
        skiprows=[1, 2], index_col=0, parse_dates=True
    )
    qqq.index.name = "date"
    qqq = qqq[["Close"]].rename(columns={"Close": "qqq_close"})
    qqq["qqq_close"] = qqq["qqq_close"].astype(float)

    vix = pd.read_csv(
        data_dir / "VIX.csv",
        skiprows=[1, 2], index_col=0, parse_dates=True
    )
    vix.index.name = "date"
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
    vix["vix_close"] = vix["vix_close"].astype(float)

    rf_rate = None
    rf_candidates = [
        data_dir / "risk_free_data.csv",
        data_dir.parent / "risk_free_data.csv",
    ]
    for rf_path in rf_candidates:
        if rf_path.exists():
            rf_rate = pd.read_csv(
                rf_path, skiprows=[1, 2], index_col=0, parse_dates=True
            )
            rf_rate.index.name = "date"
            rf_rate = rf_rate[["Close"]].rename(columns={"Close": "rf_annualized_pct"})
            rf_rate["rf_annualized_pct"] = rf_rate["rf_annualized_pct"].astype(float)
            rf_rate["rf_daily"] = (1 + rf_rate["rf_annualized_pct"] / 100) ** (1/252) - 1
            print(f"  Risk-free rate loaded from: {rf_path}")
            break
    if rf_rate is None:
        print("  WARNING: risk_free_data.csv not found — cash will earn 0%")

    return {"prices": prices, "mask": mask, "qqq": qqq, "vix": vix, "rf_rate": rf_rate}


# =============================================================================
# 2. CLEANING & SESSION SPLITTING
# =============================================================================

VALID_HOURS = list(range(14, 20))  # 14:00–19:00 inclusive
AM_HOURS = [14, 15, 16]  # AM session
PM_HOURS = [17, 18, 19]  # PM session

JUNK_TICKERS = {
    "9210611D", "9218611D", "9996651D", "ALNUW",
    "MDBUQ", "LEND", "MPWRUW", "File"
}


def clean_and_align(raw: dict) -> dict:
    """
    Filter, clean, and split into AM/PM sessions.
    Returns session-level data (2 entries per trading day).
    """
    prices = raw["prices"].copy()
    mask = raw["mask"].copy()
    qqq = raw["qqq"].copy()
    vix = raw["vix"].copy()

    # Common tickers
    tickers = sorted(
        (set(prices.columns) & set(mask.columns)) - JUNK_TICKERS
    )
    prices = prices[tickers]
    mask = mask[tickers]

    # Filter to valid hours
    prices = prices[prices.index.hour.isin(VALID_HOURS)]
    mask = mask.reindex(prices.index).fillna(0).astype(int)

    # Drop short days (need 6 bars = 2 full sessions)
    bars_per_day = prices.groupby(prices.index.date).size()
    full_days = bars_per_day[bars_per_day == len(VALID_HOURS)].index
    keep_idx = prices.index[
        prices.index.normalize().isin(pd.to_datetime(list(full_days)))
    ]
    prices = prices.loc[keep_idx]
    mask = mask.loc[keep_idx]

    # Forward-fill within each day
    prices = prices.groupby(prices.index.date).apply(lambda g: g.ffill())
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.droplevel(0)

    # --- BUILD SESSION-LEVEL CLOSE PRICES ---
    # AM close = 16:00 bar, PM close = 19:00 bar
    am_close = prices[prices.index.hour == 16].copy()
    pm_close = prices[prices.index.hour == 19].copy()

    # Tag with session identifiers (keep as timestamps for proper ordering)
    # AM session end: date at 16:00
    # PM session end: date at 19:00
    # These become our "trading_dates" — 2 per day

    # Build session close DataFrame: interleave AM and PM
    session_closes = []
    session_timestamps = []

    for day in sorted(full_days):
        day_ts = pd.Timestamp(day)
        am_ts = day_ts + pd.Timedelta(hours=16)
        pm_ts = day_ts + pd.Timedelta(hours=19)

        if am_ts in am_close.index and pm_ts in pm_close.index:
            session_closes.append(am_close.loc[am_ts])
            session_timestamps.append(am_ts)
            session_closes.append(pm_close.loc[pm_ts])
            session_timestamps.append(pm_ts)

    session_close = pd.DataFrame(session_closes, index=pd.DatetimeIndex(session_timestamps))
    session_close.index.name = "datetime"

    # Cross-session forward-fill (1 session max)
    session_close = session_close.ffill(limit=1)

    # Session-level mask (replicate daily mask for both sessions)
    daily_mask_raw = mask[mask.index.hour == 19].copy()
    daily_mask_raw.index = daily_mask_raw.index.normalize()

    session_mask_rows = []
    session_mask_timestamps = []
    for ts in session_timestamps:
        day = ts.normalize()
        if day in daily_mask_raw.index:
            session_mask_rows.append(daily_mask_raw.loc[day])
        else:
            session_mask_rows.append(pd.Series(1, index=tickers))
        session_mask_timestamps.append(ts)

    session_mask = pd.DataFrame(session_mask_rows, index=pd.DatetimeIndex(session_mask_timestamps))
    session_mask.index.name = "datetime"

    # Session-level trading dates
    trading_dates = session_close.index

    # --- Align QQQ and VIX ---
    # QQQ/VIX are daily — expand to session level (same value for AM and PM)
    qqq_session = pd.DataFrame(index=trading_dates, columns=["qqq_close"])
    vix_session = pd.DataFrame(index=trading_dates, columns=["vix_close"])

    for ts in trading_dates:
        day = ts.normalize()
        # Find nearest daily QQQ/VIX
        qqq_val = qqq.loc[:day, "qqq_close"].iloc[-1] if day >= qqq.index[0] else np.nan
        vix_val = vix.loc[:day, "vix_close"].iloc[-1] if day >= vix.index[0] else np.nan
        qqq_session.loc[ts, "qqq_close"] = qqq_val
        vix_session.loc[ts, "vix_close"] = vix_val

    qqq_session["qqq_close"] = qqq_session["qqq_close"].astype(float).ffill()
    vix_session["vix_close"] = vix_session["vix_close"].astype(float).ffill()

    # --- Align risk-free rate (half the daily rate per session) ---
    rf_rate = raw.get("rf_rate")
    if rf_rate is not None:
        rf_session = pd.DataFrame(index=trading_dates)
        rf_daily_aligned = rf_rate["rf_daily"].reindex(
            trading_dates.normalize(), method="ffill"
        )
        # Half-day rate: (1 + r_daily)^0.5 - 1
        rf_session["rf_annualized_pct"] = rf_rate["rf_annualized_pct"].reindex(
            trading_dates.normalize(), method="ffill"
        ).values
        rf_session["rf_daily"] = ((1 + rf_daily_aligned.values) ** 0.5 - 1)
        rf_session.index = trading_dates
    else:
        rf_session = pd.DataFrame(
            {"rf_annualized_pct": 0.0, "rf_daily": 0.0},
            index=trading_dates,
        )

    return {
        "hourly_prices": prices,
        "hourly_mask": mask,
        "daily_close": session_close,      # session-level closes (misnamed for compat)
        "daily_mask": session_mask,         # session-level mask
        "qqq": qqq_session,
        "vix": vix_session,
        "rf_rate": rf_session,
        "tickers": tickers,
        "trading_dates": trading_dates,
        "sessions_per_day": 2,
    }


# =============================================================================
# 3. PER-ASSET FEATURES (session-level)
# =============================================================================

def compute_session_returns(session_close, lookbacks=[1, 5, 20]):
    """
    Returns over multiple lookback windows in SESSION units.
    lookback=1 means 1 session (half-day), lookback=5 means 5 sessions (2.5 days).

    We relabel to match the daily pipeline:
        ret_1d  → 2-session return  (≈ 1 day)
        ret_5d  → 10-session return (≈ 5 days)
        ret_20d → 40-session return (≈ 20 days)
    """
    features = {}
    log_close = np.log(session_close.replace(0, np.nan))
    # Map daily lookbacks to session lookbacks (×2)
    for lb in lookbacks:
        session_lb = lb * 2  # 1 day = 2 sessions
        features[f"ret_{lb}d"] = log_close.diff(session_lb)
    return features


def compute_session_vol(session_close, windows=[5, 20]):
    """
    Realized vol from session returns.
    rvol_5d  → rolling 10 sessions
    rvol_20d → rolling 40 sessions
    """
    features = {}
    log_ret = np.log(session_close.replace(0, np.nan)).diff()
    for w in windows:
        session_w = w * 2
        features[f"rvol_{w}d"] = log_ret.rolling(session_w, min_periods=max(2, session_w // 2)).std()
    return features


def compute_intraday_features(hourly_prices, session_timestamps):
    """
    Intraday features from 3 bars per session.
    For each session timestamp, find the 3 bars belonging to that session.
    """
    log_prices = np.log(hourly_prices.replace(0, np.nan))
    log_ret = log_prices.diff()

    intraday_ret_rows = []
    intraday_rvol_rows = []

    for ts in session_timestamps:
        hour = ts.hour
        day = ts.normalize()

        if hour == 16:
            # AM session: bars at 14, 15, 16
            session_hours = [14, 15, 16]
        elif hour == 19:
            # PM session: bars at 17, 18, 19
            session_hours = [17, 18, 19]
        else:
            # Fallback
            session_hours = [hour - 2, hour - 1, hour]

        session_idx = []
        for h in session_hours:
            bar_ts = day + pd.Timedelta(hours=h)
            if bar_ts in log_prices.index:
                session_idx.append(bar_ts)

        if len(session_idx) >= 2:
            session_log_p = log_prices.loc[session_idx]
            session_log_r = log_ret.loc[session_idx]

            # Intraday return: last - first
            ret_row = session_log_p.iloc[-1] - session_log_p.iloc[0]
            # Intraday rvol: std of hourly returns
            rvol_row = session_log_r.std()
        else:
            ret_row = pd.Series(0.0, index=hourly_prices.columns)
            rvol_row = pd.Series(0.0, index=hourly_prices.columns)

        intraday_ret_rows.append(ret_row)
        intraday_rvol_rows.append(rvol_row)

    intraday_ret = pd.DataFrame(intraday_ret_rows, index=session_timestamps)
    intraday_rvol = pd.DataFrame(intraday_rvol_rows, index=session_timestamps)

    return {
        "intraday_ret": intraday_ret,
        "intraday_rvol": intraday_rvol,
    }


def cross_sectional_rank(df, mask):
    masked = df.where(mask == 1)
    ranked = masked.rank(axis=1, pct=True)
    ranked = 2 * ranked - 1
    return ranked


def compute_rsi(session_close, period=28):
    """RSI over `period` sessions (~14 days). Scaled to [-1, 1]."""
    log_ret = np.log(session_close.replace(0, np.nan)).diff()
    gain = log_ret.clip(lower=0)
    loss = (-log_ret).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period // 2).mean()
    avg_loss = loss.ewm(span=period, min_periods=period // 2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 50 - 1


def compute_macd_histogram(session_close, fast=24, slow=52, signal=18):
    """MACD histogram in session units (fast~12d, slow~26d, signal~9d)."""
    ema_fast = session_close.ewm(span=fast, min_periods=fast // 2).mean()
    ema_slow = session_close.ewm(span=slow, min_periods=slow // 2).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal // 2).mean()
    return macd_line - signal_line


def compute_bollinger_pctb(session_close, period=40, n_std=2):
    """Bollinger Band %B: (price - lower) / (upper - lower)."""
    sma = session_close.rolling(period, min_periods=period // 2).mean()
    std = session_close.rolling(period, min_periods=period // 2).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return (session_close - lower) / (upper - lower + 1e-10)


def compute_dist_from_high(session_close, period=40):
    """(price / rolling_max - 1). Always <= 0."""
    rolling_max = session_close.rolling(period, min_periods=period // 2).max()
    return session_close / (rolling_max + 1e-10) - 1


def compute_mean_reversion(session_close, period=40):
    """price / SMA(period) - 1. Positive = above moving average."""
    sma = session_close.rolling(period, min_periods=period // 2).mean()
    return session_close / (sma + 1e-10) - 1


def compute_rolling_beta(session_close, qqq_series, period=40):
    """Rolling beta of each asset vs QQQ."""
    log_ret = np.log(session_close.replace(0, np.nan)).diff()
    qqq_ret = np.log(qqq_series.replace(0, np.nan)).diff()
    qqq_var = qqq_ret.rolling(period, min_periods=period // 2).var()
    betas = {}
    for col in session_close.columns:
        cov = log_ret[col].rolling(period, min_periods=period // 2).cov(qqq_ret)
        betas[col] = cov / (qqq_var + 1e-10)
    return pd.DataFrame(betas, index=session_close.index)


def zscore_timeseries(df, period=120):
    """Per-asset expanding z-score (preserves absolute magnitude)."""
    mean = df.expanding(min_periods=max(20, period // 4)).mean()
    std = df.expanding(min_periods=max(20, period // 4)).std()
    return (df - mean) / (std + 1e-10)


def build_per_asset_features(clean):
    session_close = clean["daily_close"]
    session_mask = clean["daily_mask"]
    hourly_prices = clean["hourly_prices"]
    tickers = clean["tickers"]
    trading_dates = clean["trading_dates"]
    qqq = clean["qqq"]["qqq_close"]

    features_raw = {}

    # --- Original 7 features ---
    features_raw.update(compute_session_returns(session_close, lookbacks=[1, 5, 20]))
    features_raw.update(compute_session_vol(session_close, windows=[5, 20]))
    intraday = compute_intraday_features(hourly_prices, trading_dates)
    for k, v in intraday.items():
        features_raw[k] = v.reindex(session_close.index)

    # --- 8 new features ---
    features_raw["rsi_14"] = compute_rsi(session_close, period=28)
    features_raw["macd_hist"] = compute_macd_histogram(session_close, fast=24, slow=52, signal=18)
    features_raw["bb_pctb"] = compute_bollinger_pctb(session_close, period=40, n_std=2)
    features_raw["dist_high_20"] = compute_dist_from_high(session_close, period=40)
    features_raw["ret_60d"] = np.log(session_close.replace(0, np.nan)).diff(120)
    features_raw["mean_rev_20"] = compute_mean_reversion(session_close, period=40)
    features_raw["beta_qqq"] = compute_rolling_beta(session_close, qqq, period=40)
    ret_20d_raw = np.log(session_close.replace(0, np.nan)).diff(40)
    features_raw["raw_ret_20d"] = zscore_timeseries(ret_20d_raw, period=120)

    # --- Encoding: cross-sectional rank for most, z-scored for some ---
    rank_features = [
        "intraday_ret", "intraday_rvol", "ret_1d", "ret_20d", "ret_5d",
        "rvol_20d", "rvol_5d",
        "macd_hist", "bb_pctb", "dist_high_20", "ret_60d",
        "mean_rev_20", "beta_qqq",
    ]
    noscale_features = ["rsi_14", "raw_ret_20d"]

    features_encoded = {}
    for name in rank_features:
        df = features_raw[name].reindex(columns=tickers)
        features_encoded[name] = cross_sectional_rank(df, session_mask)
    for name in noscale_features:
        df = features_raw[name].reindex(columns=tickers)
        features_encoded[name] = df.clip(-3, 3).fillna(0)

    # Stack into MultiIndex
    panels = []
    feature_names = sorted(features_encoded.keys())
    for feat_name in feature_names:
        df = features_encoded[feat_name]
        df.columns = pd.MultiIndex.from_product(
            [[feat_name], df.columns], names=["feature", "ticker"]
        )
        panels.append(df)

    per_asset = pd.concat(panels, axis=1)
    per_asset = per_asset.swaplevel(axis=1).sort_index(axis=1)
    return per_asset


# =============================================================================
# 4. GLOBAL FEATURES (session-level)
# =============================================================================

def build_global_features(clean):
    vix = clean["vix"]["vix_close"]
    session_close = clean["daily_close"]
    session_mask = clean["daily_mask"]
    trading_dates = clean["trading_dates"]
    qqq = clean["qqq"]["qqq_close"]

    features = pd.DataFrame(index=trading_dates)

    # --- Original 5 features ---
    vix_mean = vix.expanding(min_periods=40).mean()
    vix_std = vix.expanding(min_periods=40).std()
    features["vix_level"] = (vix - vix_mean) / (vix_std + 1e-10)

    features["vix_change_5d"] = vix.pct_change(10)

    log_ret = np.log(session_close.replace(0, np.nan)).diff()
    member_ret = log_ret.where(session_mask == 1)
    features["market_ret_1d"] = member_ret.mean(axis=1)

    features["market_rvol_5d"] = features["market_ret_1d"].rolling(10).std()

    dow = trading_dates.dayofweek / 4.0
    session_flag = (trading_dates.hour >= 17).astype(float) * 0.125
    features["dow"] = dow + session_flag

    # --- 4 new global features ---
    ret_10s = np.log(session_close.replace(0, np.nan)).diff(10)
    tradable_count = (session_mask == 1).sum(axis=1).clip(lower=1)
    breadth = (ret_10s.where(session_mask == 1) > 0).sum(axis=1) / tradable_count
    features["market_breadth"] = breadth * 2 - 1

    qqq_log = np.log(qqq.replace(0, np.nan))
    qqq_r5 = qqq_log.diff(10)
    qqq_r20 = qqq_log.diff(40)
    features["qqq_ret_5d"] = (qqq_r5 - qqq_r5.expanding(40).mean()) / (qqq_r5.expanding(40).std() + 1e-10)
    features["qqq_ret_20d"] = (qqq_r20 - qqq_r20.expanding(40).mean()) / (qqq_r20.expanding(40).std() + 1e-10)

    cross_disp = log_ret.where(session_mask == 1).std(axis=1)
    features["cross_disp"] = (cross_disp - cross_disp.expanding(40).mean()) / (cross_disp.expanding(40).std() + 1e-10)

    return features


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def build_dataset(data_dir: str) -> dict:
    print("=" * 60)
    print("DATA PIPELINE — RL Portfolio (2x/DAY REBALANCING)")
    print("=" * 60)

    print("\n[1/5] Loading raw data...")
    raw = load_raw_data(data_dir)
    print(f"  Prices: {raw['prices'].shape}")
    print(f"  Mask:   {raw['mask'].shape}")
    print(f"  QQQ:    {raw['qqq'].shape}")
    print(f"  VIX:    {raw['vix'].shape}")

    print("\n[2/5] Cleaning & splitting into AM/PM sessions...")
    clean = clean_and_align(raw)
    print(f"  Tickers:          {len(clean['tickers'])}")
    print(f"  Trading sessions: {len(clean['trading_dates'])}")
    print(f"  Sessions/day:     {clean['sessions_per_day']}")
    print(f"  Date range:       {clean['trading_dates'][0]} → "
          f"{clean['trading_dates'][-1]}")
    print(f"  Hourly bars:      {len(clean['hourly_prices'])}")

    # Coverage diagnostic
    member_coverage = (
        (clean["daily_mask"] == 1) & clean["daily_close"].notna()
    ).sum(axis=1) / (clean["daily_mask"] == 1).sum(axis=1)
    print(f"  Close coverage:   "
          f"{member_coverage.mean():.1%} mean, {member_coverage.min():.1%} min")

    print("\n[3/5] Building per-asset features (ranked + z-scored)...")
    per_asset = build_per_asset_features(clean)
    n_features = len(per_asset.columns.get_level_values("feature").unique())
    print(f"  Features per asset:  {n_features}")
    print(f"  Feature names:       {sorted(per_asset.columns.get_level_values('feature').unique())}")
    print(f"  Total columns:       {per_asset.shape[1]}")

    print("\n[4/5] Building global features...")
    global_feat = build_global_features(clean)
    print(f"  Global features: {list(global_feat.columns)}")

    # Warmup: 80 sessions = 40 trading days (enough for 40-session rolling features)
    print("\n[5/5] Applying warmup period (80 sessions = 40 days)...")
    warmup = 80
    valid_dates = clean["trading_dates"][warmup:]

    per_asset = per_asset.loc[valid_dates]
    global_feat = global_feat.loc[valid_dates]
    session_close = clean["daily_close"].loc[valid_dates]
    session_mask = clean["daily_mask"].loc[valid_dates]
    qqq = clean["qqq"].loc[valid_dates]
    rf_rate = clean["rf_rate"].loc[valid_dates]

    pa_nan = per_asset.isna().mean().mean()
    gf_nan = global_feat.isna().mean().mean()
    print(f"  Valid range:       {valid_dates[0]} → {valid_dates[-1]}")
    print(f"  Valid sessions:    {len(valid_dates)} ({len(valid_dates)//2} days)")
    print(f"  Per-asset NaN:     {pa_nan:.3%}")
    print(f"  Global NaN:        {gf_nan:.3%}")
    rf_mean = rf_rate["rf_annualized_pct"].mean() if "rf_annualized_pct" in rf_rate.columns else 0.0
    print(f"  Avg risk-free:     {rf_mean:.2f}% annualized")

    metadata = {
        "valid_hours": VALID_HOURS,
        "warmup_days": warmup // 2,
        "warmup_sessions": warmup,
        "n_tickers": len(clean["tickers"]),
        "n_per_asset_features": n_features,
        "n_global_features": len(global_feat.columns),
        "n_trading_days": len(valid_dates) // 2,
        "n_trading_sessions": len(valid_dates),
        "date_range": (valid_dates[0], valid_dates[-1]),
        "per_asset_nan_rate": pa_nan,
        "global_nan_rate": gf_nan,
        "sessions_per_day": 2,
        "rebalancing": "2x_daily",
    }

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (2x/DAY MODE)")
    print("=" * 60)

    return {
        "per_asset_features": per_asset,
        "global_features": global_feat,
        "daily_close": session_close,
        "daily_mask": session_mask,
        "qqq": qqq,
        "rf_rate": rf_rate,
        "tickers": clean["tickers"],
        "trading_dates": valid_dates,
        "metadata": metadata,
    }
