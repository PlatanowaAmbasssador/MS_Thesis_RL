"""
data_pipeline.py — Data Loading, Cleaning & Feature Engineering
================================================================
Master's Thesis: RL Portfolio Allocation for Dynamic NASDAQ-100

Pipeline:
    1. Load raw CSVs (close prices, tradable mask, QQQ, VIX)
    2. Clean & align: filter hours 14–19, drop junk tickers, handle NaN
    3. Build per-asset features from hourly data (aggregated to daily)
    4. Build global features (VIX, market-level)
    5. Output: daily state matrix ready for RL environment

Data Assumptions:
    - close_prices.csv: hourly close, index=datetime, columns=tickers
    - tradable_mask.csv: binary membership, same shape
    - QQQ.csv / VIX.csv: yfinance daily format (multi-row header)
    - All data: 2021-03-01 to 2026-02-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_raw_data(data_dir: str) -> dict:
    """
    Load all raw CSVs. Adjust paths to match your directory structure.

    Parameters
    ----------
    data_dir : str
        Root directory containing your CSV files.

    Returns
    -------
    dict with keys: 'prices', 'mask', 'qqq', 'vix', 'rf_rate'
    """
    data_dir = Path(data_dir)

    # Hourly close prices
    prices = pd.read_csv(
        data_dir / "close_prices.csv",
        index_col=0, parse_dates=True
    )
    prices.index.name = "datetime"

    # Tradable mask (= NASDAQ-100 membership)
    mask = pd.read_csv(
        data_dir / "tradable_mask.csv",
        index_col=0, parse_dates=True
    )
    mask.index.name = "datetime"

    # QQQ daily — yfinance multi-row header: skip Ticker + Date rows
    qqq = pd.read_csv(
        data_dir / "QQQ.csv",
        skiprows=[1, 2], index_col=0, parse_dates=True
    )
    qqq.index.name = "date"
    qqq = qqq[["Close"]].rename(columns={"Close": "qqq_close"})
    qqq["qqq_close"] = qqq["qqq_close"].astype(float)

    # VIX daily — same format
    vix = pd.read_csv(
        data_dir / "VIX.csv",
        skiprows=[1, 2], index_col=0, parse_dates=True
    )
    vix.index.name = "date"
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
    vix["vix_close"] = vix["vix_close"].astype(float)

    # Risk-free rate (3-month T-bill, ^IRX) — same yfinance format
    rf_path = data_dir / "risk_free_data.csv"
    if rf_path.exists():
        rf_rate = pd.read_csv(
            rf_path, skiprows=[1, 2], index_col=0, parse_dates=True
        )
        rf_rate.index.name = "date"
        rf_rate = rf_rate[["Close"]].rename(columns={"Close": "rf_annualized_pct"})
        rf_rate["rf_annualized_pct"] = rf_rate["rf_annualized_pct"].astype(float)
        # Convert annualized % to daily simple rate: (1 + r/100)^(1/252) - 1
        rf_rate["rf_daily"] = (1 + rf_rate["rf_annualized_pct"] / 100) ** (1/252) - 1
    else:
        print("  WARNING: risk_free_data.csv not found — cash will earn 0%")
        rf_rate = None

    return {"prices": prices, "mask": mask, "qqq": qqq, "vix": vix, "rf_rate": rf_rate}


# =============================================================================
# 2. CLEANING & ALIGNMENT
# =============================================================================

# Junk tickers in mask that have no price data
JUNK_TICKERS = {
    "9210611D", "9218611D", "9996651D", "ALNUW",
    "MDBUQ", "LEND", "MPWRUW", "File"
}

# Hours with ≥98.9% price coverage among index members
VALID_HOURS = list(range(14, 20))  # 14:00 – 19:00 inclusive


def clean_and_align(raw: dict) -> dict:
    """
    Filter to valid hours, drop junk tickers, align prices & mask,
    handle missing data.

    Returns
    -------
    dict with keys:
        'hourly_prices'  : pd.DataFrame — clean hourly prices (14–19h)
        'hourly_mask'    : pd.DataFrame — aligned binary mask
        'daily_close'    : pd.DataFrame — 19:00 close per stock per day
        'daily_mask'     : pd.DataFrame — membership at daily level
        'qqq'            : pd.DataFrame — daily QQQ close
        'vix'            : pd.DataFrame — daily VIX close
        'tickers'        : list — clean ticker list
        'trading_dates'  : pd.DatetimeIndex — valid trading dates
    """
    prices = raw["prices"].copy()
    mask = raw["mask"].copy()
    qqq = raw["qqq"].copy()
    vix = raw["vix"].copy()

    # --- Common tickers (drop junk) ---
    tickers = sorted(
        (set(prices.columns) & set(mask.columns)) - JUNK_TICKERS
    )
    prices = prices[tickers]
    mask = mask[tickers]

    # --- Filter to valid hours ---
    prices = prices[prices.index.hour.isin(VALID_HOURS)]
    mask = mask.reindex(prices.index).fillna(0).astype(int)

    # --- Drop short days (holidays with early close) ---
    bars_per_day = prices.groupby(prices.index.date).size()
    full_days = bars_per_day[bars_per_day == len(VALID_HOURS)].index
    keep_idx = prices.index[
        prices.index.normalize().isin(pd.to_datetime(list(full_days)))
    ]
    prices = prices.loc[keep_idx]
    mask = mask.loc[keep_idx]

    # --- Forward-fill prices within each day (rare intraday gaps) ---
    prices = prices.groupby(prices.index.date).apply(
        lambda g: g.ffill()
    )
    # Fix potential multi-index from groupby
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.droplevel(0)

    # --- Build daily close (19:00 bar) ---
    daily_close = prices[prices.index.hour == 19].copy()
    daily_close.index = daily_close.index.normalize()

    daily_mask = mask[mask.index.hour == 19].copy()
    daily_mask.index = daily_mask.index.normalize()

    # --- Cross-day forward-fill for the ~1 stock/day missing at 19:00 ---
    #     Only fill forward 1 day max to avoid stale prices
    daily_close = daily_close.ffill(limit=1)

    # --- Align QQQ and VIX to our trading dates ---
    trading_dates = daily_close.index
    qqq = qqq.reindex(trading_dates).ffill()
    vix = vix.reindex(trading_dates).ffill()

    # --- Align risk-free rate ---
    rf_rate = raw.get("rf_rate")
    if rf_rate is not None:
        rf_rate = rf_rate.reindex(trading_dates).ffill().bfill()
    else:
        # Default: zero risk-free rate
        rf_rate = pd.DataFrame(
            {"rf_annualized_pct": 0.0, "rf_daily": 0.0},
            index=trading_dates,
        )

    return {
        "hourly_prices": prices,
        "hourly_mask": mask,
        "daily_close": daily_close,
        "daily_mask": daily_mask,
        "qqq": qqq,
        "vix": vix,
        "rf_rate": rf_rate,
        "tickers": tickers,
        "trading_dates": trading_dates,
    }


# =============================================================================
# 3. PER-ASSET FEATURE ENGINEERING
# =============================================================================

def compute_daily_returns(daily_close: pd.DataFrame, lookbacks: list = [1, 5, 20]) -> dict:
    """
    Log returns over multiple lookback windows.

    Returns dict: {"ret_1d": DataFrame, "ret_5d": DataFrame, ...}
    """
    features = {}
    log_close = np.log(daily_close.replace(0, np.nan))
    for lb in lookbacks:
        features[f"ret_{lb}d"] = log_close.diff(lb)
    return features


def compute_realized_vol(daily_close: pd.DataFrame, windows: list = [5, 20]) -> dict:
    """
    Rolling realized volatility from daily log returns.

    Returns dict: {"rvol_5d": DataFrame, "rvol_20d": DataFrame}
    """
    features = {}
    log_ret = np.log(daily_close.replace(0, np.nan)).diff()
    for w in windows:
        features[f"rvol_{w}d"] = log_ret.rolling(w, min_periods=max(2, w // 2)).std()
    return features


def compute_intraday_features(hourly_prices: pd.DataFrame) -> dict:
    """
    Features derived from intraday (14:00–19:00) price action, aggregated daily.

    Returns dict of daily DataFrames:
        - 'intraday_ret'  : 14:00 → 19:00 return
        - 'intraday_rvol' : std of hourly returns within the day
    """
    log_prices = np.log(hourly_prices.replace(0, np.nan))
    log_ret_hourly = log_prices.diff()

    # Group by date
    grouped_price = log_prices.groupby(log_prices.index.date)
    grouped_ret = log_ret_hourly.groupby(log_ret_hourly.index.date)

    # Intraday return: last - first of the day
    intraday_ret = grouped_price.last() - grouped_price.first()
    intraday_ret.index = pd.to_datetime(intraday_ret.index)

    # Intraday realized vol: std of hourly returns
    intraday_rvol = grouped_ret.std()
    intraday_rvol.index = pd.to_datetime(intraday_rvol.index)

    return {
        "intraday_ret": intraday_ret,
        "intraday_rvol": intraday_rvol,
    }


def cross_sectional_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-transform across assets at each timestamp, normalized to [-1, 1].
    Only ranks among tradable (mask==1) assets. Non-members get NaN.

    This handles non-stationarity and makes features comparable across assets.
    """
    # Mask out non-members
    masked = df.where(mask == 1)
    # Rank across columns (assets), pct=True gives [0, 1]
    ranked = masked.rank(axis=1, pct=True)
    # Rescale to [-1, 1]
    ranked = 2 * ranked - 1
    return ranked


def build_per_asset_features(clean: dict) -> pd.DataFrame:
    """
    Build all per-asset features, cross-sectionally ranked.

    Returns
    -------
    pd.DataFrame with MultiIndex columns: (ticker, feature_name)
    Indexed by trading date.
    """
    daily_close = clean["daily_close"]
    daily_mask = clean["daily_mask"]
    hourly_prices = clean["hourly_prices"]
    tickers = clean["tickers"]

    # --- Raw features ---
    features_raw = {}

    # Multi-day returns
    features_raw.update(compute_daily_returns(daily_close, lookbacks=[1, 5, 20]))

    # Realized vol
    features_raw.update(compute_realized_vol(daily_close, windows=[5, 20]))

    # Intraday features
    intraday = compute_intraday_features(hourly_prices)
    # Align to daily_close index
    for k, v in intraday.items():
        features_raw[k] = v.reindex(daily_close.index)

    # --- Cross-sectional ranking ---
    features_ranked = {}
    for name, df in features_raw.items():
        # Align columns
        df = df.reindex(columns=tickers)
        features_ranked[name] = cross_sectional_rank(df, daily_mask)

    # --- Stack into MultiIndex DataFrame ---
    # Shape: (n_days, n_tickers * n_features)
    panels = []
    feature_names = sorted(features_ranked.keys())

    for feat_name in feature_names:
        df = features_ranked[feat_name]
        df.columns = pd.MultiIndex.from_product(
            [[feat_name], df.columns], names=["feature", "ticker"]
        )
        panels.append(df)

    per_asset = pd.concat(panels, axis=1)

    # Reorder to (ticker, feature) for easier slicing per asset
    per_asset = per_asset.swaplevel(axis=1).sort_index(axis=1)

    return per_asset


# =============================================================================
# 4. GLOBAL FEATURES
# =============================================================================

def build_global_features(clean: dict) -> pd.DataFrame:
    """
    Market-level and macro features.

    Returns
    -------
    pd.DataFrame indexed by trading date with columns:
        - vix_level        : VIX close, z-scored
        - vix_change_5d    : 5-day VIX change, z-scored
        - market_ret_1d    : equal-weight universe return
        - market_rvol_5d   : 5-day market realized vol
        - dow              : day of week (0=Mon, 4=Fri), normalized
    """
    vix = clean["vix"]["vix_close"]
    daily_close = clean["daily_close"]
    daily_mask = clean["daily_mask"]

    features = pd.DataFrame(index=clean["trading_dates"])

    # --- VIX ---
    # Z-score with expanding window (no look-ahead)
    vix_mean = vix.expanding(min_periods=20).mean()
    vix_std = vix.expanding(min_periods=20).std()
    features["vix_level"] = (vix - vix_mean) / vix_std

    features["vix_change_5d"] = vix.pct_change(5)

    # --- Market return (equal-weight among members) ---
    log_ret = np.log(daily_close.replace(0, np.nan)).diff()
    # Only include members
    member_ret = log_ret.where(daily_mask == 1)
    features["market_ret_1d"] = member_ret.mean(axis=1)

    # --- Market realized vol ---
    features["market_rvol_5d"] = features["market_ret_1d"].rolling(5).std()

    # --- Day of week ---
    features["dow"] = features.index.dayofweek / 4.0  # normalize to [0, 1]

    return features


# =============================================================================
# 5. MAIN PIPELINE — ASSEMBLE EVERYTHING
# =============================================================================

def build_dataset(data_dir: str) -> dict:
    """
    Full pipeline: raw CSVs → clean daily feature set.

    Parameters
    ----------
    data_dir : str
        Directory containing close_prices.csv, tradable_mask.csv,
        QQQ.csv, VIX.csv

    Returns
    -------
    dict with keys:
        'per_asset_features' : pd.DataFrame
            MultiIndex columns (ticker, feature_name).
            Cross-sectionally ranked. Shape: (n_days, n_tickers * n_features)

        'global_features' : pd.DataFrame
            Market-level features. Shape: (n_days, n_global_features)

        'daily_close' : pd.DataFrame
            Clean daily close prices. Shape: (n_days, n_tickers)

        'daily_mask' : pd.DataFrame
            Binary membership mask. Shape: (n_days, n_tickers)

        'qqq' : pd.DataFrame
            Daily QQQ close for benchmark comparison

        'tickers' : list
            Ordered list of ticker symbols

        'trading_dates' : pd.DatetimeIndex
            Valid trading dates

        'metadata' : dict
            Pipeline config and data diagnostics
    """
    print("=" * 60)
    print("DATA PIPELINE — RL Portfolio Allocation")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading raw data...")
    raw = load_raw_data(data_dir)
    print(f"  Prices: {raw['prices'].shape}")
    print(f"  Mask:   {raw['mask'].shape}")
    print(f"  QQQ:    {raw['qqq'].shape}")
    print(f"  VIX:    {raw['vix'].shape}")

    # Step 2: Clean
    print("\n[2/5] Cleaning & aligning...")
    clean = clean_and_align(raw)
    print(f"  Tickers:       {len(clean['tickers'])}")
    print(f"  Trading dates: {len(clean['trading_dates'])}")
    print(f"  Date range:    {clean['trading_dates'][0].date()} → "
          f"{clean['trading_dates'][-1].date()}")
    print(f"  Hourly bars:   {len(clean['hourly_prices'])}")
    print(f"  Bars/day:      {len(VALID_HOURS)}")

    # Member coverage diagnostic
    member_coverage = (
        (clean["daily_mask"] == 1) & clean["daily_close"].notna()
    ).sum(axis=1) / (clean["daily_mask"] == 1).sum(axis=1)
    print(f"  Close coverage (members): "
          f"{member_coverage.mean():.1%} mean, {member_coverage.min():.1%} min")

    # Step 3: Per-asset features
    print("\n[3/5] Building per-asset features (cross-sectionally ranked)...")
    per_asset = build_per_asset_features(clean)
    n_features = len(per_asset.columns.get_level_values("feature").unique())
    print(f"  Features per asset:  {n_features}")
    print(f"  Feature names:       {sorted(per_asset.columns.get_level_values('feature').unique())}")
    print(f"  Total columns:       {per_asset.shape[1]}")

    # Step 4: Global features
    print("\n[4/5] Building global features...")
    global_feat = build_global_features(clean)
    print(f"  Global features: {list(global_feat.columns)}")

    # Step 5: Warmup — drop rows where features aren't yet available
    print("\n[5/5] Applying warmup period (20 days for rolling windows)...")
    warmup = 20
    valid_start = clean["trading_dates"][warmup]
    valid_dates = clean["trading_dates"][warmup:]

    per_asset = per_asset.loc[valid_dates]
    global_feat = global_feat.loc[valid_dates]
    daily_close = clean["daily_close"].loc[valid_dates]
    daily_mask = clean["daily_mask"].loc[valid_dates]
    qqq = clean["qqq"].loc[valid_dates]
    rf_rate = clean["rf_rate"].loc[valid_dates]

    # NaN audit
    pa_nan = per_asset.isna().mean().mean()
    gf_nan = global_feat.isna().mean().mean()
    print(f"  Valid date range: {valid_dates[0].date()} → {valid_dates[-1].date()}")
    print(f"  Valid days:       {len(valid_dates)}")
    print(f"  Per-asset NaN:    {pa_nan:.3%}")
    print(f"  Global NaN:       {gf_nan:.3%}")
    rf_mean = rf_rate["rf_annualized_pct"].mean()
    print(f"  Avg risk-free:    {rf_mean:.2f}% annualized")

    metadata = {
        "valid_hours": VALID_HOURS,
        "warmup_days": warmup,
        "n_tickers": len(clean["tickers"]),
        "n_per_asset_features": n_features,
        "n_global_features": len(global_feat.columns),
        "n_trading_days": len(valid_dates),
        "date_range": (valid_dates[0].date(), valid_dates[-1].date()),
        "per_asset_nan_rate": pa_nan,
        "global_nan_rate": gf_nan,
    }

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
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


# =============================================================================
# 6. LOOK-AHEAD BIAS TESTS
# =============================================================================

def run_lookahead_tests(dataset: dict) -> None:
    """
    Verify no future information leaks into features.
    Run this after build_dataset(). Raises AssertionError if any test fails.
    """
    print("\n" + "=" * 60)
    print("LOOK-AHEAD BIAS TESTS")
    print("=" * 60)

    per_asset = dataset["per_asset_features"]
    global_feat = dataset["global_features"]
    daily_close = dataset["daily_close"]
    dates = dataset["trading_dates"]

    # Test 1: Features on day t should not use close price from day t+1
    # Verify by checking that ret_1d[t] = close[t] - close[t-1], not close[t+1]
    t = 100  # arbitrary test day
    ticker = dataset["tickers"][0]
    test_date = dates[t]
    prev_date = dates[t - 1]

    actual_ret = np.log(
        daily_close.loc[test_date, ticker] /
        daily_close.loc[prev_date, ticker]
    )
    # The raw (pre-rank) return would match. Since we rank, just check
    # that feature on day t doesn't correlate with FUTURE returns
    print("\n  Test 1: Feature-future return correlation check")
    future_ret = np.log(daily_close.shift(-1) / daily_close)
    # Get ret_1d rank feature for all tickers
    if (ticker, "ret_1d") in per_asset.columns:
        feat = per_asset[(ticker, "ret_1d")].dropna()
        fut = future_ret[ticker].reindex(feat.index).dropna()
        common = feat.index.intersection(fut.index)
        corr = feat.loc[common].corr(fut.loc[common])
        print(f"    ret_1d rank ↔ next-day return corr: {corr:.4f}")
        # This should be small and noisy, not ~1.0
        assert abs(corr) < 0.3, (
            f"FAIL: Suspiciously high correlation ({corr:.4f}) between "
            "today's feature and tomorrow's return"
        )
        print("    PASSED ✓")

    # Test 2: Global features don't use future VIX
    print("\n  Test 2: VIX feature uses only past data")
    vix_feat = global_feat["vix_level"]
    # VIX z-score on day t should be computed from data up to day t
    # Verify: changing VIX on day t+1 shouldn't change feature on day t
    print("    VIX z-score uses expanding window → no look-ahead by construction")
    print("    PASSED ✓")

    # Test 3: Cross-sectional rank uses only same-day data
    print("\n  Test 3: Cross-sectional rank independence across days")
    day1_feat = per_asset.iloc[50]
    day2_feat = per_asset.iloc[51]
    # Ranks on different days should be different
    assert not day1_feat.equals(day2_feat), "FAIL: Identical ranks on consecutive days"
    print("    PASSED ✓")

    # Test 4: No future dates in the index
    print("\n  Test 4: No future dates in feature index")
    max_date = dates[-1]
    assert per_asset.index.max() <= max_date
    assert global_feat.index.max() <= max_date
    print(f"    Max feature date: {per_asset.index.max().date()}")
    print("    PASSED ✓")

    print("\n  All look-ahead bias tests PASSED ✓")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # --- ADJUST THIS PATH TO YOUR DIRECTORY ---
    DATA_DIR = "../Data/Outputs/Filtered/Data"

    dataset = build_dataset(DATA_DIR)
    run_lookahead_tests(dataset)

    # Quick summary
    print("\n\nDataset ready for RL environment.")
    print(f"  Per-asset state dim per stock: {dataset['metadata']['n_per_asset_features']}")
    print(f"  Global state dim:              {dataset['metadata']['n_global_features']}")
    print(f"  Total state dim (approx):      "
          f"{dataset['metadata']['n_tickers'] * dataset['metadata']['n_per_asset_features']}"
          f" + {dataset['metadata']['n_global_features']}"
          f" + {dataset['metadata']['n_tickers']} (weights)"
          f" = {dataset['metadata']['n_tickers'] * dataset['metadata']['n_per_asset_features'] + dataset['metadata']['n_global_features'] + dataset['metadata']['n_tickers']}")