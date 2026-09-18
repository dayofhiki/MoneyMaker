from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_flatfile_credentials
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, ExecutionScenario
from .flatfiles import MassiveFlatFileStore, MassiveFlatFilesClient, STOCKS_MINUTE_PREFIX
from .market_calendar import regular_session_bounds


MINUTE_MS = 60_000
FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")
HORIZONS = (1, 2, 5, 10, 15, 30)
STATIC_CROSS_FEATURES = (
    "rvol_cumulative_20d",
    "rvol_5m_20d",
    "premarket_return_pct",
    "premarket_volume",
    "premarket_high_return_pct",
    "premarket_high_distance_pct",
    "regular_open_gap_pct",
)


def _scenario_net_pct(
    entry_signal: pd.Series,
    raw_exit: pd.Series,
    scenario: ExecutionScenario,
) -> pd.Series:
    entry = pd.to_numeric(entry_signal, errors="coerce").to_numpy(dtype=float)
    exit_signal = pd.to_numeric(raw_exit, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(entry) & np.isfinite(exit_signal) & (entry > 0) & (exit_signal > 0)
    result = np.full(len(entry), np.nan, dtype=float)
    if not valid.any():
        return pd.Series(result, index=entry_signal.index, dtype=float)

    e = entry[valid]
    x = exit_signal[valid]
    entry_half = np.maximum(
        e * scenario.half_spread_bps / 10_000.0,
        scenario.min_half_spread_cents / 100.0,
    )
    exit_half = np.maximum(
        x * scenario.half_spread_bps / 10_000.0,
        scenario.min_half_spread_cents / 100.0,
    )
    buy = e * (1.0 + scenario.slippage_bps / 10_000.0) + entry_half
    sell = np.maximum(
        x * (1.0 - scenario.slippage_bps / 10_000.0) - exit_half,
        0.0,
    )
    sell *= 1.0 - scenario.sell_fee_bps / 10_000.0
    result[valid] = (sell / buy - 1.0) * 100.0
    return pd.Series(result, index=entry_signal.index, dtype=float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    )
    return result.where(pd.to_numeric(denominator, errors="coerce").gt(0))


def _iwm_context(iwm_bars: pd.DataFrame, grid_t: np.ndarray) -> pd.DataFrame:
    grid = pd.DataFrame({"t": grid_t})
    if iwm_bars.empty:
        for column in (
            "iwm_return_since_open_pct",
            "iwm_return_15m_pct",
            "iwm_volatility_15m_pct",
        ):
            grid[column] = np.nan
        return grid

    cols = iwm_bars.loc[:, ["t", "o", "c"]].drop_duplicates("t", keep="last")
    grid = grid.merge(cols, on="t", how="left")
    first_open = pd.to_numeric(grid["o"], errors="coerce").dropna()
    open_price = float(first_open.iloc[0]) if not first_open.empty else np.nan
    close = pd.to_numeric(grid["c"], errors="coerce")
    grid["iwm_return_since_open_pct"] = (
        (close / open_price - 1.0) * 100.0 if np.isfinite(open_price) else np.nan
    )
    grid["iwm_return_15m_pct"] = (close / close.shift(15) - 1.0) * 100.0
    one_minute = close / close.shift(1) - 1.0
    grid["iwm_volatility_15m_pct"] = one_minute.rolling(
        15, min_periods=2
    ).std(ddof=1) * 100.0
    return grid.loc[
        :,
        [
            "t",
            "iwm_return_since_open_pct",
            "iwm_return_15m_pct",
            "iwm_volatility_15m_pct",
        ],
    ]


def _future_extreme(series: pd.Series, minutes: int, mode: str) -> pd.Series:
    shifted = [pd.to_numeric(series.shift(-step), errors="coerce") for step in range(1, minutes + 1)]
    matrix = pd.concat(shifted, axis=1)
    if mode == "max":
        return matrix.max(axis=1, skipna=True)
    if mode == "min":
        return matrix.min(axis=1, skipna=True)
    raise ValueError(f"unsupported future extreme mode: {mode}")


def _build_ticker_states(
    ticker_bars: pd.DataFrame,
    *,
    ticker: str,
    trading_day: str,
    first_10_timestamp_ms: int,
    previous_close: float,
    regular_open_ms: int,
    regular_close_ms: int,
    day_crossings: list[int],
    iwm_context: pd.DataFrame,
    cross_row: pd.Series,
) -> pd.DataFrame:
    grid_t = np.arange(regular_open_ms, regular_close_ms, MINUTE_MS, dtype=np.int64)
    raw = (
        ticker_bars.loc[
            (ticker_bars["t"] >= regular_open_ms)
            & (ticker_bars["t"] < regular_close_ms),
            ["t", "o", "h", "l", "c", "v", "n"],
        ]
        .drop_duplicates("t", keep="last")
        .sort_values("t")
    )
    grid = pd.DataFrame({"t": grid_t}).merge(raw, on="t", how="left")
    grid = grid.merge(iwm_context, on="t", how="left")
    grid["observed"] = pd.to_numeric(grid["c"], errors="coerce").notna()

    for column in ("o", "h", "l", "c", "v", "n"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")

    close = grid["c"]
    high = grid["h"]
    low = grid["l"]
    volume = grid["v"].fillna(0.0)
    transactions = grid["n"].fillna(0.0)

    grid["ticker"] = ticker
    grid["trading_day"] = trading_day
    grid["previous_close"] = previous_close
    grid["first_10_timestamp_ms"] = first_10_timestamp_ms
    grid["minutes_since_10pct_cross"] = (
        grid["t"] - first_10_timestamp_ms
    ) / MINUTE_MS
    grid["minutes_from_regular_open"] = (grid["t"] - regular_open_ms) / MINUTE_MS
    grid["return_from_previous_close_pct"] = (
        (close / previous_close - 1.0) * 100.0 if previous_close > 0 else np.nan
    )

    regular_dollar = volume * close.fillna(0.0)
    cum_volume = volume.cumsum()
    cum_dollar = regular_dollar.cumsum()
    grid["regular_vwap_proxy"] = _safe_ratio(cum_dollar, cum_volume)
    grid["regular_vwap_distance_pct"] = (
        close / grid["regular_vwap_proxy"] - 1.0
    ) * 100.0
    grid["above_regular_vwap"] = grid["regular_vwap_distance_pct"].ge(0)

    running_hod = high.cummax()
    grid["running_hod"] = running_hod
    grid["running_hod_return_pct"] = (
        (running_hod / previous_close - 1.0) * 100.0
        if previous_close > 0
        else np.nan
    )
    grid["hod_distance_pct"] = (close / running_hod - 1.0) * 100.0

    after_cross = grid["t"].ge(first_10_timestamp_ms)
    running_low = low.where(after_cross).cummin()
    grid["running_low_since_10"] = running_low
    grid["rebound_from_running_low_pct"] = (close / running_low - 1.0) * 100.0

    prior_hod = running_hod.shift(1)
    tolerance = np.maximum(running_hod.abs() * 1e-12, 1e-12)
    grid["new_hod"] = high.notna() & (
        prior_hod.isna() | high.gt(prior_hod + tolerance)
    )
    hod_timestamp = grid["t"].where(grid["new_hod"]).ffill()
    grid["minutes_since_hod"] = (grid["t"] - hod_timestamp) / MINUTE_MS

    for minutes in (1, 3, 5, 10, 15):
        grid[f"trailing_return_{minutes}m_pct"] = (
            close / close.shift(minutes) - 1.0
        ) * 100.0

    one_minute_return = close / close.shift(1) - 1.0
    grid["volatility_5m_pct"] = one_minute_return.rolling(
        5, min_periods=2
    ).std(ddof=1) * 100.0
    grid["volatility_15m_pct"] = one_minute_return.rolling(
        15, min_periods=2
    ).std(ddof=1) * 100.0

    bar_range = high - low
    grid["bar_range_pct"] = _safe_ratio(bar_range, close) * 100.0
    grid["bar_body_pct"] = (close / grid["o"] - 1.0) * 100.0
    grid["bar_close_location"] = _safe_ratio(close - low, bar_range).fillna(0.5)
    range_3 = grid["bar_range_pct"].rolling(3, min_periods=1).mean()
    range_15 = grid["bar_range_pct"].rolling(15, min_periods=3).mean()
    grid["range_contraction_3m_vs_15m"] = _safe_ratio(range_3, range_15)

    grid["volume_1m"] = volume
    grid["volume_3m"] = volume.rolling(3, min_periods=1).sum()
    grid["volume_5m"] = volume.rolling(5, min_periods=1).sum()
    prior20_avg_for_1m = volume.shift(1).rolling(20, min_periods=5).mean()
    prior20_avg_for_5m = volume.shift(5).rolling(20, min_periods=5).mean()
    grid["volume_accel_1m_vs_prior20m"] = _safe_ratio(
        volume,
        prior20_avg_for_1m,
    )
    grid["volume_accel_5m_vs_prior20m"] = _safe_ratio(
        grid["volume_5m"] / 5.0,
        prior20_avg_for_5m,
    )
    rolling_3_avg = grid["volume_3m"] / 3.0
    post_cross_peak_3m = rolling_3_avg.where(after_cross).cummax()
    grid["volume_3m_vs_postcross_peak"] = _safe_ratio(
        rolling_3_avg,
        post_cross_peak_3m,
    )
    grid["transactions_1m"] = transactions
    grid["transactions_5m"] = transactions.rolling(5, min_periods=1).sum()
    grid["dollar_volume_5m"] = regular_dollar.rolling(5, min_periods=1).sum()
    grid["active_minute_fraction_15m"] = (
        grid["observed"].astype(float).rolling(15, min_periods=1).sum() / 15.0
    )

    grid["recent_deepest_pullback_15m_pct"] = grid["hod_distance_pct"].rolling(
        15, min_periods=1
    ).min()
    prior_5m_high = high.shift(1).rolling(5, min_periods=1).max()
    grid["reclaim_prior_5m_high_after_pullback"] = (
        close.ge(prior_5m_high)
        & grid["recent_deepest_pullback_15m_pct"].le(-1.5)
    )

    runners_so_far: list[int] = []
    runners_last_30m: list[int] = []
    for timestamp in grid["t"].astype("int64"):
        right = bisect_right(day_crossings, int(timestamp))
        left = bisect_left(day_crossings, int(timestamp) - 30 * MINUTE_MS)
        runners_so_far.append(right)
        runners_last_30m.append(right - left)
    grid["runners_10pct_so_far"] = runners_so_far
    grid["runners_10pct_last_30m"] = runners_last_30m

    for feature in STATIC_CROSS_FEATURES:
        value = cross_row.get(feature)
        grid[f"cross_{feature}"] = pd.to_numeric(
            pd.Series([value]), errors="coerce"
        ).iloc[0]

    phase = np.full(len(grid), "other", dtype=object)
    failure = grid["hod_distance_pct"].le(-5.0) & ~grid["above_regular_vwap"]
    consolidation = (
        grid["hod_distance_pct"].ge(-3.0)
        & grid["above_regular_vwap"]
        & grid["range_contraction_3m_vs_15m"].le(0.65)
    )
    pullback_above = (
        grid["hod_distance_pct"].le(-1.0) & grid["above_regular_vwap"]
    )
    pullback_below = (
        grid["hod_distance_pct"].le(-1.0) & ~grid["above_regular_vwap"]
    )
    impulse = (
        grid["new_hod"]
        & grid["trailing_return_3m_pct"].gt(0)
        & grid["above_regular_vwap"]
    )
    reclaim = grid["reclaim_prior_5m_high_after_pullback"]

    phase[consolidation.fillna(False)] = "consolidation"
    phase[pullback_below.fillna(False)] = "pullback_below_vwap"
    phase[pullback_above.fillna(False)] = "pullback_above_vwap"
    phase[impulse.fillna(False)] = "impulse"
    phase[reclaim.fillna(False)] = "reclaim"
    phase[failure.fillna(False)] = "failure"
    grid["phase"] = phase

    grid["entry_timestamp_ms"] = grid["t"] + MINUTE_MS
    grid["entry_price"] = grid["o"].shift(-1)

    for horizon in HORIZONS:
        exit_price = close.shift(-horizon)
        gross = (exit_price / grid["entry_price"] - 1.0) * 100.0
        grid[f"buy_return_{horizon}m_pct"] = gross
        for scenario in DEFAULT_EXECUTION_SCENARIOS:
            grid[
                f"buy_return_{horizon}m_{scenario.name}_net_return_pct"
            ] = _scenario_net_pct(grid["entry_price"], exit_price, scenario)

    future_high_15 = _future_extreme(high, 15, "max")
    future_low_15 = _future_extreme(low, 15, "min")
    grid["buy_mfe_15m_pct"] = (
        future_high_15 / grid["entry_price"] - 1.0
    ) * 100.0
    grid["buy_mae_15m_pct"] = (
        future_low_15 / grid["entry_price"] - 1.0
    ) * 100.0

    state_rows = grid.loc[
        after_cross & grid["observed"]
    ].copy()
    return state_rows.reset_index(drop=True)


def build_state_panel(
    events: pd.DataFrame,
    store: MassiveFlatFileStore,
) -> pd.DataFrame:
    required = {
        "trading_day",
        "ticker",
        "timestamp_ms",
        "threshold_pct",
        "previous_close",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"event dataset missing state-panel columns: {sorted(missing)}")

    work = events.copy()
    work["threshold_pct"] = pd.to_numeric(work["threshold_pct"], errors="coerce")
    work["timestamp_ms"] = pd.to_numeric(work["timestamp_ms"], errors="coerce")
    first10 = (
        work.loc[work["threshold_pct"].eq(10.0)]
        .sort_values(["trading_day", "ticker", "timestamp_ms"])
        .drop_duplicates(["trading_day", "ticker"], keep="first")
    )
    if first10.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for trading_day, day_cross_rows in first10.groupby("trading_day", sort=True):
        day = date.fromisoformat(str(trading_day))
        bounds = regular_session_bounds(day)
        if bounds is None:
            continue
        regular_open_ms = int(bounds[0].timestamp() * 1000)
        regular_close_ms = int(bounds[1].timestamp() * 1000)
        tickers = set(day_cross_rows["ticker"].astype(str).str.upper()) | {"IWM"}
        bars = store.minute_aggregates(day, tickers=tickers)
        if bars.empty:
            continue

        grid_t = np.arange(
            regular_open_ms,
            regular_close_ms,
            MINUTE_MS,
            dtype=np.int64,
        )
        iwm_bars = bars.loc[bars["ticker"].astype(str).str.upper().eq("IWM")]
        iwm_context = _iwm_context(iwm_bars, grid_t)
        day_crossings = sorted(
            day_cross_rows["timestamp_ms"].dropna().astype("int64").tolist()
        )

        for row in day_cross_rows.itertuples(index=False):
            ticker = str(row.ticker).upper()
            ticker_bars = bars.loc[
                bars["ticker"].astype(str).str.upper().eq(ticker)
            ].copy()
            if ticker_bars.empty:
                continue
            cross_row = pd.Series(row._asdict())
            states = _build_ticker_states(
                ticker_bars,
                ticker=ticker,
                trading_day=str(trading_day),
                first_10_timestamp_ms=int(row.timestamp_ms),
                previous_close=float(row.previous_close),
                regular_open_ms=regular_open_ms,
                regular_close_ms=regular_close_ms,
                day_crossings=day_crossings,
                iwm_context=iwm_context,
                cross_row=cross_row,
            )
            if not states.empty:
                frames.append(states)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(
        ["trading_day", "ticker", "t"],
        kind="stable",
    ).reset_index(drop=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_panel",
        description="Build minute-by-minute runner state panels after the first +10% cross.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = pd.read_parquet(args.input)
    access_key, secret_key = load_flatfile_credentials()
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    panel = build_state_panel(events, store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False, compression="zstd")
    print(
        f"Built {len(panel)} state rows across "
        f"{panel[['trading_day', 'ticker']].drop_duplicates().shape[0] if not panel.empty else 0} "
        f"runner episodes; Flat File stats: {store.stats.to_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
