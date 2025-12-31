"""Phase 1: BTC data download (Bybit hourly by default), cleaning, feature generation, and packaging."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal

import ccxt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

matplotlib.use("Agg")

DATA_DIR = Path("data")
NOTEBOOKS_DIR = Path("notebooks")
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
ROLLING_WINDOWS = [1, 7, 14, 21, 28]
# Переход на минутки с сессиями высокой волатильности
HORIZON_MINUTES = [10, 15, 30]
# Базовые пороги подняты (0.8–2.0%) с учётом комиссий/спреда
TARGET_THRESH_MAP = {
    10: [0.008, 0.012],
    15: [0.012, 0.018],
    30: [0.020, 0.030],
}
SESSION_CFG = {
    "prebuffer_min": 90,
    "session_len_min": 120,
    "std_window_min": 10,
    "vol_thresh_candidates": [1.5, 2.0, 2.5, 3.0],
    "target_sessions_per_month": 30,
    "zscore_window_days": 7,
}


@dataclass
class DownloadConfig:
    symbol: str = "BTC/USDT:USDT"  # Bybit perp
    start_date: datetime = datetime(2017, 1, 1)
    end_date: datetime = datetime.now()
    interval: str = "1m"
    source: Literal["yfinance", "bybit"] = "bybit"


def ensure_dirs() -> None:
    for path in (DATA_DIR, NOTEBOOKS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def infer_steps_per_day(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    deltas = pd.Series(index).diff().dropna().dt.total_seconds()
    median_sec = deltas.median()
    if not np.isfinite(median_sec) or median_sec <= 0:
        return 1
    return max(1, int(round(86400 / median_sec)))


def download_btc_data(cfg: DownloadConfig) -> pd.DataFrame:
    if cfg.source == "yfinance":
        df = yf.download(
            cfg.symbol,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.interval,
            progress=False,
        )
        if df.empty:
            raise RuntimeError("Не удалось загрузить данные через yfinance.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(levels[0]) for levels in df.columns]
        df = df.rename(columns=lambda c: str(c).replace(" ", "_"))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().drop_duplicates()
        keep_cols = [col for col in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if col in df.columns]
        df = df[keep_cols]
        return df.apply(pd.to_numeric, errors="coerce")

    if cfg.source == "bybit":
        timeframe = cfg.interval  # e.g. "1m"
        ex = ccxt.bybit({"enableRateLimit": True})
        ex.load_markets()
        since_ms = int(cfg.start_date.timestamp() * 1000)
        end_ms = int(cfg.end_date.timestamp() * 1000)
        limit = 1000  # Bybit обычно разрешает до 1000 свечей за запрос для 1m
        all_rows = []
        step_ms = ex.parse_timeframe(timeframe) * 1000
        total_bars = max(1, int((end_ms - since_ms) // step_ms))
        total_batches = max(1, (total_bars + limit - 1) // limit)
        batch_idx = 0
        while since_ms < end_ms:
            batch = ex.fetch_ohlcv(cfg.symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            since_ms = last_ts + step_ms
            batch_idx += 1
            pct = min(1.0, batch_idx / total_batches)
            print(f"Download batches {batch_idx}/{total_batches} [{pct:5.1%}]", end="\r", flush=True)
            if last_ts >= end_ms:
                break
        if batch_idx > 0:
            print()
        if not all_rows:
            raise RuntimeError("Не удалось загрузить данные через Bybit (ccxt).")
        df = pd.DataFrame(all_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index().drop_duplicates()
        return df.apply(pd.to_numeric, errors="coerce")

    raise ValueError(f"Unsupported source: {cfg.source}")


def plot_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(df.index, df["Close"])
    axes[0, 0].set_title("Цена закрытия (лог шкала)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df.index, df["Volume"] / 1e9, alpha=0.6)
    axes[0, 1].set_title("Объем (млрд)")
    axes[0, 1].grid(True, alpha=0.3)

    returns = df["Close"].pct_change().dropna()
    axes[1, 0].hist(returns, bins=100, alpha=0.7)
    axes[1, 0].set_title("Распределение доходностей")
    axes[1, 0].axvline(returns.mean(), color="red", linestyle="--")
    axes[1, 0].grid(True, alpha=0.3)

    missing = df.isnull()
    axes[1, 1].imshow(missing.T, aspect="auto", cmap="viridis")
    axes[1, 1].set_title("Пропуски")
    axes[1, 1].set_xlabel("Время")
    axes[1, 1].set_ylabel("Колонки")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("=" * 50)
    print("СТАТИСТИКА RAW:")
    print("=" * 50)
    print(f"Записей: {len(df)}")
    print(f"Период: {df.index[0].date()} - {df.index[-1].date()}")
    print(f"Пропуски:\n{df.isnull().sum()}")
    print(df[["Open", "High", "Low", "Close", "Volume"]].describe())


def handle_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    median_val = df[column].median()
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median_val, df[column])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Пропуски до очистки:")
    print(df.isnull().sum())
    cleaned = df.copy()
    cleaned = cleaned.ffill().bfill()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        cleaned = handle_outliers_iqr(cleaned, col)
    return cleaned


def plot_cleaning(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(raw_df.index, raw_df["Close"], alpha=0.5, label="Raw")
    axes[0, 0].plot(cleaned_df.index, cleaned_df["Close"], alpha=0.8, label="Clean")
    axes[0, 0].set_title("Цена до/после очистки")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    sns.boxplot(data=pd.DataFrame({"Raw": raw_df["Close"], "Clean": cleaned_df["Close"]}), ax=axes[0, 1])
    axes[0, 1].set_title("Boxplot Close")

    missing_after = cleaned_df.isnull().sum()
    axes[1, 0].bar(missing_after.index, missing_after.values)
    axes[1, 0].set_title("Пропуски после")
    axes[1, 0].tick_params(axis="x", rotation=45)

    r_before = raw_df["Close"].pct_change().dropna()
    r_after = cleaned_df["Close"].pct_change().dropna()
    axes[1, 1].hist(r_before, bins=100, alpha=0.5, label="До", density=True)
    axes[1, 1].hist(r_after, bins=100, alpha=0.5, label="После", density=True)
    axes[1, 1].set_title("Распределение доходностей")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_split_dates(data: pd.DataFrame, train_ratio: float = 0.85, val_ratio: float = 0.1) -> Dict[str, pd.DatetimeIndex]:
    total_len = len(data)
    train_idx = int(total_len * train_ratio)
    val_idx = int(total_len * (train_ratio + val_ratio))
    return {"train": data.index[:train_idx], "val": data.index[train_idx:val_idx], "test": data.index[val_idx:]}


def plot_splits(df: pd.DataFrame, split_dates: Dict[str, pd.DatetimeIndex], output_path: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].plot(df.index, df["Close"], color="gray", alpha=0.3)
    masks = {
        "Train": (df.index >= split_dates["train"][0]) & (df.index <= split_dates["train"][-1]),
        "Validation": (df.index >= split_dates["val"][0]) & (df.index <= split_dates["val"][-1]),
        "Test": (df.index >= split_dates["test"][0]) & (df.index <= split_dates["test"][-1]),
    }
    colors = {"Train": "green", "Validation": "orange", "Test": "red"}
    for name, mask in masks.items():
        axes[0].fill_between(df.index, 0, df["Close"].max(), where=mask, alpha=0.3, color=colors[name], label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("Разделение Train/Val/Test")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    stats = []
    for name, mask in masks.items():
        d = df[mask]
        ret = d["Close"].pct_change().dropna()
        stats.append(
            {
                "Period": name,
                "Days": len(d),
                "Start": d.index[0].date(),
                "End": d.index[-1].date(),
                "Mean Return": ret.mean() * 100,
                "Volatility": ret.std() * 100,
                "Sharpe": (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0,
                "Max Drawdown": (d["Close"] / d["Close"].cummax() - 1).min() * 100,
            }
        )
    stats_df = pd.DataFrame(stats)
    melted = stats_df.melt(id_vars=["Period"], value_vars=["Mean Return", "Volatility", "Sharpe", "Max Drawdown"])
    sns.barplot(data=melted, x="Period", y="value", hue="variable", ax=axes[1])
    axes[1].set_title("Статистика по периодам")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return stats_df


def create_basic_features(data: pd.DataFrame, steps_per_day: int = 1) -> pd.DataFrame:
    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    def win(days: int) -> int:
        return max(1, int(days * steps_per_day))

    # Volume (минимально необходимые)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window=win(20)).mean()
    df["log_volume"] = np.log(df["Volume"].replace(0, np.nan)).ffill().bfill()
    df["log_volume_diff"] = df["log_volume"].diff()
    # OBV и производные
    sign = np.sign(df["Close"].diff().fillna(0))
    df["obv"] = (sign * df["Volume"]).cumsum()
    df["obv_delta"] = df["obv"].diff()

    df["sma_7"] = df["Close"].rolling(window=win(7)).mean()
    df["sma_21"] = df["Close"].rolling(window=win(21)).mean()
    df["sma_ratio"] = df["sma_7"] / df["sma_21"]

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=win(14)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=win(14)).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr_14"] = true_range.rolling(window=win(14)).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]
    df["atr_pct_7"] = true_range.rolling(window=win(7)).mean() / df["Close"]
    df["atr_pct_21"] = true_range.rolling(window=win(21)).mean() / df["Close"]

    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["body_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    df["upper_wick_pct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    df["lower_wick_pct"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]
    df["range_vol_14"] = df["range_pct"].rolling(window=win(14)).std(ddof=0)

    exp1 = df["Close"].ewm(span=max(1, 12 * steps_per_day), adjust=False).mean()
    exp2 = df["Close"].ewm(span=max(1, 26 * steps_per_day), adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=max(1, 9 * steps_per_day), adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    for window in ROLLING_WINDOWS:
        bar_window = win(window)
        df[f"logret_mean_{window}"] = df["log_returns"].rolling(window=bar_window).mean()
        df[f"logret_std_{window}"] = df["log_returns"].rolling(window=bar_window).std(ddof=0)
        df[f"logret_cum_{window}"] = df["log_returns"].rolling(window=bar_window).sum()
        if window == 21:
            df["logret_zscore_21"] = (df["log_returns"] - df[f"logret_mean_{window}"]) / df[f"logret_std_{window}"]
    # Дополнительные формы распределения
    df["logret_skew_14"] = df["log_returns"].rolling(window=win(14)).skew()
    df["logret_kurt_14"] = df["log_returns"].rolling(window=win(14)).kurt()
    # Цикличность часа
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Цели в часах: 6h, 12h, 24h (суточная аналогия)
    for n_min in HORIZON_MINUTES:
        step_shift = max(1, int(n_min * steps_per_day / 1440))
        future_price = df["Close"].shift(-step_shift)
        default_thr = TARGET_THRESH_MAP.get(n_min, [0.01])[0]
        df[f"target_{n_min}m"] = ((future_price / df["Close"] - 1) > default_thr).astype(int)
        for thr in TARGET_THRESH_MAP.get(n_min, []):
            df[f"target_{n_min}m_{int(thr*10000)}bps"] = ((future_price / df["Close"] - 1) > thr).astype(int)

    df = df.dropna()
    return df


def plot_features(df: pd.DataFrame, output_path: Path) -> None:
    feature_cols = [
        "returns",
        "log_returns",
        "sma_ratio",
        "rsi_14",
        "atr_pct",
        "atr_pct_7",
        "atr_pct_21",
        "macd",
        "macd_hist",
        "logret_std_1",
        "logret_std_7",
        "logret_std_14",
        "logret_std_21",
        "logret_std_28",
        "logret_mean_7",
        "logret_cum_7",
        "logret_skew_14",
        "logret_kurt_14",
        "range_pct",
        "range_vol_14",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "hour_sin",
        "hour_cos",
    ]
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.index, df["Close"], label="Цена")
    ax1.set_title("Цена закрытия", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df.index, df["volume_ratio"])
    ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5)
    ax2.set_title("Volume Ratio", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df.index, df["Close"], alpha=0.5, label="Цена")
    ax3.plot(df.index, df["sma_7"], label="SMA 7d")
    ax3.plot(df.index, df["sma_21"], label="SMA 21d")
    ax3.set_title("Скользящие средние", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df.index, df["rsi_14"])
    ax4.axhline(y=70, color="r", linestyle="--", alpha=0.5)
    ax4.axhline(y=30, color="g", linestyle="--", alpha=0.5)
    ax4.set_title("RSI (14)", fontsize=12)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df.index, df["macd"], label="MACD")
    ax5.plot(df.index, df["macd_signal"], label="Signal")
    ax5.bar(df.index, df["macd_hist"], alpha=0.3, label="Histogram")
    ax5.set_title("MACD", fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(df.index, df["atr_pct"])
    ax6.set_title("ATR% (волатильность)", fontsize=12)
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(df.index, df["range_pct"], label="range%")
    ax7.plot(df.index, df["body_pct"], label="body%")
    ax7.legend()
    ax7.set_title("Свечные проценты", fontsize=12)
    ax7.grid(True, alpha=0.3)

    corr_cols = [c for c in feature_cols if c in df.columns]
    ax8 = fig.add_subplot(gs[2, 2])
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, ax=ax8, cbar_kws={"shrink": 0.8})
    ax8.set_title("Корреляция признаков", fontsize=12)

    ax9 = fig.add_subplot(gs[3, :])
    stats_table = df[corr_cols].describe().T
    ax9.axis("tight")
    ax9.axis("off")
    table = ax9.table(
        cellText=stats_table.values,
        rowLabels=stats_table.index,
        colLabels=stats_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax9.set_title("Статистика признаков", fontsize=12, y=0.95)

    plt.suptitle("Визуализация признаков", fontsize=16, y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def package_data(raw_df, cleaned_df, features_df, split_indices, metadata):
    return {
        "raw_data": raw_df,
        "cleaned_data": cleaned_df,
        "features_data": features_df,
        "split_indices": split_indices,
        "metadata": metadata,
    }


def save_package(data_package) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "btc_processed_data.pkl", "wb") as f:
        pickle.dump(data_package, f)
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(data_package["metadata"], f, indent=2, default=str)
    print("Данные сохранены: data/btc_processed_data.pkl и data/metadata.json")


def build_validation_notebook() -> None:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# Phase 1 validation"),
        nbf.v4.new_code_cell(
            "import pickle\nwith open('../data/btc_processed_data.pkl','rb') as f:\n    pkg = pickle.load(f)\nlist(pkg.keys())"
        ),
    ]
    nb_path = NOTEBOOKS_DIR / "01_phase1_validation.ipynb"
    with nb_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def generate_phase1_report(data_package) -> str:
    df = data_package["features_data"]
    meta = data_package["metadata"]
    splits = data_package["split_indices"]
    report = f"""
{'='*60}
ОТЧЕТ ПО ФАЗЕ 1 (ЧАСОВЫЕ ДАННЫЕ)
{'='*60}
1. Данные: {meta['symbol']} {meta['source']} (1m, сессии высокой волатильности)
   Период: {meta['start_date']} до {meta['end_date']}
   Баров: {len(df)}
   Признаков: {len(meta['feature_columns'])}
2. Разделение:
   Train: {splits['train'][0]} - {splits['train'][-1]} ({len(splits['train'])} баров)
   Val:   {splits['val'][0]} - {splits['val'][-1]} ({len(splits['val'])} баров)
   Test:  {splits['test'][0]} - {splits['test'][-1]} ({len(splits['test'])} баров)
3. Качество:
   Пропуски: {df.isnull().sum().sum()}
   Дубликаты индекса: {df.index.duplicated().sum()}
4. Цели: target_15m,30m,60m,90m (+варианты bps)
{'='*60}
"""
    return report


def main() -> None:
    ensure_dirs()
    cfg = DownloadConfig()

    print("=== День 1: Загрузка данных (Bybit 1h) ===")
    raw_df = download_btc_data(cfg)
    raw_fig = FIGURES_DIR / "01_raw_data.png"
    plot_raw_data(raw_df, raw_fig)

    print("\n=== День 2: Очистка данных ===")
    cleaned_df = clean_data(raw_df)
    clean_fig = FIGURES_DIR / "02_clean_vs.png"
    plot_cleaning(raw_df, cleaned_df, clean_fig)

    print("\n=== День 3: Разделение данных ===")
    split_indices = get_split_dates(cleaned_df)
    split_fig = FIGURES_DIR / "03_splits.png"
    split_stats = plot_splits(cleaned_df, split_indices, split_fig)

    print("\n=== День 4-5: Генерация признаков ===")
    steps_per_day = infer_steps_per_day(cleaned_df.index)
    features_df = create_basic_features(cleaned_df, steps_per_day=steps_per_day)
    features_fig = FIGURES_DIR / "04_features.png"
    plot_features(features_df, features_fig)

    target_cols_hours = [f"target_{n}h" for n in HORIZON_HOURS]
    for n in HORIZON_HOURS:
        for thr in TARGET_THRESH_MAP.get(n, []):
            target_cols_hours.append(f"target_{n}h_{int(thr*10000)}bps")

    metadata = {
        "symbol": cfg.symbol,
        "source": cfg.source,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "created_date": datetime.now().isoformat(),
        "feature_columns": list(features_df.columns),
        "target_columns": target_cols_hours + [f"target_{n}d" for n in [1, 7, 14, 21, 28]],
        "split_summary": split_stats.to_dict(orient="records"),
        "steps_per_day": steps_per_day,
        "interval": cfg.interval,
    }

    data_package = package_data(raw_df, cleaned_df, features_df, split_indices, metadata)
    save_package(data_package)
    build_validation_notebook()
    print(generate_phase1_report(data_package))
    print("Фаза 1 завершена, переходим к Фазе 2")


if __name__ == "__main__":
    main()
