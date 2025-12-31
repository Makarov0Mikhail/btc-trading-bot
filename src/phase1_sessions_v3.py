"""Phase 1 v3: безопасные фичи без look-ahead, сессионные таргеты и train/val/test split по времени."""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
FIGURES_DIR = Path("notebooks/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _progress(prefix: str, current: int, total: int, length: int = 20) -> None:
    """Lightweight progress bar for long loops."""
    if total <= 0:
        print(f"{prefix} [............] 0.0%", end="\r", flush=True)
        return
    pct = current / total
    filled = int(length * pct)
    bar = "#" * filled + "." * (length - filled)
    print(f"{prefix} [{bar}] {pct:6.1%}", end="\r", flush=True)
    if current >= total:
        print()

TARGET_CONFIG = {
    # Пороги снижены и добавлен целевой процент сигналов для балансировки классов
    10: {"min_threshold": 0.002, "vol_multiplier": 1.0, "max_hold_bars": 20, "signal_ratio_target": 0.1},
    15: {"min_threshold": 0.003, "vol_multiplier": 1.0, "max_hold_bars": 30, "signal_ratio_target": 0.1},
    30: {"min_threshold": 0.004, "vol_multiplier": 1.0, "max_hold_bars": 45, "signal_ratio_target": 0.1},
}

SESSION_CONFIG = {
    "prebuffer_min": 45,
    "session_len_min": 45,
    "volatility_window": 10,
    "zscore_window_days": 3,
    "volatility_threshold": 1.8,
    "cooldown_min": 30,
}


@dataclass
class DataConfig:
    symbol: str = "BTC/USDT:USDT"
    start_date: datetime = datetime(2020, 1, 1)
    end_date: datetime = datetime.now()
    interval: str = "1m"
    source: str = "bybit"


def load_raw_data() -> pd.DataFrame:
    """Загружаем сырые OHLCV: пытаемся взять из кеша, иначе скачиваем и сохраняем."""
    raw_path = DATA_DIR / "btc_raw_1m.pkl"
    _progress("Load raw", 0, 1)
    if raw_path.exists():
        _progress("Load raw", 1, 1)
        return pickle.load(open(raw_path, "rb"))
    processed_path = DATA_DIR / "btc_processed_data_1m_sessions.pkl"
    if processed_path.exists():
        pkg = pickle.load(open(processed_path, "rb"))
        if "raw_data" in pkg:
            _progress("Load raw", 1, 1)
            return pkg["raw_data"]
        if "cleaned_data" in pkg:
            df = pkg["cleaned_data"]
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            _progress("Load raw", 1, 1)
            return df[cols]
    # Если нигде нет — скачиваем и сохраняем для будущих запусков
    try:
        from phase1_preprocessing import download_btc_data
    except ImportError as e:
        raise FileNotFoundError(
            "Нет сырых минутных данных и нельзя скачать: импорт download_btc_data не найден. "
            "Добавьте data/btc_raw_1m.pkl или реализуйте загрузку."
        ) from e
    print("Сырые данные не найдены, скачиваем заново...")
    cfg = DataConfig()
    df = download_btc_data(cfg)
    if df is None or len(df) == 0:
        raise FileNotFoundError("Не удалось скачать данные, проверьте download_btc_data.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Сырые данные сохранены в {raw_path} ({len(df)} строк).")
    _progress("Load raw", 1, 1)
    return df


class SafeFeatureEngineer:
    """Безопасный расчёт фич без look-ahead bias."""

    def __init__(self, steps_per_day: int = 1440):
        self.steps_per_day = steps_per_day
        self.safe_lag = 0  # no extra lag; use current bar data

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        total_steps = 16
        step = 0

        def bump():
            nonlocal step
            step += 1
            _progress("  Features", step, total_steps)

        _progress("  Features", 0, total_steps)

        # Base features with 1-bar lookback for returns and volumes.
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return"] = df["Close"].pct_change(1)
        df["log_volume"] = np.log(df["Volume"].replace(0, np.nan)).ffill().bfill()
        bump()

        # Лаги OHLCV
        df["close_lag1"] = df["Close"].shift(1)
        df["open_lag1"] = df["Open"].shift(1)
        df["high_lag1"] = df["High"].shift(1)
        df["low_lag1"] = df["Low"].shift(1)
        df["volume_lag1"] = df["Volume"].shift(1)

        # Скользящие на лагнутых данных
        windows = [7, 14, 21]
        for w in windows:
            bars = max(1, int(w * self.steps_per_day / 1440))
            df[f"sma_{w}"] = df["Close"].rolling(bars).mean().shift(1)
            df[f"logret_std_{w}"] = df["log_return"].rolling(bars).std().shift(1)
            df[f"logret_mean_{w}"] = df["log_return"].rolling(bars).mean().shift(1)
            df[f"volume_ma_{w}"] = df["Volume"].rolling(bars).mean().shift(1)
            bump()

        # RSI на лагнутых ценах
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().shift(1)
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().shift(1)
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        bump()

        # ATR на лагнутых ценах
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean().shift(1)
        df["atr_pct"] = df["atr_14"] / df["Close"].shift(1)
        bump()

        # MACD на лагнутых ценах
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (exp1 - exp2).shift(1)
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().shift(1)
        df["macd_hist"] = (df["macd"] - df["macd_signal"]).shift(1)
        bump()

        # Лагнутые признаки свечей (без look-ahead)
        df["prev_range_pct"] = (df["High"].shift(1) - df["Low"].shift(1)) / df["Close"].shift(1)
        df["prev_body_pct"] = (df["Close"].shift(1) - df["Open"].shift(1)) / df["Open"].shift(1)
        bump()

        # Время
        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        bump()

        # Производные лаги
        df["prev_candle_body"] = df["prev_body_pct"].shift(1)
        df["prev_candle_range"] = df["prev_range_pct"].shift(1)
        df["volume_change"] = df["Volume"] / df["Volume"].shift(1)
        bump()

        for lag in [1, 2, 3]:
            df[f"log_return_lag_{lag}"] = df["log_return"].shift(lag)
            df[f"volume_change_lag_{lag}"] = df["volume_change"].shift(lag)
            bump()

        df["in_session"] = 0
        df["session_id"] = -1
        df["session_start"] = pd.NaT
        df["session_end"] = pd.NaT
        bump()
        return df


class SessionDetector:
    def __init__(self, config: Dict = None):
        self.config = config or SESSION_CONFIG

    def detect_sessions_clear(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        cfg = self.config
        # 1. Лог-доходности с лагом
        lr = np.log(df["Close"] / df["Close"].shift(1))

        # 2. Short-term volatility using data up to t-1.
        vol_window = cfg.get("volatility_window", 10)
        short_vol = lr.rolling(vol_window).std().shift(1)

        # 3. Пороги на тех же данных
        lookback = cfg.get("lookback_days", 90) * 24 * 60
        percentile = cfg.get("percentile", 97) / 100
        short_vol_for_threshold = lr.rolling(vol_window).std()
        percentile_val = short_vol_for_threshold.rolling(lookback).quantile(percentile).shift(1)
        median_vol = short_vol_for_threshold.rolling(lookback).median().shift(1)

        # 4. Минимальный порог
        min_thr = median_vol * cfg.get("min_vol_multiplier", 1.5)
        threshold = np.maximum(percentile_val, min_thr)

        # 5-6. Сравнение и старт сессии
        high_vol = short_vol > threshold
        starts = high_vol & (~high_vol.shift(1, fill_value=False))
        return starts, short_vol, threshold

    def detect_sessions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        starts, short_vol, threshold = self.detect_sessions_clear(df)
        cfg = self.config
        sessions: List[dict] = []
        last_end = None
        for start_time in df.index[starts]:
            if last_end and (start_time - last_end).total_seconds() < cfg["cooldown_min"] * 60:
                continue
            pre = start_time - timedelta(minutes=cfg["prebuffer_min"])
            end = start_time + timedelta(minutes=cfg["session_len_min"])
            sessions.append(
                {
                    "session_id": len(sessions),
                    "prebuffer_start": pre,
                    "session_start": start_time,
                    "session_end": end,
                    "volatility": float(short_vol.loc[start_time]),
                    "threshold": float(threshold.loc[start_time]) if pd.notna(threshold.loc[start_time]) else np.nan,
                }
            )
            last_end = end

        res = df.copy()
        res["in_session"] = 0
        res["session_id"] = -1
        res["session_start"] = pd.NaT
        res["session_end"] = pd.NaT
        for s in sessions:
            mask = (res.index >= s["session_start"]) & (res.index <= s["session_end"])
            res.loc[mask, "in_session"] = 1
            res.loc[mask, "session_id"] = s["session_id"]
            res.loc[mask, "session_start"] = s["session_start"]
            res.loc[mask, "session_end"] = s["session_end"]
        return res, sessions


class AdaptiveSessionDetector(SessionDetector):
    def __init__(self, config: Dict = None):
        base = {**SESSION_CONFIG}
        base.pop("volatility_threshold", None)
        base.update(
            {
                "lookback_days": 90,
                "percentile": 95,  # Снижено с 97 для большего количества сессий
                "min_vol_multiplier": 1.5,
            }
        )
        base.update(config or {})
        super().__init__(base)


class EnhancedSessionDetector(AdaptiveSessionDetector):
    def detect_sessions_with_stats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict], Dict]:
        res, sessions = self.detect_sessions(df)
        if not sessions:
            return res, sessions, {}
        durations = [(s["session_end"] - s["session_start"]).total_seconds() / 60 for s in sessions]
        stats = {
            "total_sessions": len(sessions),
            "avg_duration_min": float(np.mean(durations)),
            "min_duration_min": float(np.min(durations)),
            "max_duration_min": float(np.max(durations)),
            "total_session_bars": int(sum(len(res[(res.index >= s["session_start"]) & (res.index <= s["session_end"])]) for s in sessions)),
            "session_hours": {},
            "session_weekdays": {},
        }
        for s in sessions:
            h = s["session_start"].hour
            stats["session_hours"][h] = stats["session_hours"].get(h, 0) + 1
            wd = s["session_start"].weekday()
            stats["session_weekdays"][wd] = stats["session_weekdays"].get(wd, 0) + 1
        return res, sessions, stats


class TargetBuilder:
    def __init__(self, target_config: Dict = None):
        self.config = target_config or TARGET_CONFIG

    def _is_in_same_session(self, df: pd.DataFrame, idx1, idx2) -> bool:
        if "session_id" not in df.columns or "session_end" not in df.columns:
            return False
        if idx1 not in df.index or idx2 not in df.index:
            return False
        sid1 = df.loc[idx1, "session_id"]
        sid2 = df.loc[idx2, "session_id"]
        if pd.isna(sid1) or pd.isna(sid2) or sid1 < 0 or sid1 != sid2:
            return False
        session_end = df.loc[idx1, "session_end"]
        if pd.isna(session_end):
            return False
        return idx2 < session_end

    def create_momentum_targets(self, df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
        targets = pd.Series(np.nan, index=df.index, name=f"target_{horizon_minutes}m_momentum")
        mask = df["in_session"] == 1
        if mask.sum() == 0:
            return targets

        lookback = max(1, min(5, horizon_minutes // 2))
        idx = df.index
        delta = pd.to_timedelta(horizon_minutes, unit="m")
        future_idx = idx + delta

        close = df["Close"]
        future_close = close.reindex(future_idx).to_numpy()
        session_id = df["session_id"]
        future_sid = session_id.reindex(future_idx).to_numpy()
        session_end = df["session_end"]
        mask_arr = mask.to_numpy()

        future_idx_np = future_idx.to_numpy()

        valid = (
            mask_arr
            & ~np.isnan(future_close)
            & ~pd.isna(future_sid)
            & (future_sid == session_id.to_numpy())
            & ~pd.isna(session_end)
            & (future_idx_np <= session_end.to_numpy())
        )

        returns_series = close.pct_change(1)
        past_mean_arr = returns_series.rolling(lookback).mean().shift(1).to_numpy()
        future_ret = future_close / close.to_numpy() - 1

        valid = valid & ~np.isnan(past_mean_arr) & ~np.isnan(future_ret)

        current_mom = (past_mean_arr > 0).astype(int)
        res = np.where(
            ((current_mom == 1) & (future_ret > 0)) | ((current_mom == 0) & (future_ret < 0)),
            1,
            0,
        )
        targets.loc[df.index[valid]] = res[valid]
        return targets

    def create_dynamic_threshold_targets(self, df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
        if horizon_minutes not in self.config:
            raise ValueError(f"Horizon {horizon_minutes} не сконфигурирован")
        cfg = self.config[horizon_minutes]
        targets = pd.Series(np.nan, index=df.index, name=f"target_{horizon_minutes}m_dynamic")
        mask = df["in_session"] == 1
        if mask.sum() == 0:
            return targets

        idx = df.index
        delta = pd.to_timedelta(horizon_minutes, unit="m")
        future_idx = idx + delta

        close = df["Close"]
        future_close = close.reindex(future_idx).to_numpy()
        session_id = df["session_id"]
        future_sid = session_id.reindex(future_idx).to_numpy()
        session_end = df["session_end"]
        mask_arr = mask.to_numpy()

        future_idx_np = future_idx.to_numpy()

        valid = (
            mask_arr
            & ~np.isnan(future_close)
            & ~pd.isna(future_sid)
            & (future_sid == session_id.to_numpy())
            & ~pd.isna(session_end)
            & (future_idx_np <= session_end.to_numpy())
        )

        hist_vol = df["log_return"].rolling(100).std().shift(1)
        dyn_thr = (hist_vol * cfg["vol_multiplier"]).clip(lower=cfg["min_threshold"])
        future_ret = future_close / close.to_numpy() - 1

        hist_vol_arr = hist_vol.to_numpy()
        dyn_thr_arr = dyn_thr.to_numpy()

        valid = valid & ~np.isnan(hist_vol_arr) & ~np.isnan(future_ret)

        # Трёхклассовый таргет: 1 (Long), -1 (Short), 0 (Neutral)
        result = np.zeros(len(valid), dtype=float)
        result[valid & (future_ret > dyn_thr_arr)] = 1   # Long: сильное движение вверх
        result[valid & (future_ret < -dyn_thr_arr)] = -1  # Short: сильное движение вниз
        # 0 остаётся для neutral (в пределах порога)
        result[~valid] = np.nan  # NaN для невалидных
        targets.loc[:] = result
        return targets

    def create_better_targets(self, df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
        """Трёхклассовый таргет: 1 / 0 / -1 с учётом волатильности и границ сессии.
        
        NaN для баров вне сессий или без валидных данных.
        0 (neutral) только для valid баров внутри сессии с малым движением.
        """
        # Инициализируем NaN - только valid бары получат значения
        targets = pd.Series(np.nan, index=df.index, name=f"target_{horizon_minutes}m")
        mask = df["in_session"] == 1
        if mask.sum() == 0:
            return targets

        future_idx = df.index + pd.Timedelta(minutes=horizon_minutes)
        future_close = df["Close"].reindex(future_idx).to_numpy()
        current_close = df["Close"].to_numpy()
        future_ret = future_close / current_close - 1

        cfg = self.config.get(horizon_minutes, {})
        hist_vol = df["log_return"].rolling(20).std().shift(1).to_numpy()
        min_thr = cfg.get("min_threshold", 0.0015)
        vol_mult = cfg.get("vol_multiplier", 1.0)

        base_dyn = hist_vol * vol_mult
        dyn_thr = np.maximum(base_dyn, min_thr)

        session_id = df["session_id"].to_numpy()
        future_sid = df["session_id"].reindex(future_idx).to_numpy()
        session_end = df["session_end"]
        future_idx_np = future_idx.to_numpy()
        session_end_np = session_end.to_numpy()

        valid = (
            mask.to_numpy()
            & ~np.isnan(future_ret)
            & ~pd.isna(future_sid)
            & (future_sid == session_id)
            & ~pd.isna(session_end)
            & (future_idx_np <= session_end_np)
            & ~np.isnan(hist_vol)
        )

        # Сначала ставим 0 (neutral) для всех valid, потом перезаписываем 1/-1
        targets_arr = targets.to_numpy().astype(float)
        targets_arr[valid] = 0  # neutral для valid баров
        targets_arr[valid & (future_ret > dyn_thr)] = 1
        targets_arr[valid & (future_ret < -dyn_thr)] = -1
        return pd.Series(targets_arr, index=df.index, name=f"target_{horizon_minutes}m")


def analyze_targets_statistics(session_df: pd.DataFrame, horizons: List[int]) -> Dict:
    stats: Dict[str, Dict] = {}
    for h in horizons:
        target_cols = [f"target_{h}m", f"target_{h}m_momentum", f"target_{h}m_dynamic"]
        for tgt in target_cols:
            if tgt not in session_df.columns:
                continue
            series = session_df[tgt].dropna()
            if len(series) == 0:
                continue
            entry: Dict[str, Any] = {
                "total_samples": int(len(series)),
                "in_session_samples": int(len(series)),
                "class_distribution": dict(series.value_counts(normalize=True).round(3)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "sharpe_ratio": float(series.mean() / (series.std() + 1e-8)),
            }
            if tgt.endswith("m") and not tgt.endswith("momentum") and not tgt.endswith("dynamic"):
                entry["three_class"] = True
                entry["long_ratio"] = float((series == 1).mean())
                entry["short_ratio"] = float((series == -1).mean())
                entry["neutral_ratio"] = float((series == 0).mean())
            stats[tgt] = entry
    return stats


class DataSplitter:
    @staticmethod
    def time_split(df: pd.DataFrame, train_end, val_end) -> Dict[str, pd.DataFrame]:
        train_end_dt = pd.to_datetime(train_end) if not isinstance(train_end, pd.Timestamp) else train_end
        val_end_dt = pd.to_datetime(val_end) if not isinstance(val_end, pd.Timestamp) else val_end
        train = df[df.index < train_end_dt].copy()
        val = df[(df.index >= train_end_dt) & (df.index < val_end_dt)].copy()
        test = df[df.index >= val_end_dt].copy()
        return {"train": train, "val": val, "test": test, "split_dates": {"train_end": train_end, "val_end": val_end}}


def main():
    print("=== Phase 1 v3: фичи, сессии, таргеты, split ===")
    t0 = time.time()
    raw = load_raw_data().sort_index()
    raw_start, raw_end = raw.index[0], raw.index[-1]
    print(f"[1/5] Данные загружены: {len(raw):,} строк, период {raw_start} - {raw_end}")
    # Определяем даты сплита пропорцией 21m/9m/остальное от конца
    end_dt = raw_end
    start_dt = max(raw_start, end_dt - pd.DateOffset(months=36))
    train_end_dt = start_dt + pd.DateOffset(months=21)  # Уменьшено с 24 для большего VAL
    val_end_dt = train_end_dt + pd.DateOffset(months=9)   # Увеличено с 6 для большего VAL

    splitter = DataSplitter()
    raw_splits = splitter.time_split(raw.loc[start_dt:end_dt], train_end=train_end_dt, val_end=val_end_dt)

    processed = {}
    horizons = [10, 15, 30]
    tb = TargetBuilder()
    fe = SafeFeatureEngineer(steps_per_day=1440)
    detector = EnhancedSessionDetector()

    for split_name, split_df in raw_splits.items():
        if not isinstance(split_df, pd.DataFrame):
            continue
        if split_df.empty:
            processed[split_name] = pd.DataFrame()
            continue
        t1 = time.time()
        feats = fe.calculate_all_features(split_df)
        session_df, sessions, session_stats = detector.detect_sessions_with_stats(feats)
        print(f"[2/5] {split_name}: фичи рассчитаны ({len(feats.columns)}), сессии={len(sessions)} за {time.time()-t1:.1f} c")
        if sessions and session_stats:
            print(f"    Статистика сессий {split_name}: avg {session_stats['avg_duration_min']:.1f} мин, баров {session_stats['total_session_bars']:,}")
        validate_features_extended(session_df, split_name)
        for h in horizons:
            session_df[f"target_{h}m_momentum"] = tb.create_momentum_targets(session_df, h)
            session_df[f"target_{h}m_dynamic"] = tb.create_dynamic_threshold_targets(session_df, h)
            session_df[f"target_{h}m"] = tb.create_better_targets(session_df, h)
        processed[split_name] = session_df

    print("\n[3/5] Проверка распределения таргетов:")
    for split_name in ["train", "val", "test"]:
        split_df = processed.get(split_name, pd.DataFrame())
        if split_df.empty:
            continue
        print(f"  Split {split_name}:")
        validate_targets(split_df, horizons)

    target_stats = analyze_targets_statistics(processed.get("train", pd.DataFrame()), horizons)
    print("\n[4/5] Анализ таргетов (train):")
    for h in horizons:
        better_key = f"target_{h}m"
        if better_key in target_stats:
            stats = target_stats[better_key]
            print(f"    Horizon {h} мин (better target):")
            print(f"      Образцы: {stats['total_samples']:,}")
            print(f"      Распределение: {stats.get('class_distribution', {})}")
            if stats.get("three_class"):
                print(f"      Лонги: {stats.get('long_ratio', 0):.1%}, Шорты: {stats.get('short_ratio', 0):.1%}, Нейтрал: {stats.get('neutral_ratio', 0):.1%}")

    full_parts = [processed.get("train", pd.DataFrame()), processed.get("val", pd.DataFrame()), processed.get("test", pd.DataFrame())]
    non_empty = [p for p in full_parts if not p.empty]
    full_data = pd.concat(non_empty).sort_index() if non_empty else pd.DataFrame()
    out = {
        "full_data": full_data,
        "train_data": processed.get("train", pd.DataFrame()),
        "val_data": processed.get("val", pd.DataFrame()),
        "test_data": processed.get("test", pd.DataFrame()),
        "sessions": [],  # сессии по сплитам не суммируем
        "config": {"target_config": TARGET_CONFIG, "session_config": SESSION_CONFIG, "split_dates": {"train_end": train_end_dt, "val_end": val_end_dt}},
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "btc_processed_v3.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[5/5] Сохранено: data/btc_processed_v3.pkl "
          f"(train {len(out['train_data'])}, val {len(out['val_data'])}, test {len(out['test_data'])})")
    print(f"Всего Phase 1 v3 заняла {time.time()-t0:.1f} секунд ({(time.time()-t0)/60:.1f} минут)")


def validate_features_extended(df: pd.DataFrame, split_name: str, threshold: float = 0.1):
    """Расширенная валидация лагов"""
    if df.empty or "Close" not in df.columns:
        return

    future_ret = {
        1: df["Close"].shift(-1) / df["Close"] - 1,
        2: df["Close"].shift(-2) / df["Close"] - 1,
        3: df["Close"].shift(-3) / df["Close"] - 1,
    }

    safe_features = {
        "lag",
        "prev",
        "shift",
        "rolling",
        "sma_",
        "ema_",
        "rsi_",
        "atr_",
        "macd",
        "std_",
        "mean_",
        "volume_ma_",
        "logret_std_",
    }

    warned = 0
    for col in df.columns:
        if any(safe in col.lower() for safe in safe_features):
            continue
        if col in [
            "Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "session_start",
            "session_end",
            "session_id",
            "in_session",
            "hour",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "hour_cos",
        ]:
            continue

        corr_vals = []
        for lag, fr in future_ret.items():
            corr = df[col].corr(fr)
            corr_vals.append(corr)

        if any(abs(c) > threshold for c in corr_vals):
            if warned == 0:
                print(f"Валидация лагов ({split_name}): обнаружены корреляции с будущим > {threshold}")
            warned += 1
            if warned <= 5:
                print(f"  {col}: corr@+1={corr_vals[0]:.3f}, @+2={corr_vals[1]:.3f}, @+3={corr_vals[2]:.3f}")

    if warned > 5:
        print(f"  ... и еще {warned-5} фич")


def validate_features(df: pd.DataFrame, split_name: str, threshold: float = 0.1) -> None:
    """Совместимость: вызываем расширенную валидацию."""
    return validate_features_extended(df, split_name, threshold)


def validate_targets(df: pd.DataFrame, horizons: List[int]):
    """Проверяет качество таргетов"""
    for h in horizons:
        col = f"target_{h}m"
        if col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            print(f"  Target {h}min:")
            print(f"    Samples: {len(series):,}")
            print(f"    Distribution: {dict(series.value_counts(normalize=True).round(3))}")

            if series.nunique() == 3:  # трёхклассовый
                long_pct = (series == 1).mean()
                short_pct = (series == -1).mean()
                neutral_pct = (series == 0).mean()
                print(f"    Long: {long_pct:.1%}, Short: {short_pct:.1%}, Neutral: {neutral_pct:.1%}")

                if (long_pct + short_pct) < 0.1:
                    print(f"    WARNING: Only {(long_pct+short_pct):.1%} trading signals!")


if __name__ == "__main__":
    main()
