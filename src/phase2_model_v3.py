"""Phase 2 v3: Train/Val/Test пайплайн с Optuna CV и торговой симуляцией на unseen тесте."""
from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# Подавляем все warnings включая sklearn parallel
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("notebooks/figures")
for p in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


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




def validate_temporal_alignment(df: pd.DataFrame, horizons: List[int]) -> None:
    """Basic temporal alignment checks for targets and features."""
    if df.empty or "Close" not in df.columns:
        return

    print("\n[Temporal alignment] Checking targets and feature leakage...")
    for h in horizons:
        target_col = f"target_{h}m"
        if target_col not in df.columns:
            continue
        targets = df[target_col].dropna()
        if len(targets) == 0:
            continue
        long_signals = int((targets == 1).sum())
        short_signals = int((targets == -1).sum())
        neutral_signals = int((targets == 0).sum())
        total_signals = len(targets)
        print(
            f"  Horizon {h}m: total={total_signals}, long={long_signals}, short={short_signals}, neutral={neutral_signals}"
        )
        if "in_session" in df.columns:
            in_session_mask = df["in_session"] == 1
            aligned_mask = in_session_mask.reindex(targets.index).fillna(False)
            in_session_count = int(targets[aligned_mask].count())
            print(f"    In-session signals: {in_session_count}")

    features_to_check = ["log_return", "log_return_lag_1", "sma_7", "rsi_14"]
    future_1 = df["Close"].shift(-1) / df["Close"] - 1
    future_2 = df["Close"].shift(-2) / df["Close"] - 1
    for feature in features_to_check:
        if feature not in df.columns:
            continue
        corr_1 = df[feature].corr(future_1)
        corr_2 = df[feature].corr(future_2)
        if (corr_1 is not None and abs(corr_1) > 0.1) or (corr_2 is not None and abs(corr_2) > 0.1):
            print(f"  Warning: {feature} corr@+1={corr_1:.3f}, corr@+2={corr_2:.3f}")


def validate_temporal_alignment_detailed(df: pd.DataFrame, horizons: List[int]) -> Dict[str, Any]:
    """Detailed temporal alignment checks for targets and feature leakage."""
    results: Dict[str, Any] = {}
    if df.empty or "Close" not in df.columns:
        return results

    for h in horizons:
        target_col = f"target_{h}m"
        if target_col not in df.columns:
            continue
        targets = df[target_col].dropna()
        if len(targets) == 0:
            continue

        signal_counts = {
            "total": len(targets),
            "long": int((targets == 1).sum()),
            "short": int((targets == -1).sum()),
            "neutral": int((targets == 0).sum()),
        }

        if "in_session" in df.columns:
            in_session_mask = df["in_session"] == 1
            aligned = in_session_mask.reindex(targets.index).fillna(False)
            signal_counts["in_session"] = int(targets[aligned].count())
            signal_counts["out_of_session"] = int(targets[~aligned].count())

        if "session_end" in df.columns:
            problematic = 0
            for idx in targets.index:
                if idx in df.index:
                    session_end = df.loc[idx, "session_end"]
                    if pd.notna(session_end):
                        future_time = idx + pd.Timedelta(minutes=h)
                        if future_time > session_end:
                            problematic += 1
            signal_counts["outside_session_bounds"] = problematic

        feature_leakage: Dict[str, float] = {}
        for feature in ["log_return", "sma_7", "rsi_14"]:
            if feature not in df.columns:
                continue
            for lag in [1, 2, 3]:
                future_target = targets.shift(-lag)
                valid_mask = ~df[feature].isna() & ~future_target.isna()
                if valid_mask.any():
                    corr = df.loc[valid_mask, feature].corr(future_target[valid_mask])
                    if corr is not None and abs(corr) > 0.1:
                        feature_leakage[f"{feature}_lag{lag}"] = float(corr)

        results[f"horizon_{h}"] = {
            "signal_counts": signal_counts,
            "feature_leakage": feature_leakage,
            "issues": len(feature_leakage) > 0 or signal_counts.get("outside_session_bounds", 0) > 0,
        }

    return results


def validate_temporal_alignment_comprehensive(
    df: pd.DataFrame,
    horizons: List[int],
    min_horizon_samples: int = 10,
) -> Dict[str, Any]:
    """Comprehensive temporal alignment checks."""
    results: Dict[str, Any] = {}
    if df.empty or "Close" not in df.columns:
        return results

    for h in horizons:
        target_col = f"target_{h}m"
        if target_col not in df.columns:
            continue

        targets = df[target_col].dropna()
        if len(targets) == 0:
            continue

        data_availability = {
            "total_targets": len(targets),
            "available_for_horizon": 0,
            "missing_future_data": 0,
        }

        for idx in targets.index:
            future_time = idx + pd.Timedelta(minutes=h)
            if future_time in df.index:
                data_availability["available_for_horizon"] += 1
            else:
                data_availability["missing_future_data"] += 1

        session_violations = 0
        if "session_end" in df.columns and "in_session" in df.columns:
            for idx in targets.index:
                if idx in df.index and df.loc[idx, "in_session"] == 1:
                    session_end = df.loc[idx, "session_end"]
                    if pd.notna(session_end):
                        future_time = idx + pd.Timedelta(minutes=h)
                        if future_time > session_end:
                            session_violations += 1

        feature_leakage: Dict[str, Dict[str, Any]] = {}
        features_to_check = [
            "log_return",
            "log_return_lag_1",
            "log_return_lag_2",
            "sma_7",
            "sma_14",
            "rsi_14",
            "atr_pct",
            "macd_hist",
        ]

        for feature in features_to_check:
            if feature not in df.columns:
                continue
            for lag in [1, 2, 3, h]:
                future_target = targets.shift(-lag)
                valid_mask = ~df[feature].isna() & ~future_target.isna()
                if valid_mask.sum() > min_horizon_samples:
                    corr = df.loc[valid_mask, feature].corr(future_target[valid_mask])
                    if corr is not None and abs(corr) > 0.1:
                        feature_leakage[f"{feature}_lag{lag}"] = {
                            "correlation": float(corr),
                            "samples": int(valid_mask.sum()),
                        }

        signal_distribution = {
            "long": int((targets == 1).sum()),
            "short": int((targets == -1).sum()),
            "neutral": int((targets == 0).sum()),
            "total": len(targets),
        }

        if "in_session" in df.columns:
            in_session_mask = df["in_session"] == 1
            aligned = in_session_mask.reindex(targets.index).fillna(False)
            signal_distribution["in_session"] = int(targets[aligned].count())
            signal_distribution["out_of_session"] = int(targets[~aligned].count())

        issues = []
        if data_availability["missing_future_data"] > 0:
            issues.append(f"Missing future data: {data_availability['missing_future_data']}")
        if session_violations > 0:
            issues.append(f"Session violations: {session_violations}")
        if len(feature_leakage) > 0:
            issues.append(f"Feature leakage: {len(feature_leakage)} features")
        if signal_distribution.get("in_session", 0) < signal_distribution["total"] * 0.9:
            issues.append("Significant out-of-session signals")

        results[f"horizon_{h}"] = {
            "data_availability": data_availability,
            "session_violations": session_violations,
            "feature_leakage": feature_leakage,
            "signal_distribution": signal_distribution,
            "issues": issues,
            "has_issues": len(issues) > 0,
        }

    return results


def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Безопасное predict_proba: если один класс, возвращаем 0 или 1 по classes_."""
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.ones(len(X)) if model.classes_[0] == 1 else np.zeros(len(X))
    return proba[:, 1]


SIM_CFG = {
    "commission": 0.001,
    "spread_pct": 0.0002,
    "risk_per_trade": 0.20,  # 20% депозита
    "max_position_pct": 0.90,  # 90% депозита
    "default_stop_pct": 0.005,
    "default_take_pct": 0.015,
    "max_hold_minutes": 30,
    "min_hold_bars": 3,
}


def calculate_sharpe(portfolio: np.ndarray) -> float:
    r = np.diff(portfolio) / portfolio[:-1]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(252 * 24 * 60)


def max_drawdown(portfolio: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio)
    dd = (portfolio - peak) / peak
    return abs(dd.min() * 100)


class RealisticSimulator:
    def __init__(self, cfg: Dict = None):
        self.cfg = dict(cfg) if cfg else dict(SIM_CFG)  # копия, чтобы не мутировать оригинал

    def calc_stop(self, data: pd.DataFrame, idx: int) -> float:
        base = self.cfg["default_stop_pct"]
        if "atr_pct" in data.columns:
            base = max(base, data["atr_pct"].iloc[idx] * 1.5)
        if "session_id" in data.columns:
            sid = data["session_id"].iloc[idx]
            if sid != -1:
                mask = (data["session_id"] == sid) & (data.index <= data.index[idx])
                vol = data.loc[mask, "log_return"].std()
                if vol > 0:
                    base = max(base, vol * 2)
        return min(base, 0.03)

    def position_size(self, capital: float, price: float, stop_pct: float) -> float:
        risk_amt = capital * self.cfg["risk_per_trade"]
        pos_risk = risk_amt / (price * abs(stop_pct) + 1e-8)
        # Учитываем комиссию при расчёте максимальной позиции
        commission_rate = self.cfg.get("commission", 0.001)
        max_capital = capital * self.cfg["max_position_pct"] / (1 + commission_rate)
        pos_cap = max_capital / price
        return min(pos_risk, pos_cap)

    def simulate(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        probas: pd.Series,
        initial_capital: float = 10000,
        horizon: int | None = None,
        stop_mult: float = 1.0,
        take_mult: float = 2.0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Execute on SAME BAR (мгновенное исполнение по Close текущего бара)."""
        capital = initial_capital
        position = 0.0  # >0 long size, <0 short size (abs = qty)
        entry_price = 0.0
        entry_time = None
        portfolio = [capital]
        trades: List[Dict] = []

        sig = signals.reindex(data.index).fillna(0)
        prob = probas.reindex(data.index).fillna(0)

        # More realistic parameters
        self.cfg["commission"] = 0.0007
        self.cfg["risk_per_trade"] = 0.20  # 20% депозита
        self.cfg["max_position_pct"] = 0.80  # 80% депозита
        if horizon is not None:
            self.cfg["max_hold_minutes"] = horizon

        MAX_TRADES_PER_SESSION = 2
        session_trade_counts: Dict[int, int] = {}
        min_hold = self.cfg["min_hold_bars"]

        for i in range(len(data)):
            current_time = data.index[i]
            signal = sig.iloc[i]
            confidence = prob.iloc[i]
            in_session = True
            if "in_session" in data.columns:
                in_session = data["in_session"].iloc[i] == 1

            # ========== МГНОВЕННОЕ ИСПОЛНЕНИЕ НА ТЕКУЩЕМ БАРЕ ==========
            if signal != 0 and position == 0 and in_session:
                sid = data["session_id"].iloc[i] if "session_id" in data else -1
                if sid != -1 and session_trade_counts.get(sid, 0) >= MAX_TRADES_PER_SESSION:
                    pass  # Пропускаем - превышен лимит сделок в сессии
                else:
                    # Исполнение по Close текущего бара
                    exec_price = data["Close"].iloc[i]

                    if signal == 1:  # LONG
                        exec_price = exec_price * (1 + self.cfg["spread_pct"] / 2)
                        stop_pct = self.calc_stop(data, i) * stop_mult
                        size = self.position_size(capital, exec_price, stop_pct)
                        trade_value = size * exec_price
                        comm_in = trade_value * self.cfg["commission"]

                        if trade_value + comm_in <= capital and size > 0:
                            capital -= (trade_value + comm_in)
                            position = size
                            entry_price = exec_price
                            entry_time = current_time
                            trades.append(
                                {
                                    "entry_time": entry_time,
                                    "entry_price": entry_price,
                                    "size": size,
                                    "confidence": confidence,
                                    "commission": comm_in,
                                    "stop_loss_pct": stop_pct,
                                    "take_profit_pct": stop_pct * take_mult,
                                    "session_id": sid,
                                    "entry_bar": i,
                                    "direction": "LONG",
                                }
                            )
                            if sid != -1:
                                session_trade_counts[sid] = session_trade_counts.get(sid, 0) + 1

                    elif signal == -1:  # SHORT
                        exec_price = exec_price * (1 - self.cfg["spread_pct"] / 2)
                        stop_pct = self.calc_stop(data, i) * stop_mult
                        size = self.position_size(capital, exec_price, stop_pct)
                        trade_value = size * exec_price
                        comm_in = trade_value * self.cfg["commission"]

                        if trade_value + comm_in <= capital and size > 0:
                            capital += trade_value - comm_in
                            position = -size
                            entry_price = exec_price
                            entry_time = current_time
                            trades.append(
                                {
                                    "entry_time": entry_time,
                                    "entry_price": entry_price,
                                    "size": size,
                                    "confidence": confidence,
                                    "commission": comm_in,
                                    "stop_loss_pct": stop_pct,
                                    "take_profit_pct": stop_pct * take_mult,
                                    "session_id": sid,
                                    "entry_bar": i,
                                    "direction": "SHORT",
                                }
                            )
                            if sid != -1:
                                session_trade_counts[sid] = session_trade_counts.get(sid, 0) + 1
            # ========== КОНЕЦ МГНОВЕННОГО ИСПОЛНЕНИЯ ==========

            # ========== ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ ПРИ ВЫХОДЕ ИЗ СЕССИИ ==========
            if not in_session:
                if position != 0 and trades:
                    current_trade = trades[-1]
                    price = data["Close"].iloc[i]
                    entry_bar_pos = current_trade.get("entry_bar", i)
                    holding_bars_pos = i - entry_bar_pos
                    
                    if position > 0:
                        pnl_pct_close = price / entry_price - 1
                        exit_price_close = price * (1 - self.cfg["spread_pct"] / 2)
                        exit_val = position * exit_price_close
                        comm_out = exit_val * self.cfg["commission"]
                        capital += exit_val - comm_out
                        pnl_usd = exit_val - comm_out - (position * entry_price) - current_trade["commission"]
                    else:
                        pnl_pct_close = entry_price / price - 1
                        exit_price_close = price * (1 + self.cfg["spread_pct"] / 2)
                        buy_cost = abs(position) * exit_price_close
                        comm_out = buy_cost * self.cfg["commission"]
                        capital -= buy_cost + comm_out
                        pnl_usd = (entry_price - exit_price_close) * abs(position) - current_trade["commission"] - comm_out
                    
                    current_trade.update({
                        "exit_time": current_time,
                        "exit_price": exit_price_close,
                        "exit_commission": comm_out,
                        "exit_reason": "SESSION_END",
                        "pnl_pct": pnl_pct_close,
                        "pnl_usd": pnl_usd,
                        "holding_bars": holding_bars_pos,
                        "holding_minutes": (current_time - entry_time).total_seconds() / 60 if entry_time else 0,
                    })
                    position = 0.0
                
                portfolio.append(capital if position == 0 else portfolio[-1])
                continue
            # ========== КОНЕЦ ЗАКРЫТИЯ ПРИ ВЫХОДЕ ИЗ СЕССИИ ==========

            if position != 0 and trades:
                current_trade = trades[-1]
                entry_bar = current_trade.get("entry_bar", i)
                if i > entry_bar:
                    price = data["Close"].iloc[i]
                    if position > 0:
                        pnl_pct = price / entry_price - 1
                    else:
                        pnl_pct = entry_price / price - 1
                    stop_pct = current_trade.get("stop_loss_pct", self.cfg["default_stop_pct"])
                    take_pct = current_trade.get("take_profit_pct", self.cfg["default_take_pct"])
                    holding_bars = i - entry_bar

                    exit_reason = None
                    if pnl_pct <= -stop_pct:
                        exit_reason = "STOP_LOSS"
                    elif pnl_pct >= take_pct:
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars < min_hold:
                        exit_reason = None
                    # MODEL_EXIT убран - neutral сигнал не означает "выходи"
                    elif entry_time and (current_time - entry_time).total_seconds() > self.cfg["max_hold_minutes"] * 60:
                        exit_reason = "MAX_HOLD"

                    if exit_reason:
                        if position > 0:
                            exit_price = price * (1 - self.cfg["spread_pct"] / 2)
                            exit_val = position * exit_price
                            comm_out = exit_val * self.cfg["commission"]
                            capital += exit_val - comm_out
                            pnl_usd = exit_val - comm_out - (position * entry_price) - current_trade["commission"]
                        else:
                            exit_price = price * (1 + self.cfg["spread_pct"] / 2)
                            buy_cost = abs(position) * exit_price
                            comm_out = buy_cost * self.cfg["commission"]
                            capital -= buy_cost + comm_out
                            pnl_usd = (entry_price - exit_price) * abs(position) - current_trade["commission"] - comm_out
                        current_trade.update(
                            {
                                "exit_time": current_time,
                                "exit_price": exit_price,
                                "exit_commission": comm_out,
                                "exit_reason": exit_reason,
                                "pnl_pct": pnl_pct,
                                "pnl_usd": pnl_usd,
                                "holding_bars": holding_bars,
                                "holding_minutes": (current_time - entry_time).total_seconds() / 60 if entry_time else 0,
                            }
                        )
                        position = 0.0

            # Блок отложенных сигналов удалён - теперь мгновенное исполнение

            if position > 0:
                current_price = data["Close"].iloc[i]
                unrealized = position * (current_price - entry_price)
                portfolio.append(capital + unrealized)
            elif position < 0:
                current_price = data["Close"].iloc[i]
                unrealized = abs(position) * (entry_price - current_price)
                portfolio.append(capital + unrealized)
            else:
                portfolio.append(capital)

        if position > 0:
            last_price = data["Close"].iloc[-1] * (1 - self.cfg["spread_pct"] / 2)
            exit_val = position * last_price
            comm_out = exit_val * self.cfg["commission"]
            capital += exit_val - comm_out
            if trades:
                trades[-1].update(
                    {
                        "exit_time": data.index[-1],
                        "exit_price": last_price,
                        "exit_commission": comm_out,
                        "exit_reason": "FORCE_CLOSE",
                        "pnl_pct": last_price / entry_price - 1,
                        "pnl_usd": exit_val - (position * entry_price) - trades[-1]["commission"] - comm_out,
                    }
                )
        elif position < 0:
            last_price = data["Close"].iloc[-1] * (1 + self.cfg["spread_pct"] / 2)
            buy_cost = abs(position) * last_price
            comm_out = buy_cost * self.cfg["commission"]
            capital -= buy_cost + comm_out
            if trades:
                trades[-1].update(
                    {
                        "exit_time": data.index[-1],
                        "exit_price": last_price,
                        "exit_commission": comm_out,
                        "exit_reason": "FORCE_CLOSE",
                        "pnl_pct": entry_price / last_price - 1,
                        "pnl_usd": (entry_price - last_price) * abs(position) - trades[-1]["commission"] - comm_out,
                    }
                )

        return np.array(portfolio), trades



class HorizonAwareSimulator(RealisticSimulator):
    def simulate_with_horizon(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        probas: pd.Series,
        horizon: int,
        initial_capital: float = 10000,
        stop_mult: float = 1.0,
        take_mult: float = 2.0,
        trailing_mult: float = 0.0,  # 0 = отключен, >0 = активация trailing при достижении trailing_mult * stop_pct
        breakeven_mult: float = 0.0,  # 0 = отключен, >0 = перевод стопа в безубыток при достижении breakeven_mult * stop_pct
        proba_long_series: pd.Series = None,  # Предвычисленные вероятности LONG
        proba_short_series: pd.Series = None,  # Предвычисленные вероятности SHORT
        exit_confidence_drop: float = 0.0,  # Порог падения confidence для выхода (0 = отключено)
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Simulate with horizon-aware exits, entry on SAME BAR (мгновенное исполнение)."""
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        portfolio = [capital]
        trades: List[Dict] = []

        sig = signals.reindex(data.index).fillna(0)
        prob = probas.reindex(data.index).fillna(0)
        
        # Предвычисленные вероятности для MODEL_EXIT (быстрый lookup вместо predict_proba на каждом баре)
        proba_long_lookup = proba_long_series.reindex(data.index).fillna(0.5) if proba_long_series is not None else None
        proba_short_lookup = proba_short_series.reindex(data.index).fillna(0.5) if proba_short_series is not None else None

        # УБИРАЕМ pending_state - теперь мгновенное исполнение
        self.cfg["commission"] = 0.0007
        self.cfg["risk_per_trade"] = 0.20  # 20% депозита
        self.cfg["max_position_pct"] = 0.80  # 80% депозита
        self.cfg["max_hold_minutes"] = max(self.cfg.get("max_hold_minutes", 0), horizon)

        MAX_TRADES_PER_SESSION = 2
        session_trade_counts: Dict[int, int] = {}
        min_hold = self.cfg["min_hold_bars"]

        for i in range(len(data)):
            current_time = data.index[i]
            signal = sig.iloc[i]
            confidence = prob.iloc[i]
            in_session = True
            if "in_session" in data.columns:
                in_session = data["in_session"].iloc[i] == 1

            # ========== МГНОВЕННОЕ ИСПОЛНЕНИЕ НА ТЕКУЩЕМ БАРЕ ==========
            if signal != 0 and position == 0 and in_session:
                sid = data["session_id"].iloc[i] if "session_id" in data else -1

                # Проверка лимита сделок в сессии
                if sid != -1 and session_trade_counts.get(sid, 0) >= MAX_TRADES_PER_SESSION:
                    pass  # Пропускаем - превышен лимит сделок в сессии
                else:
                    # Исполнение по Close текущего бара
                    exec_price = data["Close"].iloc[i]

                    if signal == 1:  # LONG
                        exec_price = exec_price * (1 + self.cfg["spread_pct"] / 2)
                        stop_pct = self.calc_stop(data, i) * stop_mult
                        size = self.position_size(capital, exec_price, stop_pct)
                        trade_value = size * exec_price
                        comm_in = trade_value * self.cfg["commission"]

                        if trade_value + comm_in <= capital and size > 0:
                            capital -= (trade_value + comm_in)
                            position = size
                            entry_price = exec_price
                            entry_time = current_time
                            entry_bar = i
                            # Сохраняем разницу вероятностей для MODEL_EXIT
                            entry_proba_diff = 0.0
                            if proba_long_lookup is not None and proba_short_lookup is not None:
                                entry_proba_diff = proba_long_lookup.iloc[i] - proba_short_lookup.iloc[i]
                            trades.append(
                                {
                                    "entry_time": entry_time,
                                    "entry_price": entry_price,
                                    "size": size,
                                    "confidence": confidence,
                                    "entry_proba_diff": entry_proba_diff,
                                    "commission": comm_in,
                                    "stop_loss_pct": stop_pct,
                                    "take_profit_pct": stop_pct * take_mult,
                                    "session_id": sid,
                                    "entry_bar": entry_bar,
                                    "direction": "LONG",
                                    "target_exit_bar": entry_bar + horizon,
                                }
                            )
                            if sid != -1:
                                session_trade_counts[sid] = session_trade_counts.get(sid, 0) + 1

                    elif signal == -1:  # SHORT
                        exec_price = exec_price * (1 - self.cfg["spread_pct"] / 2)
                        stop_pct = self.calc_stop(data, i) * stop_mult
                        size = self.position_size(capital, exec_price, stop_pct)
                        trade_value = size * exec_price
                        comm_in = trade_value * self.cfg["commission"]

                        if trade_value + comm_in <= capital and size > 0:
                            capital += trade_value - comm_in
                            position = -size
                            entry_price = exec_price
                            entry_time = current_time
                            entry_bar = i
                            # Сохраняем разницу вероятностей для MODEL_EXIT (для short: short - long)
                            entry_proba_diff = 0.0
                            if proba_long_lookup is not None and proba_short_lookup is not None:
                                entry_proba_diff = proba_short_lookup.iloc[i] - proba_long_lookup.iloc[i]
                            trades.append(
                                {
                                    "entry_time": entry_time,
                                    "entry_price": entry_price,
                                    "size": size,
                                    "confidence": confidence,
                                    "entry_proba_diff": entry_proba_diff,
                                    "commission": comm_in,
                                    "stop_loss_pct": stop_pct,
                                    "take_profit_pct": stop_pct * take_mult,
                                    "session_id": sid,
                                    "entry_bar": entry_bar,
                                    "direction": "SHORT",
                                    "target_exit_bar": entry_bar + horizon,
                                }
                            )
                            if sid != -1:
                                session_trade_counts[sid] = session_trade_counts.get(sid, 0) + 1
            # ========== КОНЕЦ МГНОВЕННОГО ИСПОЛНЕНИЯ ==========

            # ========== ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ ПРИ ВЫХОДЕ ИЗ СЕССИИ ==========
            if not in_session:
                if position != 0 and trades:
                    current_trade = trades[-1]
                    price = data["Close"].iloc[i]
                    entry_bar_pos = current_trade.get("entry_bar", i)
                    holding_bars_pos = i - entry_bar_pos
                    
                    if position > 0:
                        pnl_pct_close = price / entry_price - 1
                        exit_price_close = price * (1 - self.cfg["spread_pct"] / 2)
                        exit_val = position * exit_price_close
                        comm_out = exit_val * self.cfg["commission"]
                        capital += exit_val - comm_out
                        pnl_usd = exit_val - comm_out - (position * entry_price) - current_trade["commission"]
                    else:
                        pnl_pct_close = entry_price / price - 1
                        exit_price_close = price * (1 + self.cfg["spread_pct"] / 2)
                        buy_cost = abs(position) * exit_price_close
                        comm_out = buy_cost * self.cfg["commission"]
                        capital -= buy_cost + comm_out
                        pnl_usd = (entry_price - exit_price_close) * abs(position) - current_trade["commission"] - comm_out
                    
                    current_trade.update({
                        "exit_time": current_time,
                        "exit_price": exit_price_close,
                        "exit_commission": comm_out,
                        "exit_reason": "SESSION_END",
                        "pnl_pct": pnl_pct_close,
                        "pnl_usd": pnl_usd,
                        "holding_bars": holding_bars_pos,
                        "holding_minutes": (current_time - entry_time).total_seconds() / 60 if entry_time else 0,
                    })
                    position = 0.0
                
                portfolio.append(capital if position == 0 else portfolio[-1])
                continue
            # ========== КОНЕЦ ЗАКРЫТИЯ ПРИ ВЫХОДЕ ИЗ СЕССИИ ==========

            if position != 0 and trades:
                current_trade = trades[-1]
                entry_bar = current_trade.get("entry_bar", i)
                if i > entry_bar:
                    price = data["Close"].iloc[i]
                    if position > 0:
                        pnl_pct = price / entry_price - 1
                    else:
                        pnl_pct = entry_price / price - 1
                    
                    # Получаем текущий стоп (может быть уже модифицирован trailing/breakeven)
                    stop_pct = current_trade.get("current_stop_pct", current_trade.get("stop_loss_pct", self.cfg["default_stop_pct"]))
                    original_stop_pct = current_trade.get("stop_loss_pct", self.cfg["default_stop_pct"])
                    take_pct = current_trade.get("take_profit_pct", self.cfg["default_take_pct"])
                    holding_bars = i - entry_bar
                    target_exit_bar = current_trade.get("target_exit_bar", entry_bar + horizon)
                    
                    # ========== TRAILING STOP & BREAK-EVEN ЛОГИКА ==========
                    max_pnl = current_trade.get("max_pnl_pct", 0.0)
                    if pnl_pct > max_pnl:
                        max_pnl = pnl_pct
                        current_trade["max_pnl_pct"] = max_pnl
                    
                    # Break-even: если достигли breakeven_mult * stop_pct прибыли, переносим стоп в 0
                    if breakeven_mult > 0 and not current_trade.get("breakeven_activated", False):
                        if max_pnl >= breakeven_mult * original_stop_pct:
                            stop_pct = 0.0001  # практически безубыток (с маленьким буфером)
                            current_trade["current_stop_pct"] = stop_pct
                            current_trade["breakeven_activated"] = True
                    
                    # Trailing stop: если достигли trailing_mult * stop_pct, начинаем подтягивать стоп
                    if trailing_mult > 0 and max_pnl >= trailing_mult * original_stop_pct:
                        # Trailing stop = max_pnl - original_stop_pct (подтягиваем на расстоянии оригинального стопа)
                        trailing_stop = max_pnl - original_stop_pct
                        if trailing_stop > stop_pct:
                            stop_pct = trailing_stop
                            current_trade["current_stop_pct"] = stop_pct
                            current_trade["trailing_activated"] = True
                    # ========== КОНЕЦ TRAILING/BREAK-EVEN ==========

                    exit_reason = None
                    if i >= target_exit_bar:
                        exit_reason = "HORIZON_EXIT"
                    elif pnl_pct <= -stop_pct:
                        exit_reason = "STOP_LOSS"
                    elif pnl_pct >= take_pct:
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars < min_hold:
                        exit_reason = None
                    # ========== MODEL_EXIT: модель решает выходить ==========
                    # ТОЛЬКО если прибыль >= 30% от тейка - фиксируем значимую прибыль при развороте
                    elif exit_confidence_drop > 0 and pnl_pct >= take_pct * 0.3 and proba_long_lookup is not None and proba_short_lookup is not None:
                        # Текущая разница вероятностей (как при входе)
                        current_proba_long = proba_long_lookup.iloc[i]
                        current_proba_short = proba_short_lookup.iloc[i]
                        
                        if position > 0:  # LONG - смотрим разницу в пользу long
                            # При входе было: proba_long - proba_short >= min_confidence (положительная)
                            # Если разница стала отрицательной или сильно упала - выходим с прибылью
                            current_diff = current_proba_long - current_proba_short
                            entry_diff = current_trade.get("entry_proba_diff", current_trade.get("confidence", 0.5))
                            # Выходим если: разница стала отрицательной (модель теперь за SHORT)
                            if current_diff < 0 or (entry_diff - current_diff) >= exit_confidence_drop:
                                exit_reason = "MODEL_EXIT"
                                current_trade["exit_proba_diff"] = current_diff
                                current_trade["confidence_drop"] = entry_diff - current_diff
                        else:  # SHORT - смотрим разницу в пользу short
                            # При входе было: proba_short - proba_long >= min_confidence (положительная)
                            current_diff = current_proba_short - current_proba_long
                            entry_diff = current_trade.get("entry_proba_diff", abs(current_trade.get("confidence", 0.5)))
                            if current_diff < 0 or (entry_diff - current_diff) >= exit_confidence_drop:
                                exit_reason = "MODEL_EXIT"
                                current_trade["exit_proba_diff"] = current_diff
                                current_trade["confidence_drop"] = entry_diff - current_diff
                    # ========== КОНЕЦ MODEL_EXIT ==========
                    elif entry_time and (current_time - entry_time).total_seconds() > self.cfg["max_hold_minutes"] * 60:
                        exit_reason = "MAX_HOLD"

                    if exit_reason:
                        if position > 0:
                            exit_price = price * (1 - self.cfg["spread_pct"] / 2)
                            exit_val = position * exit_price
                            comm_out = exit_val * self.cfg["commission"]
                            capital += exit_val - comm_out
                            pnl_usd = exit_val - comm_out - (position * entry_price) - current_trade["commission"]
                        else:
                            exit_price = price * (1 + self.cfg["spread_pct"] / 2)
                            buy_cost = abs(position) * exit_price
                            comm_out = buy_cost * self.cfg["commission"]
                            capital -= buy_cost + comm_out
                            pnl_usd = (entry_price - exit_price) * abs(position) - current_trade["commission"] - comm_out
                        current_trade.update(
                            {
                                "exit_time": current_time,
                                "exit_price": exit_price,
                                "exit_commission": comm_out,
                                "exit_reason": exit_reason,
                                "pnl_pct": pnl_pct,
                                "pnl_usd": pnl_usd,
                                "holding_bars": holding_bars,
                                "holding_minutes": (current_time - entry_time).total_seconds() / 60 if entry_time else 0,
                            }
                        )
                        position = 0.0

            # Блок отложенных сигналов удалён - теперь мгновенное исполнение

            # Расчет стоимости портфеля
            if position > 0:
                current_price = data["Close"].iloc[i]
                unrealized = position * (current_price - entry_price)
                portfolio.append(capital + unrealized)
            elif position < 0:
                current_price = data["Close"].iloc[i]
                unrealized = abs(position) * (entry_price - current_price)
                portfolio.append(capital + unrealized)
            else:
                portfolio.append(capital)

        # Принудительное закрытие в конце
        if position > 0:
            last_price = data["Close"].iloc[-1] * (1 - self.cfg["spread_pct"] / 2)
            exit_val = position * last_price
            comm_out = exit_val * self.cfg["commission"]
            capital += exit_val - comm_out
            if trades:
                trades[-1].update(
                    {
                        "exit_time": data.index[-1],
                        "exit_price": last_price,
                        "exit_commission": comm_out,
                        "exit_reason": "FORCE_CLOSE",
                        "pnl_pct": last_price / entry_price - 1,
                        "pnl_usd": exit_val - (position * entry_price) - trades[-1]["commission"] - comm_out,
                    }
                )
        elif position < 0:
            last_price = data["Close"].iloc[-1] * (1 + self.cfg["spread_pct"] / 2)
            buy_cost = abs(position) * last_price
            comm_out = buy_cost * self.cfg["commission"]
            capital -= buy_cost + comm_out
            if trades:
                trades[-1].update(
                    {
                        "exit_time": data.index[-1],
                        "exit_price": last_price,
                        "exit_commission": comm_out,
                        "exit_reason": "FORCE_CLOSE",
                        "pnl_pct": entry_price / last_price - 1,
                        "pnl_usd": (entry_price - last_price) * abs(position) - trades[-1]["commission"] - comm_out,
                    }
                )

        return np.array(portfolio), trades


class HorizonAwareFeaturePreparer:
    @staticmethod
    def prepare_features_for_horizon(
        data: pd.DataFrame,
        horizon: int,
        target_col: str,
        min_samples: int = 100,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features with windows adapted to horizon."""
        if horizon <= 10:
            volatility_windows = [5, 10, 15]
            ma_windows = [3, 5, 7]
        elif horizon <= 20:
            volatility_windows = [10, 15, 20]
            ma_windows = [5, 10, 15]
        else:
            volatility_windows = [15, 20, 30]
            ma_windows = [10, 20, 30]

        features = [
            "log_return",
            "log_return_lag_2",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "prev_candle_body",
            # Фичи объёма:
            "volume_change",
            "volume_change_lag_1",
        ]

        for w in ma_windows:
            features.append(f"sma_{w}")
        for w in volatility_windows:
            features.append(f"logret_std_{w}")

        features.extend(
            [
                "atr_pct",
                "rsi_14",
                "macd_hist",
            ]
        )

        data = data.sort_index()
        existing_features = [f for f in features if f in data.columns]
        X = data[existing_features].copy()
        y = data[target_col].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if "lag" in col or "prev" in col:
                continue
            X[col] = X[col].ffill(limit=5)

        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) > min_samples * 2:
            X = X.iloc[min_samples:]
            y = y.iloc[min_samples:]

        return X, y


class SafeFeaturePreparer:
    @staticmethod
    def prepare_features_safe(
        data: pd.DataFrame,
        horizon: int,
        target_col: str,
        min_samples: int = 100,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return HorizonAwareFeaturePreparer.prepare_features_for_horizon(data, horizon, target_col, min_samples)


class ModelManager:
    @staticmethod
    def select_best_target(data: pd.DataFrame, horizon: int) -> str:
        """Выбираем трёхклассовый таргет с наибольшим количеством non-zero сигналов."""
        candidates = [f"target_{horizon}m", f"target_{horizon}m_dynamic"]  # Убрали momentum - он бинарный
        best = None
        best_score = -1
        for tgt in candidates:
            if tgt not in data:
                continue
            series = data[tgt].dropna()
            if len(series) < 100:
                continue
            # Для трёхклассового таргета считаем количество Long + Short (не Neutral)
            n_classes = series.nunique()
            if n_classes < 3:
                continue  # Нужен трёхклассовый таргет
            non_zero = (series != 0).sum()  # Long + Short
            if non_zero > best_score:
                best_score = non_zero
                best = tgt
        if best is None:
            raise ValueError(f"No valid three-class target for horizon {horizon}")
        return best

    @staticmethod
    def prepare_features(data: pd.DataFrame, horizon: int, target_col: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = target_col or ModelManager.select_best_target(data, horizon)
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data for horizon {horizon}")
        features = [
            "log_return",
            "return",
            "log_volume",
            "close_lag1",
            "open_lag1",
            "high_lag1",
            "low_lag1",
            "sma_7",
            "sma_14",
            "logret_std_7",
            "logret_std_14",
            "logret_mean_7",
            "logret_mean_14",
            "rsi_14",
            "atr_pct",
            "macd_hist",
            "prev_range_pct",
            "prev_body_pct",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "prev_candle_body",
            "prev_candle_range",
            "volume_change",
            "log_return_lag_1",
            "log_return_lag_2",
            "volume_change_lag_1",
        ]
        feats = [c for c in features if c in data.columns]
        X = data[feats].replace([np.inf, -np.inf], np.nan)
        y = data[target_col].copy()
        mask = ~X.isna().any(axis=1) & ~y.isna()
        return X[mask], y[mask]

    @staticmethod
    def prepare_features_optimized(data: pd.DataFrame, horizon: int, target_col: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Сокращённый список безопасных фич без look-ahead."""
        target_col = target_col or ModelManager.select_best_target(data, horizon)
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data for horizon {horizon}")
        return HorizonAwareFeaturePreparer.prepare_features_for_horizon(data, horizon, target_col)


class EnhancedModelManager(ModelManager):
    @staticmethod
    def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, model_type: str = "long") -> Dict:
        tmp_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        tmp_model.fit(X, y)
        fi = pd.DataFrame({"feature": X.columns, "importance": tmp_model.feature_importances_}).sort_values("importance", ascending=False)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        fi["mutual_info"] = mi_scores
        corrs = []
        for col in X.columns:
            if len(X[col].unique()) > 1:
                corrs.append(abs(np.corrcoef(X[col], y)[0, 1]))
            else:
                corrs.append(0)
        fi["abs_correlation"] = corrs
        return {
            "top_10_features": fi.head(10).to_dict("records"),
            "feature_importance_stats": {
                "max_importance": float(fi["importance"].max()),
                "min_importance": float(fi["importance"].min()),
                "mean_importance": float(fi["importance"].mean()),
            },
        }

    @staticmethod
    def analyze_model_performance_by_time(X: pd.DataFrame, y: pd.Series, data: pd.DataFrame, model) -> Dict:
        results: Dict[str, Any] = {}
        hours = data["hour"].unique()
        hourly_perf = {}
        for hour in sorted(hours):
            mask = data["hour"] == hour
            X_hour = X[mask]
            y_hour = y[mask]
            if len(X_hour) > 10:
                y_pred = model.predict(X_hour)
                hourly_perf[int(hour)] = {"samples": int(len(X_hour)), "accuracy": float(accuracy_score(y_hour, y_pred))}
        results["hourly_performance"] = hourly_perf
        if "day_of_week" in data.columns:
            weekday_perf = {}
            for wd in sorted(data["day_of_week"].unique()):
                mask = data["day_of_week"] == wd
                X_wd = X[mask]
                y_wd = y[mask]
                if len(X_wd) > 10:
                    y_pred = model.predict(X_wd)
                    weekday_perf[int(wd)] = {"samples": int(len(X_wd)), "accuracy": float(accuracy_score(y_wd, y_pred))}
            results["weekday_performance"] = weekday_perf
        if "in_session" in data.columns:
            session_mask = data["in_session"] == 1
            non_mask = ~session_mask
            if session_mask.any():
                y_pred = model.predict(X[session_mask])
                results["in_session"] = {"samples": int(session_mask.sum()), "accuracy": float(accuracy_score(y[session_mask], y_pred))}
            if non_mask.any():
                y_pred = model.predict(X[non_mask])
                results["non_session"] = {"samples": int(non_mask.sum()), "accuracy": float(accuracy_score(y[non_mask], y_pred))}
        return results


class FeatureImpactAnalyzer:
    """Комплексный анализ влияния фич: не только сила, но и направление (помогает/вредит)."""

    @staticmethod
    def analyze_pre_training(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Анализ ДО обучения: MI, корреляция, univariate AUC."""
        from sklearn.metrics import roc_auc_score
        from scipy.stats import spearmanr

        results = []
        y_binary = (y != 0).astype(int) if y.nunique() > 2 else y

        for col in X.columns:
            feat = X[col].values
            valid_mask = ~np.isnan(feat)
            if valid_mask.sum() < 100:
                continue

            feat_valid = feat[valid_mask]
            y_valid = y_binary.values[valid_mask]

            # Mutual Information
            mi = mutual_info_classif(feat_valid.reshape(-1, 1), y_valid, random_state=42)[0]

            # Spearman correlation (монотонная связь)
            spearman_corr, spearman_p = spearmanr(feat_valid, y_valid)

            # Univariate AUC (если бы только эта фича предсказывала)
            try:
                auc = roc_auc_score(y_valid, feat_valid)
                # AUC < 0.5 означает обратную связь, нормализуем
                auc_adjusted = max(auc, 1 - auc)
            except:
                auc_adjusted = 0.5

            results.append({
                "feature": col,
                "mutual_info": mi,
                "spearman_corr": spearman_corr,
                "spearman_p": spearman_p,
                "univariate_auc": auc_adjusted,
                "predictive_power": mi * auc_adjusted,  # комбинированный скор
            })

        df = pd.DataFrame(results).sort_values("predictive_power", ascending=False)
        return df

    @staticmethod
    def analyze_permutation_importance(
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "f1",
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Permutation Importance с направлением: показывает помогает фича или вредит.
        
        Положительное значение = фича помогает (без неё хуже)
        Отрицательное значение = фича ВРЕДИТ (без неё лучше!)
        """
        from sklearn.metrics import f1_score, accuracy_score

        if metric == "f1":
            scorer = lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            scorer = accuracy_score

        # Базовый скор
        y_pred_base = model.predict(X_val)
        base_score = scorer(y_val, y_pred_base)

        results = []
        for col in X_val.columns:
            scores_permuted = []
            for _ in range(n_repeats):
                X_permuted = X_val.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                y_pred_perm = model.predict(X_permuted)
                scores_permuted.append(scorer(y_val, y_pred_perm))

            mean_permuted = np.mean(scores_permuted)
            std_permuted = np.std(scores_permuted)

            # importance = base_score - permuted_score
            # Если положительный: фича помогает (без неё скор упал)
            # Если отрицательный: фича ВРЕДИТ (без неё скор вырос!)
            importance = base_score - mean_permuted

            results.append({
                "feature": col,
                "importance": importance,
                "importance_std": std_permuted,
                "base_score": base_score,
                "permuted_score": mean_permuted,
                "helps": importance > 0,  # True = помогает, False = вредит
                "verdict": "HELPS ✅" if importance > 0.001 else ("HURTS ❌" if importance < -0.001 else "NEUTRAL"),
            })

        df = pd.DataFrame(results).sort_values("importance", ascending=False)
        return df

    @staticmethod
    def analyze_feature_in_trades(
        trades: List[Dict],
        data: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Анализ значений фич в прибыльных vs убыточных сделках.
        
        Если фича работает правильно:
        - В прибыльных сделках должен быть характерный паттерн
        - В убыточных - другой или случайный
        """
        closed_trades = [t for t in trades if t.get("exit_time") and t.get("pnl_usd") is not None]
        if len(closed_trades) < 20:
            return pd.DataFrame({"error": ["Недостаточно сделок для анализа"]})

        winning = [t for t in closed_trades if t.get("pnl_usd", 0) > 0]
        losing = [t for t in closed_trades if t.get("pnl_usd", 0) < 0]

        results = []
        for col in feature_cols:
            if col not in data.columns:
                continue

            win_values = []
            lose_values = []

            for t in winning:
                entry_time = t.get("entry_time")
                if entry_time in data.index:
                    val = data.loc[entry_time, col]
                    if pd.notna(val):
                        win_values.append(val)

            for t in losing:
                entry_time = t.get("entry_time")
                if entry_time in data.index:
                    val = data.loc[entry_time, col]
                    if pd.notna(val):
                        lose_values.append(val)

            if len(win_values) < 5 or len(lose_values) < 5:
                continue

            win_mean = np.mean(win_values)
            lose_mean = np.mean(lose_values)
            win_std = np.std(win_values)
            lose_std = np.std(lose_values)

            # Effect size (Cohen's d) - насколько различаются распределения
            pooled_std = np.sqrt((win_std**2 + lose_std**2) / 2)
            effect_size = (win_mean - lose_mean) / (pooled_std + 1e-8)

            # Статистический тест
            from scipy.stats import mannwhitneyu
            try:
                stat, p_value = mannwhitneyu(win_values, lose_values, alternative='two-sided')
            except:
                p_value = 1.0

            results.append({
                "feature": col,
                "win_mean": win_mean,
                "lose_mean": lose_mean,
                "win_std": win_std,
                "lose_std": lose_std,
                "effect_size": effect_size,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "pattern": "WIN > LOSE" if win_mean > lose_mean else "LOSE > WIN",
            })

        df = pd.DataFrame(results).sort_values("effect_size", key=abs, ascending=False)
        return df

    @staticmethod
    def full_analysis(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model,
        trades: List[Dict],
        data: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Полный анализ влияния фич."""
        print("\n" + "=" * 60)
        print("КОМПЛЕКСНЫЙ АНАЛИЗ ВЛИЯНИЯ ФИЧ")
        print("=" * 60)

        # 1. Pre-training analysis
        print("\n[1/3] Анализ до обучения (MI, корреляция, AUC)...")
        pre_training = FeatureImpactAnalyzer.analyze_pre_training(X_train, y_train)
        print("Top-5 по предсказательной силе:")
        for _, row in pre_training.head(5).iterrows():
            print(f"  {row['feature']}: MI={row['mutual_info']:.4f}, AUC={row['univariate_auc']:.3f}, "
                  f"Spearman={row['spearman_corr']:.3f}")

        # 2. Permutation importance
        print("\n[2/3] Permutation Importance (помогает/вредит)...")
        perm_importance = FeatureImpactAnalyzer.analyze_permutation_importance(model, X_val, y_val)
        
        helping = perm_importance[perm_importance["importance"] > 0.001]
        hurting = perm_importance[perm_importance["importance"] < -0.001]
        
        print(f"  Помогают модели: {len(helping)} фич")
        for _, row in helping.head(5).iterrows():
            print(f"    ✅ {row['feature']}: +{row['importance']:.4f}")
        
        if len(hurting) > 0:
            print(f"  ВРЕДЯТ модели: {len(hurting)} фич")
            for _, row in hurting.iterrows():
                print(f"    ❌ {row['feature']}: {row['importance']:.4f} (РЕКОМЕНДУЕТСЯ УДАЛИТЬ)")

        # 3. Trade-based analysis
        print("\n[3/3] Анализ фич в сделках (прибыльные vs убыточные)...")
        trade_analysis = FeatureImpactAnalyzer.analyze_feature_in_trades(
            trades, data, list(X_train.columns)
        )
        if "error" not in trade_analysis.columns:
            significant = trade_analysis[trade_analysis["significant"]]
            print(f"  Статистически значимые различия: {len(significant)} фич")
            for _, row in significant.head(5).iterrows():
                print(f"    {row['feature']}: effect={row['effect_size']:.3f}, "
                      f"p={row['p_value']:.4f}, {row['pattern']}")

        print("\n" + "=" * 60)

        return {
            "pre_training": pre_training,
            "permutation_importance": perm_importance,
            "trade_analysis": trade_analysis,
        }


class SafeWalkForwardSplitter:
    """Safe walk-forward split with index alignment checks."""

    @staticmethod
    def split_with_gap_safe(
        X: pd.DataFrame,
        y: pd.Series,
        data: pd.DataFrame,
        horizon: int,
        n_windows: int = 3,
        test_size_pct: float = 0.2,
        gap_minutes: int = 60,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index")

        if not X.index.equals(data.index):
            common_idx = X.index.intersection(data.index)
            if len(common_idx) < len(X) * 0.9:
                raise ValueError("Less than 90% index overlap between X and data")
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            data = data.loc[common_idx]

        required_cols = ["session_id"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        n_samples = len(X)
        if n_samples == 0:
            return []

        indices = np.arange(n_samples)
        test_size_bars = max(int(n_samples * test_size_pct), horizon * 2)
        gap_bars = max(gap_minutes, horizon)
        splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = []

        def check_session_overlap(start_idx, end_idx, test_start_idx, test_end_idx) -> bool:
            train_sessions = set(data.iloc[start_idx:end_idx]["session_id"].unique())
            test_sessions = set(data.iloc[test_start_idx:test_end_idx]["session_id"].unique())
            train_sessions.discard(-1)
            test_sessions.discard(-1)
            return len(train_sessions.intersection(test_sessions)) == 0

        for window in range(n_windows):
            train_end = n_samples - test_size_bars * (window + 1) - gap_bars * (window + 1)
            if train_end <= horizon * 10:
                break

            test_start = train_end + gap_bars
            test_end = test_start + test_size_bars
            if test_end > n_samples:
                break

            if not check_session_overlap(0, train_end, test_start, test_end):
                found = False
                for offset in range(-gap_bars, gap_bars + 1, horizon):
                    adj_train_end = train_end + offset
                    adj_test_start = adj_train_end + gap_bars
                    adj_test_end = adj_test_start + test_size_bars
                    if adj_train_end <= horizon * 10 or adj_test_end > n_samples:
                        continue
                    if check_session_overlap(0, adj_train_end, adj_test_start, adj_test_end):
                        train_end, test_start, test_end = adj_train_end, adj_test_start, adj_test_end
                        found = True
                        break
                if not found:
                    continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        return splits


def optimize_with_walkforward(X_train: pd.DataFrame, y_train: pd.Series, data_train: pd.DataFrame, horizon: int, n_trials: int = 20, full_data: pd.DataFrame = None) -> Dict:
    """Оптимизация с walk-forward + ENSEMBLE top-5 trials для robustness.
    
    Args:
        full_data: ПОЛНЫЕ данные (включая out-of-session бары) для правильного SESSION_END
    """
    sim = HorizonAwareSimulator()
    splitter = SafeWalkForwardSplitter()
    total_trials = n_trials

    X_train = X_train.sort_index()
    y_train = y_train.reindex(X_train.index)
    data_train = data_train.loc[X_train.index]
    
    # Если full_data не передан, используем data_train (для обратной совместимости)
    if full_data is None:
        full_data = data_train

    def objective(trial):
        scores = []
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 180),  # сужен диапазон
            "max_depth": trial.suggest_int("max_depth", 6, 12),  # сужен: избегаем переобучения
            "min_samples_split": trial.suggest_int("min_samples_split", 15, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 8, 25),
            "random_state": 42,
        }
        # ПОВЫШЕННЫЕ ПОРОГИ для лучшей точности (accuracy 70%+ при thr >= 0.60)
        thr_long = trial.suggest_float("thr_long", 0.60, 0.75)  # 0.60+ для accuracy 70%+
        thr_short = trial.suggest_float("thr_short", 0.55, 0.70)  # СНИЖЕНО: больше шортов
        # Минимальная разница вероятностей для входа (фильтрует слабые сигналы)
        min_confidence = trial.suggest_float("min_confidence", 0.15, 0.30)  # ПОВЫШЕНО: отсекаем Q1-Q2
        
        # === RISK/REWARD CONSTRAINT ===
        # При WR ~60% нужен RR >= 1.5 для прибыли (с учётом комиссий и проскальзывания)
        # Сначала выбираем stop_mult, затем take_mult ГАРАНТИРОВАННО >= stop_mult * min_rr
        min_rr_ratio = 1.5  # СНИЖЕНО: больше TAKE_PROFIT выходов
        stop_mult = trial.suggest_float("stop_mult", 0.6, 1.0)  # Оригинальный диапазон
        # take_mult ВСЕГДА >= stop_mult * min_rr_ratio (гарантированный RR >= 1.5)
        take_mult_min = stop_mult * min_rr_ratio
        take_mult = trial.suggest_float("take_mult", max(take_mult_min, 1.2), max(take_mult_min + 0.5, 2.0))  # СНИЖЕНО: ближе цели
        
        # Trailing/breakeven - консервативные значения
        trailing_mult = trial.suggest_float("trailing_mult", 0.4, 0.9)  # чуть шире
        breakeven_mult = trial.suggest_float("breakeven_mult", 0.3, 0.7)  # раннее включение
        
        # MODEL_EXIT: порог падения confidence для выхода (0.1 = 10% падение)
        exit_confidence_drop = trial.suggest_float("exit_confidence_drop", 0.05, 0.25)
        
        _progress(f"      Optuna H{horizon}", trial.number + 1, total_trials)

        # УВЕЛИЧЕНО: 5 CV-окон вместо 3 для лучшей robustness
        splits = splitter.split_with_gap_safe(X_train, y_train, data_train, horizon, n_windows=5, test_size_pct=0.15, gap_minutes=horizon)
        for X_tr, X_val, y_tr, y_val in splits:
            if len(X_val) < 100 or len(X_tr) < 1000:
                continue
            # две модели: лонг и шорт
            y_tr_long = (y_tr == 1).astype(int)
            y_tr_short = (y_tr == -1).astype(int)
            y_val_long = (y_val == 1).astype(int)
            y_val_short = (y_val == -1).astype(int)

            model_long = RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1)
            model_short = RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1)
            if y_tr_long.sum() == 0 or y_val_long.sum() == 0 or y_tr_short.sum() == 0 or y_val_short.sum() == 0:
                scores.append(-50)
                continue
            model_long.fit(X_tr, y_tr_long)
            model_short.fit(X_tr, y_tr_short)
            proba_long = safe_predict_proba(model_long, X_val)
            proba_short = safe_predict_proba(model_short, X_val)
            proba_diff = proba_long - proba_short  # Разница вероятностей
            preds_val = pd.Series(0, index=X_val.index)
            
            # Сначала определяем val_data для фильтра тренда
            val_data = data_train.loc[X_val.index]
            
            # TREND FILTER: используем SMA для фильтрации против тренда
            # Если цена ниже SMA - не лонгуем, если выше - не шортим
            # FIX: используем 'Close' с заглавной буквы (как в данных)
            close_prices = val_data['Close'] if 'Close' in val_data.columns else (val_data['close'] if 'close' in val_data.columns else None)
            sma_col = 'sma_20' if 'sma_20' in val_data.columns else ('sma_14' if 'sma_14' in val_data.columns else None)
            
            if close_prices is not None and sma_col is not None:
                sma_values = val_data[sma_col]
                trend_up = close_prices > sma_values  # Цена выше SMA = аптренд
                trend_down = close_prices < sma_values  # Цена ниже SMA = даунтренд
                
                # Long: только в аптренде или нейтрально
                long_mask = (proba_long >= thr_long) & (proba_short < thr_short) & (proba_diff >= min_confidence) & trend_up
                # Short: только в даунтренде или нейтрально  
                short_mask = (proba_short >= thr_short) & (proba_long < thr_long) & (proba_diff <= -min_confidence) & trend_down
            else:
                # Fallback без фильтра тренда
                long_mask = (proba_long >= thr_long) & (proba_short < thr_short) & (proba_diff >= min_confidence)
                short_mask = (proba_short >= thr_short) & (proba_long < thr_long) & (proba_diff <= -min_confidence)
            
            preds_val[long_mask] = 1
            preds_val[short_mask] = -1
            
            # OPTIMIZED: Use sparse simulation - only process bars where we might have activity
            # Get bars from session start before first signal to session end after last signal
            signal_indices = preds_val[preds_val != 0].index
            if len(signal_indices) == 0:
                scores.append(-30)
                continue
            # Find range with buffer for session boundaries
            first_sig = signal_indices.min()
            last_sig = signal_indices.max()
            # Get data range with some buffer for session boundaries
            buffer_bars = horizon * 3  # Buffer for session ends
            start_idx = max(0, full_data.index.get_loc(first_sig) - buffer_bars) if first_sig in full_data.index else 0
            end_idx = min(len(full_data), full_data.index.get_loc(last_sig) + buffer_bars) if last_sig in full_data.index else len(full_data)
            sim_range = full_data.iloc[start_idx:end_idx]
            
            extended_signals = pd.Series(0, index=sim_range.index)
            extended_signals.loc[extended_signals.index.intersection(preds_val.index)] = preds_val.loc[preds_val.index.intersection(sim_range.index)]
            extended_probas = pd.Series(0.0, index=sim_range.index)
            proba_series = pd.Series(proba_long - proba_short, index=X_val.index)
            extended_probas.loc[extended_probas.index.intersection(proba_series.index)] = proba_series.loc[proba_series.index.intersection(sim_range.index)]
            
            # Предвычисленные вероятности для MODEL_EXIT (быстрый lookup)
            proba_long_series = pd.Series(proba_long, index=X_val.index)
            proba_short_series = pd.Series(proba_short, index=X_val.index)
            
            portfolio, trades = sim.simulate_with_horizon(
                sim_range,
                extended_signals,
                extended_probas,
                horizon=horizon,
                stop_mult=stop_mult,
                take_mult=take_mult,
                trailing_mult=trailing_mult,
                breakeven_mult=breakeven_mult,
                proba_long_series=proba_long_series,
                proba_short_series=proba_short_series,
                exit_confidence_drop=exit_confidence_drop,
            )
            returns = np.diff(portfolio) / portfolio[:-1]
            num_trades = len([t for t in trades if t.get("exit_time")])
            if num_trades < 5 or len(returns) < 2:
                scores.append(-30)
                continue
            downside = returns[returns < 0]
            if len(downside) > 1 and downside.std() > 0:
                sortino = returns.mean() / downside.std() * np.sqrt(252 * 24 * 60)
            else:
                sortino = 0
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1
            
            # НОВОЕ: Добавляем PnL в score для прямой оптимизации прибыли
            final_pnl = (portfolio[-1] / portfolio[0] - 1) * 100  # % прибыли
            
            # Комбинированный score: Sortino * PF * 50 + PnL * 10
            # Это заставляет Optuna учитывать реальную прибыль, а не только risk-adjusted метрики
            combined_score = sortino * profit_factor * 50 + final_pnl * 10
            scores.append(combined_score)

        if not scores:
            return -50
        
        # НОВОЕ: Штраф за низкую консистентность между CV-окнами (high variance = overfitting)
        score_std = np.std(scores) if len(scores) > 1 else 0
        mean_score = float(np.mean(scores))
        # Штраф за высокую дисперсию (нестабильность)
        consistency_penalty = min(score_std * 0.5, mean_score * 0.3)  # max 30% штраф
        
        return mean_score - consistency_penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    # ========== ENSEMBLE TOP-5 TRIALS ==========
    # Вместо одного лучшего trial, усредняем top-5 для robustness
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1000, reverse=True)[:5]
    
    if len(top_trials) < 3:
        return study.best_params
    
    # Усредняем числовые параметры из top-5
    ensemble_params = {}
    param_names = list(study.best_params.keys())
    
    for param in param_names:
        values = [t.params.get(param) for t in top_trials if param in t.params]
        if values:
            if isinstance(values[0], int):
                ensemble_params[param] = int(np.median(values))  # median для int
            else:
                ensemble_params[param] = float(np.mean(values))  # mean для float
    
    print(f"    Ensemble из top-{len(top_trials)} trials (scores: {[f'{t.value:.1f}' for t in top_trials]})")
    
    return ensemble_params


def analyze_trades(trades: List[Dict]) -> Dict[str, Any]:
    closed = [t for t in trades if t.get("exit_time") is not None]
    if not closed:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "profit_factor": 0,
            "largest_win_pct": 0,
            "largest_loss_pct": 0,
            "avg_holding_minutes": 0,
        }
    wins = [t for t in closed if t.get("pnl_pct", 0) > 0]
    losses = [t for t in closed if t.get("pnl_pct", 0) < 0]
    total_profit = sum([t.get("pnl_usd", 0) for t in wins])
    total_loss = abs(sum([t.get("pnl_usd", 0) for t in losses]))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
    holding = [t.get("holding_minutes", 0) for t in closed if t.get("holding_minutes", 0) > 0]
    return {
        "total_trades": len(closed),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(closed) if closed else 0,
        "avg_win_pct": np.mean([t.get("pnl_pct", 0) for t in wins]) if wins else 0,
        "avg_loss_pct": np.mean([t.get("pnl_pct", 0) for t in losses]) if losses else 0,
        "profit_factor": profit_factor,
        "largest_win_pct": max([t.get("pnl_pct", 0) for t in wins], default=0),
        "largest_loss_pct": min([t.get("pnl_pct", 0) for t in losses], default=0),
        "avg_holding_minutes": np.mean(holding) if holding else 0,
    }


def analyze_trades_by_direction(trades: List[Dict]) -> Dict[str, Any]:
    long_trades = [t for t in trades if t.get("direction") == "LONG"]
    short_trades = [t for t in trades if t.get("direction") == "SHORT"]
    return {
        "long": analyze_trades(long_trades),
        "short": analyze_trades(short_trades),
        "total_trades": len(trades),
        "long_count": len(long_trades),
        "short_count": len(short_trades),
    }


def analyze_trades_detailed(trades: List[Dict], data: pd.DataFrame | None = None) -> Dict[str, Any]:
    if not trades:
        return {"error": "Нет сделок"}
    closed_trades = [t for t in trades if t.get("exit_time")]
    if not closed_trades:
        return {"error": "Нет закрытых сделок"}
    
    # Сохраняем список сделок для визуализации
    trades_list = []
    for t in closed_trades:
        trades_list.append({
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "direction": t.get("direction", "long"),
            "entry_price": float(t.get("entry_price", 0)),
            "exit_price": float(t.get("exit_price", 0)),
            "pnl_pct": float(t.get("pnl_pct", 0)),
            "pnl_usd": float(t.get("pnl_usd", 0)),
            "exit_reason": t.get("exit_reason", "UNKNOWN"),
            "holding_time_minutes": float(t.get("holding_minutes", 0)),
            "confidence": float(t.get("confidence", 0)),
        })
    
    results: Dict[str, Any] = {
        "summary": analyze_trades(closed_trades),
        "by_direction": analyze_trades_by_direction(closed_trades),
        "trades_list": trades_list,  # Добавлено для визуализации
    }
    holding_times = [t.get("holding_minutes", 0) for t in closed_trades]
    results["holding_time_stats"] = {
        "mean": float(np.mean(holding_times)),
        "median": float(np.median(holding_times)),
        "min": float(np.min(holding_times)),
        "max": float(np.max(holding_times)),
        "std": float(np.std(holding_times)),
    }
    exit_reasons: Dict[str, int] = {}
    for trade in closed_trades:
        reason = trade.get("exit_reason", "UNKNOWN")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    results["exit_reasons"] = exit_reasons

    if data is not None:
        hourly_pnl: Dict[int, Dict[str, float]] = {}
        for trade in closed_trades:
            entry_time = trade.get("entry_time")
            if entry_time is None or entry_time not in data.index:
                continue
            hour = entry_time.hour
            pnl = trade.get("pnl_usd", 0)
            if hour not in hourly_pnl:
                hourly_pnl[hour] = {"total_pnl": 0.0, "count": 0, "wins": 0}
            hourly_pnl[hour]["total_pnl"] += pnl
            hourly_pnl[hour]["count"] += 1
            if pnl > 0:
                hourly_pnl[hour]["wins"] += 1
        hourly_stats: Dict[int, Dict[str, float]] = {}
        for hour, stats in hourly_pnl.items():
            hourly_stats[hour] = {
                "avg_pnl": stats["total_pnl"] / stats["count"],
                "win_rate": stats["wins"] / stats["count"],
                "total_trades": stats["count"],
                "total_pnl": stats["total_pnl"],
            }
        results["hourly_performance"] = hourly_stats

    if all("confidence" in t for t in closed_trades):
        confidences = [t.get("confidence", 0) for t in closed_trades]
        pnls = [t.get("pnl_usd", 0) for t in closed_trades]
        if len(confidences) > 4:
            conf_quartiles = pd.qcut(confidences, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            quartile_stats: Dict[str, Dict[str, float]] = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                mask = conf_quartiles == q
                mask_arr = np.asarray(mask)
                if mask_arr.any():
                    qs = [pnls[i] for i in range(len(pnls)) if mask_arr[i]]
                    cs = [confidences[i] for i in range(len(confidences)) if mask_arr[i]]
                    quartile_stats[q] = {
                        "avg_pnl": float(np.mean(qs)),
                        "win_rate": float(sum(1 for p in qs if p > 0) / len(qs)),
                        "avg_confidence": float(np.mean(cs)),
                        "trades": int(len(qs)),
                    }
            results["confidence_quartiles"] = quartile_stats
    return results


def analyze_threshold_sensitivity(X_test: pd.DataFrame, y_test: pd.Series, model_long, model_short) -> List[Dict]:
    proba_long = safe_predict_proba(model_long, X_test)
    proba_short = safe_predict_proba(model_short, X_test)
    thresholds = np.arange(0.3, 0.81, 0.05)
    results = []
    for thr in thresholds:
        preds = pd.Series(0, index=X_test.index)
        preds[(proba_long >= thr) & (proba_short < thr)] = 1
        preds[(proba_short >= thr) & (proba_long < thr)] = -1
        total_signals = int((preds != 0).sum())
        long_signals = int((preds == 1).sum())
        short_signals = int((preds == -1).sum())
        y_test_binary = (y_test != 0).astype(int) if y_test is not None else None
        preds_binary = (preds != 0).astype(int)
        accuracy = float(accuracy_score(y_test_binary, preds_binary)) if y_test_binary is not None and len(y_test_binary) > 0 else 0.0
        results.append(
            {
                "threshold": float(thr),
                "total_signals": total_signals,
                "long_signals": long_signals,
                "short_signals": short_signals,
                "signal_ratio": float(total_signals / len(X_test)) if len(X_test) > 0 else 0.0,
                "accuracy": accuracy,
            }
        )
    return results


def create_diagnostic_plots(portfolio: np.ndarray, trades: List[Dict], X_test: pd.DataFrame, y_test: pd.Series, test_preds: pd.Series, horizon: int):
    # Equity
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio, label=f"Equity (PnL: {(portfolio[-1]/10000-1)*100:.1f}%)")
    plt.axhline(y=10000, color="r", linestyle="--", alpha=0.3, label="Initial")
    plt.xlabel("Bar")
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Equity Curve - Horizon {horizon} min")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / f"equity_curve_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()

    peak = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - peak) / peak * 100
    plt.figure(figsize=(12, 4))
    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color="red")
    plt.plot(drawdown, color="red", linewidth=1)
    plt.xlabel("Bar")
    plt.ylabel("Drawdown (%)")
    plt.title(f"Drawdown - Horizon {horizon} min (Max: {abs(drawdown.min()):.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / f"drawdown_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()

    if trades:
        closed_trades = [t for t in trades if t.get("exit_time")]
        if closed_trades:
            pnls = [t.get("pnl_usd", 0) for t in closed_trades]
            directions = [t.get("direction", "UNKNOWN") for t in closed_trades]
            plt.figure(figsize=(10, 6))
            bins = np.linspace(min(pnls), max(pnls), 20) if len(set(pnls)) > 1 else 10
            long_pnls = [pnls[i] for i in range(len(pnls)) if directions[i] == "LONG"]
            short_pnls = [pnls[i] for i in range(len(pnls)) if directions[i] == "SHORT"]
            if long_pnls:
                plt.hist(long_pnls, bins=bins, alpha=0.5, label=f"Long ({len(long_pnls)})", color="green")
            if short_pnls:
                plt.hist(short_pnls, bins=bins, alpha=0.5, label=f"Short ({len(short_pnls)})", color="red")
            plt.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            plt.xlabel("Trade PnL ($)")
            plt.ylabel("Frequency")
            plt.title(f"Trade PnL Distribution - Horizon {horizon} min")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(FIGURES_DIR / f"trade_pnl_dist_h{horizon}.png", dpi=150, bbox_inches="tight")
            plt.close()

            if all("confidence" in t for t in closed_trades):
                confidences = [t.get("confidence", 0) for t in closed_trades]
                plt.figure(figsize=(10, 6))
                plt.scatter(confidences, pnls, alpha=0.5)
                plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
                plt.xlabel("Model Confidence")
                plt.ylabel("Trade PnL ($)")
                plt.title(f"Confidence vs PnL - Horizon {horizon} min")
                if len(set(confidences)) > 1:
                    z = np.polyfit(confidences, pnls, 1)
                    p = np.poly1d(z)
                    plt.plot(confidences, p(confidences), "r--", alpha=0.8, label=f"Trend: y={z[0]:.1f}x{z[1]:+.1f}")
                    plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(FIGURES_DIR / f"confidence_vs_pnl_h{horizon}.png", dpi=150, bbox_inches="tight")
                plt.close()


def save_detailed_results(
    results: Dict,
    horizon: int,
    feat_analysis_long: Dict,
    feat_analysis_short: Dict,
    detailed_trade_analysis: Dict,
    threshold_analysis: List[Dict],
    session_cfg: Dict,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_path = RESULTS_DIR / f"detailed_results_h{horizon}_{ts}.json"
    detailed_results = {
        "horizon": horizon,
        "timestamp": ts,
        "metrics": results.get("metrics", {}),
        "financial": {
            "pnl": results.get("pnl", 0),
            "sharpe": results.get("sharpe", 0),
            "max_drawdown": results.get("max_drawdown", 0),
        },
        "feature_analysis": {"long_model": feat_analysis_long, "short_model": feat_analysis_short},
        "trades_detailed": detailed_trade_analysis,
        "threshold_sensitivity": threshold_analysis,
        "best_params": results.get("best_params", {}),
        "config": {"session_config": session_cfg, "sim_config": SIM_CFG},
    }
    with open(res_path, "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    summary_path = RESULTS_DIR / f"summary_h{horizon}_{ts}.csv"
    summary_data = {
        "horizon": [horizon],
        "timestamp": [ts],
        "pnl": [results.get("pnl", 0)],
        "sharpe": [results.get("sharpe", 0)],
        "max_drawdown": [results.get("max_drawdown", 0)],
        "total_trades": [detailed_trade_analysis.get("summary", {}).get("total_trades", 0)],
        "win_rate": [detailed_trade_analysis.get("summary", {}).get("win_rate", 0)],
        "profit_factor": [detailed_trade_analysis.get("summary", {}).get("profit_factor", 0)],
        "long_accuracy": [results.get("metrics", {}).get("long_accuracy", 0)],
        "short_accuracy": [results.get("metrics", {}).get("short_accuracy", 0)],
        "long_trades": [detailed_trade_analysis.get("by_direction", {}).get("long_count", 0)],
        "short_trades": [detailed_trade_analysis.get("by_direction", {}).get("short_count", 0)],
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"Детальные результаты сохранены: {res_path}")
    print(f"Сводка сохранена: {summary_path}")


def save_results(results: Dict, horizon: int) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_path = RESULTS_DIR / f"trading_results_h{horizon}_{ts}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Результаты сохранены: {res_path}")


def main():
    print("=== Phase 2 v3: Train/Val/Test ===")
    data_path = DATA_DIR / "btc_processed_v3.pkl"
    if not data_path.exists():
        raise FileNotFoundError("Запустите phase1_sessions_v3.py для генерации btc_processed_v3.pkl")
    pkg = pickle.load(open(data_path, "rb"))
    train_data = pkg["train_data"]
    val_data = pkg["val_data"]
    test_data = pkg["test_data"]
    session_cfg = pkg.get("config", {}).get("session_config", {})
    horizons = [15]  # Фокус только на H15 - единственный прибыльный горизонт
    summary = []

    print("\n[Data check] Валидация фич и таргетов:")
    for split_name, split_df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"  Split {split_name}:")
        validate_features_extended(split_df, split_name)
        validate_targets(split_df, horizons)

    validate_temporal_alignment(train_data, horizons)
    # SKIP heavy validations for faster execution
    # detailed_alignment = validate_temporal_alignment_detailed(train_data, horizons)
    # if detailed_alignment:
    #     issues = [k for k, v in detailed_alignment.items() if v.get("issues")]
    #     if issues:
    #         print(f"[Temporal alignment] Issues detected: {', '.join(issues)}")
    # comprehensive_alignment = validate_temporal_alignment_comprehensive(train_data, horizons)
    # if comprehensive_alignment:
    #     issues = [k for k, v in comprehensive_alignment.items() if v.get("has_issues")]
    #     if issues:
    #         print(f"[Temporal alignment] Comprehensive issues: {', '.join(issues)}")

    for h_idx, h in enumerate(horizons, start=1):
        if session_cfg and session_cfg.get("session_len_min", h * 2) < h:
            print(f"⚠️ Длительность сессии {session_cfg.get('session_len_min')} мин меньше горизонта {h} мин, пропускаем.")
            continue
        _progress("Horizons", h_idx - 1, len(horizons))
        print(f"\n=== Горизонт {h} минут ===")
        target_col = ModelManager.select_best_target(train_data, h)
        X_train, y_train = ModelManager.prepare_features_optimized(train_data, h, target_col)
        X_val, y_val = ModelManager.prepare_features_optimized(val_data, h, target_col)
        X_test, y_test = ModelManager.prepare_features_optimized(test_data, h, target_col)
        if len(X_train) < 100 or len(X_val) < 50 or len(X_test) < 50:
            print("Недостаточно данных, пропускаем.")
            continue
        # Используем train+val для подбора, чтобы параметры были ближе к продакшену; тест остаётся unseen
        X_dev = pd.concat([X_train, X_val]).sort_index()
        y_dev = pd.concat([y_train, y_val]).reindex(X_dev.index)
        dev_data = pd.concat([train_data.loc[X_train.index], val_data.loc[X_val.index]]).sort_index()
        # CRITICAL: Pass FULL data (including out-of-session bars) for proper SESSION_END handling
        full_dev_data = pd.concat([train_data, val_data]).sort_index()
        best_params = optimize_with_walkforward(X_dev, y_dev, dev_data, h, n_trials=100, full_data=full_dev_data)
        print("=" * 50)
        print(f"ДЕТАЛЬНЫЙ АНАЛИЗ H{h}")
        print("=" * 50)
        # финальные модели лонг/шорт
        y_dev_long = (y_dev == 1).astype(int)
        y_dev_short = (y_dev == -1).astype(int)
        model_long = RandomForestClassifier(
            n_estimators=best_params.get("n_estimators", 120),
            max_depth=best_params.get("max_depth", 10),
            min_samples_split=best_params.get("min_samples_split", 20),
            min_samples_leaf=best_params.get("min_samples_leaf", 10),
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        model_short = RandomForestClassifier(
            n_estimators=best_params.get("n_estimators", 120),
            max_depth=best_params.get("max_depth", 10),
            min_samples_split=best_params.get("min_samples_split", 20),
            min_samples_leaf=best_params.get("min_samples_leaf", 10),
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        model_long.fit(X_dev, y_dev_long)
        model_short.fit(X_dev, y_dev_short)
        thr_long = best_params.get("thr_long", 0.6)
        thr_short = best_params.get("thr_short", 0.6)
        min_confidence = best_params.get("min_confidence", 0.1)
        
        # === СОХРАНЯЕМ МОДЕЛИ И ПАРАМЕТРЫ ДЛЯ REALTIME BACKTESTER ===
        model_save_path = MODELS_DIR / f"phase2_h{h}_final.pkl"
        model_package = {
            'model_long': model_long,
            'model_short': model_short,
            'best_params': best_params,
            'feature_cols': list(X_dev.columns),
            'target_col': target_col,
            'horizon': h,
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"    Модели сохранены: {model_save_path}")
        # === КОНЕЦ СОХРАНЕНИЯ ===
        
        proba_long = safe_predict_proba(model_long, X_test)
        proba_short = safe_predict_proba(model_short, X_test)
        proba_diff = proba_long - proba_short
        if proba_long.sum() == 0 or proba_short.sum() == 0:
            print("⚠️ Модели вернули только один класс на тесте, пропускаем чувствительность порогов для этого горизонта.")
        test_preds = pd.Series(0, index=X_test.index)
        
        # TREND FILTER для финальных предсказаний
        # FIX: используем 'Close' с заглавной буквы (как в данных)
        close_prices = test_data['Close'] if 'Close' in test_data.columns else (test_data['close'] if 'close' in test_data.columns else None)
        sma_col = 'sma_20' if 'sma_20' in test_data.columns else ('sma_14' if 'sma_14' in test_data.columns else None)
        
        if close_prices is not None and sma_col is not None:
            close_test = test_data.loc[X_test.index, 'Close'] if 'Close' in test_data.columns else test_data.loc[X_test.index, 'close']
            sma_test = test_data.loc[X_test.index, sma_col]
            trend_up = close_test > sma_test
            trend_down = close_test < sma_test
            
            # Long только в аптренде
            long_mask = (proba_long >= thr_long) & (proba_short < thr_short) & (proba_diff >= min_confidence) & trend_up
            # Short только в даунтренде
            short_mask = (proba_short >= thr_short) & (proba_long < thr_long) & (proba_diff <= -min_confidence) & trend_down
            
            test_preds[long_mask] = 1
            test_preds[short_mask] = -1
            print(f"    [Trend Filter] Отфильтровано: {(~trend_up & (proba_long >= thr_long)).sum()} лонгов, {(~trend_down & (proba_short >= thr_short)).sum()} шортов")
        else:
            test_preds[(proba_long >= thr_long) & (proba_short < thr_short) & (proba_diff >= min_confidence)] = 1
            test_preds[(proba_short >= thr_short) & (proba_long < thr_long) & (proba_diff <= -min_confidence)] = -1
        y_train_long = (y_train == 1).astype(int)
        y_train_short = (y_train == -1).astype(int)
        feat_analysis_long = EnhancedModelManager.analyze_feature_importance(X_train, y_train_long, "long")
        feat_analysis_short = EnhancedModelManager.analyze_feature_importance(X_train, y_train_short, "short")
        y_test_long = (y_test == 1).astype(int)
        y_test_short = (y_test == -1).astype(int)
        preds_long = (test_preds == 1).astype(int)
        preds_short = (test_preds == -1).astype(int)
        metrics = {
            "long_accuracy": accuracy_score(y_test_long, preds_long),
            "short_accuracy": accuracy_score(y_test_short, preds_short),
            "long_precision": precision_score(y_test_long, preds_long, zero_division=0),
            "short_precision": precision_score(y_test_short, preds_short, zero_division=0),
            "long_recall": recall_score(y_test_long, preds_long, zero_division=0),
            "short_recall": recall_score(y_test_short, preds_short, zero_division=0),
            "long_f1": f1_score(y_test_long, preds_long, zero_division=0),
            "short_f1": f1_score(y_test_short, preds_short, zero_division=0),
        }
        sim = HorizonAwareSimulator()
        # CRITICAL FIX: Use FULL data range (including out-of-session bars) for proper SESSION_END handling
        # But signals are only generated for in-session bars (X_test.index)
        full_data_range = test_data.loc[X_test.index.min():X_test.index.max()]
        # Extend signals to full range (0 for out-of-session bars)
        extended_signals = pd.Series(0, index=full_data_range.index)
        extended_signals.loc[test_preds.index] = test_preds
        extended_probas = pd.Series(0.0, index=full_data_range.index)
        proba_series = pd.Series(proba_long - proba_short, index=X_test.index)
        extended_probas.loc[proba_series.index] = proba_series
        
        # Предвычисленные вероятности для MODEL_EXIT
        proba_long_series = pd.Series(proba_long, index=X_test.index)
        proba_short_series = pd.Series(proba_short, index=X_test.index)
        
        portfolio, trades = sim.simulate_with_horizon(
            full_data_range,
            extended_signals,
            extended_probas,
            horizon=h,
            stop_mult=best_params.get("stop_mult", 1.0),
            take_mult=best_params.get("take_mult", 2.0),
            trailing_mult=best_params.get("trailing_mult", 0.0),
            breakeven_mult=best_params.get("breakeven_mult", 0.0),
            proba_long_series=proba_long_series,
            proba_short_series=proba_short_series,
            exit_confidence_drop=best_params.get("exit_confidence_drop", 0.0),
        )
        pnl = (portfolio[-1] / 10000 - 1) * 100
        overall_stats = analyze_trades(trades)
        trade_stats = analyze_trades_by_direction(trades)
        sharpe = calculate_sharpe(portfolio)
        mdd = max_drawdown(portfolio)
        detailed_trade_analysis = analyze_trades_detailed(trades, full_data_range)
        threshold_analysis = analyze_threshold_sensitivity(X_test, y_test, model_long, model_short)
        create_diagnostic_plots(portfolio, trades, X_test, y_test, test_preds, h)
        
        # Комплексный анализ влияния фич (помогают/вредят)
        print(f"\n[0/4] Комплексный анализ влияния фич для H{h}:")
        feature_impact = FeatureImpactAnalyzer.full_analysis(
            X_train, y_train_long,  # анализируем для long модели
            X_val, y_val,
            model_long,
            trades,
            test_data.loc[X_test.index],
        )
        
        print(f"\n[1/4] Важность фич (лонг/шорт):")
        for i, feat in enumerate(feat_analysis_long["top_10_features"][:5], 1):
            print(f"    LONG {i}. {feat['feature']}: imp={feat['importance']:.4f}, MI={feat.get('mutual_info', 0):.4f}")
        for i, feat in enumerate(feat_analysis_short["top_10_features"][:5], 1):
            print(f"    SHORT {i}. {feat['feature']}: imp={feat['importance']:.4f}, MI={feat.get('mutual_info', 0):.4f}")
        print(f"\n[2/4] Детальный анализ сделок: всего {detailed_trade_analysis.get('summary', {}).get('total_trades', 0)}")
        if "summary" in detailed_trade_analysis:
            print(f"    Win Rate: {detailed_trade_analysis['summary'].get('win_rate',0):.1%}, PF: {detailed_trade_analysis['summary'].get('profit_factor',0):.2f}")
        if "by_direction" in detailed_trade_analysis:
            dir_stats = detailed_trade_analysis["by_direction"]
            print(f"    Лонги: {dir_stats.get('long_count',0)}, Шорты: {dir_stats.get('short_count',0)}")
        if "exit_reasons" in detailed_trade_analysis:
            print("    Причины выхода:")
            for reason, count in detailed_trade_analysis["exit_reasons"].items():
                print(f"      {reason}: {count}")
        print(f"\n[3/4] Чувствительность порогов (текущее thr_long={thr_long:.2f}, thr_short={thr_short:.2f}):")
        for entry in threshold_analysis:
            print(f"    thr={entry['threshold']:.2f}: signals={entry['total_signals']} (long {entry['long_signals']}, short {entry['short_signals']}), acc={entry['accuracy']:.1%}")
        print(f"\n[4/4] Диагностические графики сохранены в {FIGURES_DIR}")
        results = {
            "horizon": h,
            "metrics": metrics,
            "pnl": pnl,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "trades": trade_stats,
            "trades_overall": overall_stats,
            "best_params": best_params,
        }
        summary.append({"horizon": h, "pnl": pnl, "win_rate": overall_stats["win_rate"], "sharpe": sharpe})
        save_detailed_results(results, h, feat_analysis_long, feat_analysis_short, detailed_trade_analysis, threshold_analysis, session_cfg)
        # confusion plot
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"phase2_v3_cm_h{h}.png", dpi=150)
        plt.close()
        _progress("Horizons", h_idx, len(horizons))

    if summary:
        summary_df = pd.DataFrame(summary)
        print("\nСводка по горизонтам:")
        print(summary_df.to_string(index=False))
    else:
        print("Нет успешных запусков.")


if __name__ == "__main__":
    main()
