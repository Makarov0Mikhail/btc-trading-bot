# Подавляем warnings ДО любых импортов
import warnings
import os
import sys
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['JOBLIB_VERBOSITY'] = '0'
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='joblib')

# Подавляем все warnings от sklearn/joblib
def warn_with_traceback(*args, **kwargs):
    pass
warnings.warn = warn_with_traceback

"""
Реалистичный бэктестер для торговли BTC в режиме реального времени.

Этот модуль симулирует работу в реальном времени:
1. Получает свечи по одной (как с биржи)
2. Детектит сессии на лету
3. Рассчитывает фичи в реальном времени
4. Модель принимает решения о входе/выходе

Для перехода на реальную биржу нужно заменить:
- CandleProvider на реальный источник данных
- OrderExecutor на реальное API биржи
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

# Пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class SessionConfig:
    """Конфигурация детектора сессий (идентична AdaptiveSessionDetector из phase1_sessions_v3.py)."""
    prebuffer_min: int = 45
    session_len_min: int = 45
    volatility_window: int = 10
    lookback_days: int = 90
    percentile: float = 95  # ВАЖНО: 95 как в AdaptiveSessionDetector, не 97!
    min_vol_multiplier: float = 1.5
    cooldown_min: int = 30


@dataclass
class TradingConfig:
    """Конфигурация торговли."""
    horizon: int = 15
    max_trades_per_session: int = 2  # До 2 сделок на сессию
    position_pct: float = 1.0  # 100% капитала
    commission: float = 0.0007
    slippage: float = 0.0001  # 0.01% slippage для реалистичности
    
    # Параметры модели (лучшие из оптимизации)
    thr_long: float = 0.6063
    thr_short: float = 0.6156
    min_confidence: float = 0.1562
    stop_mult: float = 0.9694
    take_mult: float = 1.8170
    trailing_mult: float = 0.6682
    breakeven_mult: float = 0.6466
    exit_confidence_drop: float = 0.1349


@dataclass
class ModelConfig:
    """Конфигурация RandomForest."""
    n_estimators: int = 118
    max_depth: int = 12
    min_samples_split: int = 31
    min_samples_leaf: int = 22


# ============================================================================
# РАСЧЁТ ФИЧ В РЕАЛЬНОМ ВРЕМЕНИ
# ============================================================================

class RealtimeFeatureCalculator:
    """Расчёт фичей в реальном времени без look-ahead bias."""
    
    # Фичи для H15 - ТОЛЬКО ТЕ ЧТО РЕАЛЬНО ЕСТЬ В ДАННЫХ
    # (11 фич как в оригинале после фильтрации existing_features)
    FEATURE_COLS = [
        "log_return", "log_return_lag_2", 
        "hour_sin", "hour_cos", "is_weekend",
        "prev_candle_body", 
        "volume_change", "volume_change_lag_1",
        "atr_pct", "rsi_14", "macd_hist",
    ]
    
    # Дополнительные фичи для trend filter (не для модели)
    EXTRA_COLS = ["sma_14"]
    
    def __init__(self, lookback_bars: int = 100):
        """
        Args:
            lookback_bars: сколько баров хранить для расчёта фичей
        """
        self.lookback_bars = lookback_bars
        self.candles: List[Dict] = []
        self._df_cache: Optional[pd.DataFrame] = None
    
    def add_candle(self, candle: Dict) -> None:
        """Добавить новую свечу."""
        self.candles.append(candle)
        # Храним только последние N баров
        if len(self.candles) > self.lookback_bars:
            self.candles = self.candles[-self.lookback_bars:]
        self._df_cache = None
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Конвертировать свечи в DataFrame."""
        if self._df_cache is not None:
            return self._df_cache
        
        df = pd.DataFrame(self.candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })
        self._df_cache = df
        return df
    
    def calculate_features(self) -> Optional[pd.Series]:
        """Рассчитать фичи для последнего бара."""
        if len(self.candles) < 30:  # Минимум для RSI и прочего
            return None
        
        df = self._to_dataframe()
        
        # === Базовые фичи ===
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return"] = df["Close"].pct_change(1)
        df["log_volume"] = np.log(df["Volume"].replace(0, np.nan)).ffill().bfill()
        
        # Лаги OHLCV
        df["close_lag1"] = df["Close"].shift(1)
        df["open_lag1"] = df["Open"].shift(1)
        df["high_lag1"] = df["High"].shift(1)
        df["low_lag1"] = df["Low"].shift(1)
        
        # Скользящие средние - для H15: ma_windows=[5,10,15], volatility_windows=[10,15,20]
        for w in [5, 7, 10, 14, 15, 20]:
            df[f"sma_{w}"] = df["Close"].rolling(w).mean().shift(1)
        for w in [7, 10, 14, 15, 20]:
            df[f"logret_std_{w}"] = df["log_return"].rolling(w).std().shift(1)
            df[f"logret_mean_{w}"] = df["log_return"].rolling(w).mean().shift(1)
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().shift(1)
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().shift(1)
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14).mean().shift(1)
        df["atr_pct"] = atr_14 / df["Close"].shift(1)
        
        # MACD - ИДЕНТИЧНО phase1_sessions_v3.py
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (exp1 - exp2).shift(1)
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().shift(1)
        df["macd_hist"] = (df["macd"] - df["macd_signal"]).shift(1)
        
        # Свечные паттерны
        df["prev_range_pct"] = (df["High"].shift(1) - df["Low"].shift(1)) / df["Close"].shift(1)
        df["prev_body_pct"] = (df["Close"].shift(1) - df["Open"].shift(1)) / df["Open"].shift(1)
        
        # Время
        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Производные
        df["prev_candle_body"] = df["prev_body_pct"].shift(1)
        df["prev_candle_range"] = df["prev_range_pct"].shift(1)
        df["volume_change"] = df["Volume"] / df["Volume"].shift(1)
        
        # Лаги
        df["log_return_lag_1"] = df["log_return"].shift(1)
        df["log_return_lag_2"] = df["log_return"].shift(2)
        df["volume_change_lag_1"] = df["volume_change"].shift(1)
        
        # Возвращаем последнюю строку
        last_row = df.iloc[-1]
        
        # Для модели - только FEATURE_COLS (11 фич)
        all_cols = self.FEATURE_COLS + self.EXTRA_COLS
        features = last_row[[c for c in all_cols if c in df.columns]]
        
        if features[self.FEATURE_COLS].isna().any():
            return None
        
        return features


# ============================================================================
# ДЕТЕКТОР СЕССИЙ В РЕАЛЬНОМ ВРЕМЕНИ (идентичен phase1_sessions_v3.py)
# ============================================================================

class RealtimeSessionDetector:
    """
    Детектор сессий в реальном времени.
    
    ИДЕНТИЧЕН логике из phase1_sessions_v3.py SessionDetector:
    1. short_vol = log_return.rolling(volatility_window).std() на ПРЕДЫДУЩЕМ баре (shift(1))
    2. threshold = max(percentile_val, median * min_vol_multiplier) на ПРЕДЫДУЩЕМ баре
    3. Старт сессии когда short_vol > threshold И предыдущий short_vol <= threshold
    """
    
    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self.sessions: List[Dict] = []
        self.current_session: Optional[Dict] = None
        self.last_session_end: Optional[datetime] = None
        
        # Храним историю log_returns для расчёта rolling volatility
        # Нужно lookback_days * 1440 + volatility_window баров
        self.lookback_bars = self.config.lookback_days * 24 * 60 + self.config.volatility_window + 10
        self.log_returns: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Кэш для rolling volatility (short_vol_for_threshold без shift)
        self._vol_cache: List[float] = []
        
    def _calculate_rolling_std(self, values: List[float], window: int) -> float:
        """Рассчитать rolling std для последних window значений."""
        if len(values) < window:
            return np.nan
        arr = np.array(values[-window:])
        return float(np.std(arr, ddof=1))  # ddof=1 как в pandas
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Рассчитать percentile для массива."""
        arr = np.array(values)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.percentile(valid, percentile))
    
    def _calculate_median(self, values: List[float]) -> float:
        """Рассчитать median для массива."""
        arr = np.array(values)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.median(valid))
    
    def add_candle(self, timestamp: datetime, close: float, prev_close: float) -> None:
        """
        Добавить новую свечу для расчёта волатильности.
        
        Args:
            timestamp: время свечи
            close: цена закрытия
            prev_close: цена закрытия предыдущей свечи
        """
        # Рассчитываем log_return
        if prev_close > 0:
            lr = np.log(close / prev_close)
        else:
            lr = 0.0
        
        self.log_returns.append(lr)
        self.timestamps.append(timestamp)
        
        # Рассчитываем short_vol_for_threshold (БЕЗ shift - текущий бар)
        # Это нужно для rolling percentile/median
        vol_window = self.config.volatility_window
        if len(self.log_returns) >= vol_window:
            current_vol = self._calculate_rolling_std(self.log_returns, vol_window)
        else:
            current_vol = np.nan
        self._vol_cache.append(current_vol)
        
        # Обрезаем историю до lookback_bars
        if len(self.log_returns) > self.lookback_bars:
            self.log_returns = self.log_returns[-self.lookback_bars:]
            self.timestamps = self.timestamps[-self.lookback_bars:]
            self._vol_cache = self._vol_cache[-self.lookback_bars:]
    
    def update(self, timestamp: datetime) -> Optional[Dict]:
        """
        Проверить условия старта сессии на текущем баре.
        
        Логика идентична phase1_sessions_v3.py:
        - short_vol = lr.rolling(vol_window).std().shift(1) -> используем iloc[-2] из _vol_cache
        - percentile_val = short_vol_for_threshold.rolling(lookback).quantile(percentile).shift(1)
        - median_vol = short_vol_for_threshold.rolling(lookback).median().shift(1)
        - threshold = max(percentile_val, median_vol * min_vol_multiplier)
        - high_vol = short_vol > threshold
        - starts = high_vol & (~high_vol.shift(1))
        
        Returns:
            Новая сессия если началась, иначе None
        """
        # Проверяем текущую сессию
        if self.current_session:
            if timestamp >= self.current_session['session_end']:
                self.current_session = None
        
        # Если уже в сессии - не создаём новую
        if self.current_session:
            return None
        
        # Проверяем cooldown
        if self.last_session_end:
            cooldown_end = self.last_session_end + timedelta(minutes=self.config.cooldown_min)
            if timestamp < cooldown_end:
                return None
        
        # Минимум данных для расчёта
        lookback_bars = self.config.lookback_days * 24 * 60
        if len(self._vol_cache) < lookback_bars + 2:
            return None
        
        # === ИДЕНТИЧНО phase1_sessions_v3.py ===
        # short_vol = lr.rolling(vol_window).std().shift(1)
        # Т.е. волатильность на ПРЕДЫДУЩЕМ баре (iloc[-2] из _vol_cache)
        short_vol = self._vol_cache[-2] if len(self._vol_cache) >= 2 else np.nan
        short_vol_prev = self._vol_cache[-3] if len(self._vol_cache) >= 3 else np.nan
        
        if np.isnan(short_vol):
            return None
        
        # percentile_val = short_vol_for_threshold.rolling(lookback).quantile(percentile).shift(1)
        # median_vol = short_vol_for_threshold.rolling(lookback).median().shift(1)
        # Берём lookback баров ДО текущего (shift(1))
        lookback_data = self._vol_cache[-(lookback_bars + 1):-1]  # исключаем текущий
        
        percentile_val = self._calculate_percentile(lookback_data, self.config.percentile)
        median_vol = self._calculate_median(lookback_data)
        
        if np.isnan(percentile_val) or np.isnan(median_vol):
            return None
        
        # threshold = max(percentile_val, median_vol * min_vol_multiplier)
        min_thr = median_vol * self.config.min_vol_multiplier
        threshold = max(percentile_val, min_thr)
        
        # high_vol = short_vol > threshold
        # starts = high_vol & (~high_vol.shift(1))
        high_vol_current = short_vol > threshold
        
        # Для предыдущего бара нужен предыдущий threshold
        # Но в phase1 threshold также shift(1), поэтому threshold на прошлом баре
        # был бы рассчитан на данных ещё на 1 бар раньше
        lookback_data_prev = self._vol_cache[-(lookback_bars + 2):-2] if len(self._vol_cache) >= lookback_bars + 2 else []
        if len(lookback_data_prev) >= lookback_bars:
            percentile_val_prev = self._calculate_percentile(lookback_data_prev, self.config.percentile)
            median_vol_prev = self._calculate_median(lookback_data_prev)
            min_thr_prev = median_vol_prev * self.config.min_vol_multiplier
            threshold_prev = max(percentile_val_prev, min_thr_prev)
            high_vol_prev = short_vol_prev > threshold_prev if not np.isnan(short_vol_prev) else False
        else:
            high_vol_prev = False
        
        # Старт сессии: текущий high_vol И предыдущий NOT high_vol
        if high_vol_current and not high_vol_prev:
            session = {
                'session_id': len(self.sessions),
                'prebuffer_start': timestamp - timedelta(minutes=self.config.prebuffer_min),
                'session_start': timestamp,
                'session_end': timestamp + timedelta(minutes=self.config.session_len_min),
                'volatility': float(short_vol),
                'threshold': float(threshold),
            }
            self.sessions.append(session)
            self.current_session = session
            self.last_session_end = session['session_end']
            return session
        
        return None
    
    def is_in_session(self, timestamp: datetime) -> Tuple[bool, int, Optional[datetime]]:
        """
        Проверить находимся ли в сессии.
        
        Returns:
            (in_session, session_id, session_end)
        """
        if self.current_session:
            if self.current_session['session_start'] <= timestamp <= self.current_session['session_end']:
                return True, self.current_session['session_id'], self.current_session['session_end']
        return False, -1, None
    
    def get_current_session_end(self) -> Optional[datetime]:
        """Получить время окончания текущей сессии."""
        if self.current_session:
            return self.current_session['session_end']
        return None


# ============================================================================
# МЕНЕДЖЕР ПОЗИЦИЙ
# ============================================================================

@dataclass
class Position:
    """Открытая позиция."""
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss_pct: float
    take_profit_pct: float
    entry_proba_diff: float
    session_id: int
    entry_bar: int
    target_exit_bar: int
    
    # Trailing/Breakeven
    max_pnl: float = 0.0
    current_stop_pct: float = 0.0
    trailing_activated: bool = False
    breakeven_activated: bool = False
    
    def __post_init__(self):
        self.current_stop_pct = self.stop_loss_pct


@dataclass
class Trade:
    """Завершённая сделка."""
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    confidence: float
    session_id: int
    holding_bars: int


# ============================================================================
# ТОРГОВЫЙ ДВИЖОК
# ============================================================================

class TradingEngine:
    """Основной торговый движок."""
    
    def __init__(
        self,
        model_long: RandomForestClassifier,
        model_short: RandomForestClassifier,
        trading_config: TradingConfig = None,
        session_config: SessionConfig = None,
        on_trade_callback: Callable[[Trade], None] = None,
        on_signal_callback: Callable[[str, float, float], None] = None,
    ):
        self.model_long = model_long
        self.model_short = model_short
        self.config = trading_config or TradingConfig()
        self.session_config = session_config or SessionConfig()
        
        self.feature_calculator = RealtimeFeatureCalculator(lookback_bars=200)
        self.session_detector = RealtimeSessionDetector(self.session_config)
        
        self.on_trade = on_trade_callback
        self.on_signal = on_signal_callback
        
        # Состояние
        self.capital = 10000.0
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.bar_index = 0
        self.session_trade_counts: Dict[int, int] = {}
        
        # Статистика
        self.total_signals = 0
        self.total_trades = 0
    
    def reset(self, initial_capital: float = 10000.0):
        """Сбросить состояние."""
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.bar_index = 0
        self.session_trade_counts = {}
        self.total_signals = 0
        self.total_trades = 0
        self.feature_calculator = RealtimeFeatureCalculator(lookback_bars=200)
        self.session_detector = RealtimeSessionDetector(self.session_config)
    
    def process_candle(self, candle: Dict) -> Optional[str]:
        """
        Обработать новую свечу.
        
        Args:
            candle: {'timestamp': datetime, 'open': float, 'high': float, 
                    'low': float, 'close': float, 'volume': float,
                    'in_session': int (optional), 'session_id': int (optional)}
        
        Returns:
            Действие: 'ENTRY_LONG', 'ENTRY_SHORT', 'EXIT', None
        """
        timestamp = candle['timestamp']
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        candle['timestamp'] = timestamp
        
        self.bar_index += 1
        
        # 1. Добавляем свечу в калькулятор фичей
        self.feature_calculator.add_candle(candle)
        
        # 2. Получаем prev_close для расчёта log_return в детекторе сессий
        if len(self.feature_calculator.candles) >= 2:
            prev_close = self.feature_calculator.candles[-2]['close']
        else:
            prev_close = candle['close']
        
        # 3. Обновляем детектор сессий (ВСЕГДА realtime, игнорируем разметку из данных)
        self.session_detector.add_candle(timestamp, candle['close'], prev_close)
        new_session = self.session_detector.update(timestamp)
        
        if new_session:
            print(f"[{timestamp}] Новая сессия #{new_session['session_id']} "
                  f"(vol={new_session['volatility']:.6f}, thr={new_session['threshold']:.6f})")
        
        in_session, session_id, session_end = self.session_detector.is_in_session(timestamp)
        
        # 4. ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ ПРИ ВЫХОДЕ ИЗ СЕССИИ (как в оригинале)
        action = None
        if not in_session:
            if self.position:
                holding_bars = self.bar_index - self.position.entry_bar
                if self.position.direction == 'LONG':
                    pnl_pct = candle['close'] / self.position.entry_price - 1
                else:
                    pnl_pct = self.position.entry_price / candle['close'] - 1
                action = self._execute_exit(candle, timestamp, 'SESSION_END', pnl_pct, holding_bars)
            # Вне сессии - не делаем ничего больше
            return action
        
        # 5. Если есть позиция и в сессии - проверяем выход
        just_exited = False
        if self.position:
            action = self._check_exit(candle, timestamp)
            if action and self.position is None:
                just_exited = True  # Только что закрыли позицию - не открываем новую в этом баре
        
        # 6. Если нет позиции и в сессии - проверяем вход (но не сразу после выхода!)
        if self.position is None and in_session and not just_exited:
            action = self._check_entry(candle, timestamp, session_id)
        
        return action
        
        return action
    
    def _check_entry(self, candle: Dict, timestamp: datetime, session_id: int) -> Optional[str]:
        """Проверить условия входа."""
        # Лимит сделок на сессию
        if self.session_trade_counts.get(session_id, 0) >= self.config.max_trades_per_session:
            return None
        
        # Рассчитываем фичи
        features = self.feature_calculator.calculate_features()
        if features is None:
            return None
        
        # Предсказание модели - только FEATURE_COLS (11 фич), без sma_14
        model_features = features[self.feature_calculator.FEATURE_COLS]
        X = pd.DataFrame([model_features.values], columns=self.feature_calculator.FEATURE_COLS)
        proba_long = self.model_long.predict_proba(X)[0, 1]
        proba_short = self.model_short.predict_proba(X)[0, 1]
        proba_diff = proba_long - proba_short
        
        self.total_signals += 1
        
        # NO TREND FILTER - убран для соответствия phase2
        price = candle['close']
        
        # Проверяем условия входа БЕЗ фильтра тренда
        signal = 0
        # LONG
        if (proba_long >= self.config.thr_long and 
            proba_short < self.config.thr_short and 
            proba_diff >= self.config.min_confidence):
            signal = 1  # LONG
        # SHORT
        elif (proba_short >= self.config.thr_short and 
              proba_long < self.config.thr_long and 
              proba_diff <= -self.config.min_confidence):
            signal = -1  # SHORT
        
        if signal == 0:
            return None
        
        # Рассчитываем размер позиции
        atr_pct = features.get('atr_pct', 0.01)
        if atr_pct <= 0:
            atr_pct = 0.01
        
        # Формула стопа как в phase2: max(default_stop, atr*1.5) * stop_mult
        default_stop = 0.005
        base_stop = max(default_stop, atr_pct * 1.5)
        stop_pct = min(base_stop * self.config.stop_mult, 0.03)  # Cap at 3%
        take_pct = stop_pct * self.config.take_mult
        
        # Размер позиции
        max_capital = self.capital * self.config.position_pct / (1 + self.config.commission)
        size = max_capital / price
        
        # Исполненная цена с учётом slippage
        exec_price = price * (1 + self.config.slippage) if signal == 1 else price * (1 - self.config.slippage)
        
        # Комиссия на вход (от исполненной цены)
        comm_in = exec_price * size * self.config.commission
        
        # Вычитаем: стоимость позиции по исполненной цене + комиссия
        # FIX: раньше вычитали price*size, а при возврате использовали exec_price*size
        self.capital -= (exec_price * size + comm_in)
        
        direction = 'LONG' if signal == 1 else 'SHORT'
        
        self.position = Position(
            direction=direction,
            entry_price=exec_price,
            entry_time=timestamp,
            size=size,
            stop_loss_pct=stop_pct,
            take_profit_pct=take_pct,
            entry_proba_diff=proba_diff if signal == 1 else -proba_diff,
            session_id=session_id,
            entry_bar=self.bar_index,
            target_exit_bar=self.bar_index + self.config.horizon,
        )
        
        self.session_trade_counts[session_id] = self.session_trade_counts.get(session_id, 0) + 1
        
        if self.on_signal:
            self.on_signal(direction, exec_price, proba_diff)
        
        print(f"[{timestamp}] {direction} @ {exec_price:.2f} "
              f"(proba_diff={proba_diff:.3f}, stop={stop_pct*100:.2f}%, take={take_pct*100:.2f}%)")
        
        return f'ENTRY_{direction}'
    
    def _check_exit(self, candle: Dict, timestamp: datetime) -> Optional[str]:
        """Проверить условия выхода."""
        pos = self.position
        price = candle['close']
        
        # Рассчитываем PnL
        if pos.direction == 'LONG':
            pnl_pct = price / pos.entry_price - 1
        else:
            pnl_pct = pos.entry_price / price - 1
        
        # Обновляем max PnL для trailing
        if pnl_pct > pos.max_pnl:
            pos.max_pnl = pnl_pct
        
        holding_bars = self.bar_index - pos.entry_bar
        
        # Trailing stop
        if self.config.trailing_mult > 0 and pos.max_pnl >= pos.stop_loss_pct * self.config.trailing_mult:
            trailing_stop = pos.max_pnl - pos.stop_loss_pct
            if trailing_stop > pos.current_stop_pct:
                pos.current_stop_pct = trailing_stop
                pos.trailing_activated = True
        
        # Breakeven
        if self.config.breakeven_mult > 0 and pos.max_pnl >= pos.stop_loss_pct * self.config.breakeven_mult:
            if not pos.breakeven_activated:
                pos.current_stop_pct = max(pos.current_stop_pct, 0.0001)  # Чуть выше нуля
                pos.breakeven_activated = True
        
        # Проверяем условия выхода
        exit_reason = None
        
        # 1. Horizon exit
        if self.bar_index >= pos.target_exit_bar:
            exit_reason = 'HORIZON_EXIT'
        
        # 2. Stop loss
        elif pnl_pct <= -pos.current_stop_pct:
            exit_reason = 'STOP_LOSS'
        
        # 3. Take profit
        elif pnl_pct >= pos.take_profit_pct:
            exit_reason = 'TAKE_PROFIT'
        
        # 4. MODEL_EXIT
        elif (self.config.exit_confidence_drop > 0 and 
              pnl_pct >= pos.take_profit_pct * 0.3 and
              holding_bars >= 2):
            
            features = self.feature_calculator.calculate_features()
            if features is not None:
                # Только FEATURE_COLS (11 фич), без sma_14
                model_features = features[self.feature_calculator.FEATURE_COLS]
                X = pd.DataFrame([model_features.values], columns=self.feature_calculator.FEATURE_COLS)
                proba_long = self.model_long.predict_proba(X)[0, 1]
                proba_short = self.model_short.predict_proba(X)[0, 1]
                
                if pos.direction == 'LONG':
                    current_diff = proba_long - proba_short
                    # Выходим если confidence упала или развернулась
                    if (pos.entry_proba_diff - current_diff >= self.config.exit_confidence_drop or
                        current_diff < 0):
                        exit_reason = 'MODEL_EXIT'
                else:
                    current_diff = proba_short - proba_long
                    if (pos.entry_proba_diff - current_diff >= self.config.exit_confidence_drop or
                        current_diff < 0):
                        exit_reason = 'MODEL_EXIT'
        
        if exit_reason:
            return self._execute_exit(candle, timestamp, exit_reason, pnl_pct, holding_bars)
        
        return None
    
    def _execute_exit(self, candle: Dict, timestamp: datetime, exit_reason: str, 
                      pnl_pct: float, holding_bars: int) -> str:
        """Выполнить выход из позиции."""
        pos = self.position
        price = candle['close']
        
        # Slippage на выход
        if pos.direction == 'LONG':
            exec_price = price * (1 - self.config.slippage)
        else:
            exec_price = price * (1 + self.config.slippage)
        
        # Пересчитываем PnL с учётом slippage
        if pos.direction == 'LONG':
            final_pnl_pct = exec_price / pos.entry_price - 1
        else:
            final_pnl_pct = pos.entry_price / exec_price - 1
        
        # Комиссия на выход
        comm_out = exec_price * pos.size * self.config.commission
        
        # Возвращаем капитал
        if pos.direction == 'LONG':
            pnl_usd = (exec_price - pos.entry_price) * pos.size - comm_out
        else:
            pnl_usd = (pos.entry_price - exec_price) * pos.size - comm_out
        
        self.capital += pos.entry_price * pos.size + pnl_usd
        
        trade = Trade(
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exec_price,
            size=pos.size,
            pnl_pct=final_pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            confidence=pos.entry_proba_diff,
            session_id=pos.session_id,
            holding_bars=holding_bars,
        )
        
        self.trades.append(trade)
        self.total_trades += 1
        self.position = None
        
        result = 'WIN' if final_pnl_pct > 0 else 'LOSE'
        print(f"[{timestamp}] {exit_reason}: {pos.direction} @ {exec_price:.2f} "
              f"PnL={final_pnl_pct*100:.2f}% ({result})")
        
        if self.on_trade:
            self.on_trade(trade)
        
        return 'EXIT'
    
    def get_stats(self) -> Dict:
        """Получить статистику."""
        if not self.trades:
            return {'trades': 0, 'pnl': 0, 'win_rate': 0}
        
        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct <= 0]
        
        total_pnl = sum(t.pnl_pct for t in self.trades) * 100
        
        avg_win = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0
        
        # Profit Factor
        gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_curve = [10000.0]
        for t in self.trades:
            equity_curve.append(equity_curve[-1] * (1 + t.pnl_pct * self.config.position_pct))
        
        peak = 10000.0
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'final_capital': self.capital,
            'exit_reasons': exit_reasons,
            'sessions': len(self.session_detector.sessions),
        }


# ============================================================================
# ПРОВАЙДЕР ДАННЫХ (для бэктеста)
# ============================================================================

class HistoricalCandleProvider:
    """Провайдер исторических свечей для бэктеста (ТОЛЬКО сырые OHLCV)."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame с колонками Open, High, Low, Close, Volume и индексом datetime
        """
        self.df = df.copy()
        self.current_index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict:
        if self.current_index >= len(self.df):
            raise StopIteration
        
        row = self.df.iloc[self.current_index]
        candle = {
            'timestamp': self.df.index[self.current_index],
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume'],
        }
        # НЕ добавляем разметку сессий - всё в realtime!
        
        self.current_index += 1
        return candle
    
    def __len__(self):
        return len(self.df)


# ============================================================================
# ВАЛИДАЦИЯ REALTIME ДЕТЕКТОРА
# ============================================================================

def validate_session_detection(data_path: str = None, sample_pct: float = 0.2) -> Dict:
    """
    Валидация: сравнение realtime детектора с phase1 разметкой.
    
    Args:
        data_path: путь к данным
        sample_pct: доля данных для валидации (0.2 = 20%)
    
    Returns:
        Статистика совпадения
    """
    print("=" * 60)
    print("ВАЛИДАЦИЯ REALTIME ДЕТЕКТОРА СЕССИЙ")
    print("=" * 60)
    
    # Загружаем данные
    data_path = data_path or str(DATA_DIR / "btc_processed_v3.pkl")
    with open(data_path, 'rb') as f:
        pkg = pickle.load(f)
    
    # Используем train_data для валидации (там есть разметка)
    full_data = pkg['train_data']
    
    # Берём последние sample_pct данных (чтобы был прогрев)
    sample_size = int(len(full_data) * sample_pct)
    warmup_size = 90 * 24 * 60 + 1000  # 90 дней + запас
    
    # Данные для прогрева + данные для валидации
    start_idx = max(0, len(full_data) - sample_size - warmup_size)
    data = full_data.iloc[start_idx:]
    
    validation_start_idx = warmup_size if start_idx == 0 else warmup_size
    
    print(f"Всего данных: {len(full_data)} баров")
    print(f"Для валидации: {len(data) - validation_start_idx} баров ({sample_pct*100:.0f}%)")
    print(f"Период: {data.index[validation_start_idx]} - {data.index[-1]}")
    
    # Создаём realtime детектор
    session_config = SessionConfig()
    detector = RealtimeSessionDetector(session_config)
    
    # Прогоняем все данные
    realtime_in_session = []
    realtime_sessions_detected = []
    prev_close = None
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        close = row['Close']
        
        if prev_close is not None:
            detector.add_candle(timestamp, close, prev_close)
            new_session = detector.update(timestamp)
            if new_session and i >= validation_start_idx:
                realtime_sessions_detected.append(new_session)
        
        if i >= validation_start_idx:
            in_session, session_id, _ = detector.is_in_session(timestamp)
            realtime_in_session.append(1 if in_session else 0)
        
        prev_close = close
        
        if (i + 1) % 50000 == 0:
            print(f"  Обработано {i+1}/{len(data)} баров...")
    
    # Сравнение с phase1 (только validation часть)
    validation_data = data.iloc[validation_start_idx:]
    phase1_in_session = validation_data['in_session'].tolist() if 'in_session' in validation_data.columns else [0] * len(validation_data)
    phase1_sessions = validation_data[validation_data['in_session'] == 1]['session_id'].nunique() if 'in_session' in validation_data.columns else 0
    
    # Статистика
    realtime_total = sum(realtime_in_session)
    phase1_total = sum(phase1_in_session)
    
    # Совпадения (bar-level)
    matches = sum(1 for r, p in zip(realtime_in_session, phase1_in_session) if r == p)
    accuracy = matches / len(realtime_in_session) * 100 if realtime_in_session else 0
    
    # Сравнение времён старта сессий
    phase1_starts = set()
    if 'session_start' in validation_data.columns:
        phase1_starts = set(validation_data[validation_data['in_session'] == 1]['session_start'].dropna().unique())
    
    realtime_starts = set(s['session_start'] for s in realtime_sessions_detected)
    
    # Сколько сессий совпали по времени старта
    session_matches = len(phase1_starts & realtime_starts)
    
    # Детальное сравнение по сессиям
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    print(f"Phase1 сессий: {phase1_sessions}")
    print(f"Realtime сессий: {len(realtime_sessions_detected)}")
    print(f"Совпали по времени старта: {session_matches}")
    print()
    print(f"Phase1 баров в сессиях: {phase1_total}")
    print(f"Realtime баров в сессиях: {realtime_total}")
    print()
    print(f"Совпадение bar-level: {accuracy:.2f}%")
    
    # Показываем первые 5 расхождений
    if phase1_starts != realtime_starts:
        only_phase1 = phase1_starts - realtime_starts
        only_realtime = realtime_starts - phase1_starts
        if only_phase1:
            print(f"\nТолько в Phase1 ({len(only_phase1)}):")
            for s in sorted(only_phase1)[:3]:
                print(f"  {s}")
        if only_realtime:
            print(f"\nТолько в Realtime ({len(only_realtime)}):")
            for s in sorted(only_realtime)[:3]:
                print(f"  {s}")
    
    print("=" * 60)
    
    return {
        'phase1_sessions': phase1_sessions,
        'realtime_sessions': len(realtime_sessions_detected),
        'session_matches': session_matches,
        'phase1_bars': phase1_total,
        'realtime_bars': realtime_total,
        'accuracy': accuracy,
        'realtime_sessions_list': realtime_sessions_detected,
    }


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def run_backtest(
    data_path: str = None,
    model_path: str = None,
    test_start: str = "2025-09-22",
    test_end: str = "2025-12-15",
    initial_capital: float = 10000.0,
    verbose: bool = True,
) -> Dict:
    """
    Запустить бэктест.
    
    Args:
        data_path: путь к pkl с данными
        model_path: путь к сохранённым моделям (или None для обучения)
        test_start: начало тестового периода
        test_end: конец тестового периода
        initial_capital: начальный капитал
        verbose: выводить логи
    
    Returns:
        Статистика бэктеста
    """
    print("=" * 60)
    print("РЕАЛИСТИЧНЫЙ БЭКТЕСТЕР v1.0")
    print("=" * 60)
    
    # 1. Загружаем данные
    data_path = data_path or str(DATA_DIR / "btc_processed_v3.pkl")
    print(f"\n1. Загружаем данные из {data_path}...")
    
    with open(data_path, 'rb') as f:
        pkg = pickle.load(f)
    
    train_data = pkg['train_data']
    val_data = pkg.get('val_data', pd.DataFrame())
    test_data = pkg['test_data']
    
    # Фильтруем тестовый период
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    test_data = test_data[(test_data.index >= test_start_dt) & (test_data.index <= test_end_dt)]
    
    print(f"   Train: {len(train_data)} баров")
    print(f"   Val: {len(val_data)} баров")
    print(f"   Test: {len(test_data)} баров ({test_start} - {test_end})")
    
    # 2. ЗАГРУЖАЕМ ОБУЧЕННЫЕ МОДЕЛИ ИЗ phase2_model_v3.py
    trading_config = TradingConfig()
    horizon = trading_config.horizon
    
    model_path = model_path or str(MODELS_DIR / f"phase2_h{horizon}_final.pkl")
    print(f"\n2. Загружаем модели из {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Модели не найдены: {model_path}\n"
            f"Сначала запустите phase2_model_v3.py для обучения и сохранения моделей!"
        )
    
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    model_long = model_pkg['model_long']
    model_short = model_pkg['model_short']
    best_params = model_pkg['best_params']
    feature_cols = model_pkg['feature_cols']
    
    # Обновляем trading_config параметрами из обученной модели
    trading_config.thr_long = best_params.get('thr_long', trading_config.thr_long)
    trading_config.thr_short = best_params.get('thr_short', trading_config.thr_short)
    trading_config.min_confidence = best_params.get('min_confidence', trading_config.min_confidence)
    trading_config.stop_mult = best_params.get('stop_mult', trading_config.stop_mult)
    trading_config.take_mult = best_params.get('take_mult', trading_config.take_mult)
    trading_config.trailing_mult = best_params.get('trailing_mult', trading_config.trailing_mult)
    trading_config.breakeven_mult = best_params.get('breakeven_mult', trading_config.breakeven_mult)
    trading_config.exit_confidence_drop = best_params.get('exit_confidence_drop', trading_config.exit_confidence_drop)
    
    print(f"   Horizon: {model_pkg.get('horizon', horizon)}")
    print(f"   Features: {len(feature_cols)} - {feature_cols}")
    print(f"   Params: thr_long={trading_config.thr_long:.4f}, thr_short={trading_config.thr_short:.4f}")
    print(f"           min_conf={trading_config.min_confidence:.4f}")
    print(f"           stop={trading_config.stop_mult:.4f}, take={trading_config.take_mult:.4f}")
    print("   Модели загружены!")
    
    # Обновляем FEATURE_COLS в калькуляторе
    RealtimeFeatureCalculator.FEATURE_COLS = feature_cols
    
    # 3. Создаём торговый движок
    print("\n3. Инициализируем торговый движок...")
    
    engine = TradingEngine(
        model_long=model_long,
        model_short=model_short,
        trading_config=trading_config,
        session_config=SessionConfig(),
    )
    engine.reset(initial_capital)
    
    # 4. Запускаем бэктест
    print(f"\n4. Запускаем бэктест ({len(test_data)} баров)...")
    print("-" * 60)
    
    # КРИТИЧНО: Прогрев на 90+ днях для детектора сессий (lookback_days=90)
    session_config = SessionConfig()
    warmup_days = session_config.lookback_days + 7  # 90 + 7 = 97 дней
    warmup_start = test_start_dt - timedelta(days=warmup_days)
    
    # Ищем данные для прогрева в train_data, val_data, test_data или full_data
    warmup_data = None
    
    if 'full_data' in pkg:
        # Лучший вариант - есть все данные
        full_data = pkg['full_data']
        warmup_data = full_data[(full_data.index >= warmup_start) & (full_data.index < test_start_dt)]
    else:
        # Собираем из train + val + test (warmup может быть в начале test_data!)
        all_data = pd.concat([train_data, val_data, test_data]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep='first')]
        warmup_data = all_data[(all_data.index >= warmup_start) & (all_data.index < test_start_dt)]
    
    if warmup_data is not None and len(warmup_data) > 0:
        print(f"   Прогрев детектора сессий на {len(warmup_data)} барах ({warmup_days} дней)...")
        warmup_provider = HistoricalCandleProvider(warmup_data)
        
        prev_close = None
        for candle in warmup_provider:
            # Добавляем в feature calculator
            engine.feature_calculator.add_candle(candle)
            
            # Добавляем в session detector
            if prev_close is not None:
                engine.session_detector.add_candle(candle['timestamp'], candle['close'], prev_close)
            prev_close = candle['close']
        
        print(f"   Прогрев завершён. Волатильность: {len(engine.session_detector._vol_cache)} точек")
    else:
        print(f"   ВНИМАНИЕ: Нет данных для прогрева! Детектор сессий может работать некорректно.")
    
    # Основной бэктест - используем СЫРЫЕ данные (только OHLCV)
    test_provider = HistoricalCandleProvider(test_data)
    
    total_bars = len(test_data)
    last_pct = 0
    realtime_session_bars = 0  # Счётчик баров в сессиях (realtime детектор)
    
    for i, candle in enumerate(test_provider):
        engine.process_candle(candle)
        
        # Подсчитываем бары в сессии (realtime)
        in_session, _, _ = engine.session_detector.is_in_session(candle['timestamp'])
        if in_session:
            realtime_session_bars += 1
        
        # Прогресс каждые 5%
        pct = int((i + 1) / total_bars * 100)
        if pct >= last_pct + 5:
            last_pct = pct
            trades_so_far = len(engine.trades)
            capital = engine.capital
            sessions_so_far = len(engine.session_detector.sessions)
            print(f"   [{pct:3d}%] Бар {i+1}/{total_bars} | Сессий: {sessions_so_far} | Сделок: {trades_so_far} | Капитал: ${capital:.2f}")
    
    print("-" * 60)
    
    # 5. Статистика
    stats = engine.get_stats()
    
    # Добавляем статистику realtime сессий
    stats['realtime_sessions'] = len(engine.session_detector.sessions)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА (REALTIME)")
    print("=" * 60)
    print(f"Период: {test_start} - {test_end}")
    print(f"Начальный капитал: ${initial_capital:,.2f}")
    print(f"Конечный капитал: ${stats['final_capital']:,.2f}")
    print()
    print(f"Сделок: {stats['trades']}")
    print(f"Win Rate: {stats['win_rate']:.1f}% ({stats['wins']} WIN / {stats['losses']} LOSE)")
    print(f"Total PnL: {stats['total_pnl']:+.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"Avg Win: {stats['avg_win']:+.2f}%")
    print(f"Avg Loss: {stats['avg_loss']:.2f}%")
    print()
    print("Причины выхода:")
    for reason, count in stats['exit_reasons'].items():
        print(f"  {reason}: {count}")
    print()
    print(f"Realtime сессий обнаружено: {stats['realtime_sessions']}")
    print(f"Баров в сессиях (realtime): {realtime_session_bars} / {total_bars}")
    
    # Сравнение с разметкой из данных (если есть)
    if 'in_session' in test_data.columns:
        data_session_bars = int((test_data['in_session'] == 1).sum())
        data_sessions = test_data[test_data['in_session'] == 1]['session_id'].nunique()
        print(f"Сессий в данных (phase1): {data_sessions}")
        print(f"Баров в сессиях (phase1): {data_session_bars}")
    print("=" * 60)
    
    return {
        'stats': stats,
        'trades': [
            {
                'direction': t.direction,
                'entry_time': str(t.entry_time),
                'exit_time': str(t.exit_time),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct * 100,
                'exit_reason': t.exit_reason,
            }
            for t in engine.trades
        ],
        'engine': engine,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Режим валидации - сравнение realtime детектора с phase1
        validate_session_detection()
    else:
        # Обычный бэктест
        result = run_backtest(
            test_start="2025-09-22",
            test_end="2025-12-15",
            initial_capital=10000.0,
        )
        
        # Сохраняем результаты
        output = {
            'stats': result['stats'],
            'trades': result['trades'],
            'timestamp': datetime.now().isoformat(),
        }
        
        output_path = RESULTS_DIR / f"realtime_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nРезультаты сохранены: {output_path}")
