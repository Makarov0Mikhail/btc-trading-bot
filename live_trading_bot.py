"""
Live Trading Bot для Bybit Testnet (Demo).

Использует логику из realtime_backtester.py, но:
1. Получает свечи через WebSocket в реальном времени
2. Отправляет реальные ордера на биржу
3. Управляет стоп-лоссами и тейк-профитами на бирже

ВАЖНО: Это демо-бот для testnet.bybit.com!
"""

import os
import sys
import json
import time
import pickle
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import deque
import threading
import signal

# Подавляем warnings
import warnings
warnings.filterwarnings('ignore')

# Bybit API
from pybit.unified_trading import HTTP, WebSocket

# Пути
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

def setup_logging():
    """Настройка логирования в файл и консоль."""
    log_file = LOGS_DIR / f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


# ============================================================================
# RETRY DECORATOR для API запросов
# ============================================================================

def retry_on_error(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Декоратор для повторных попыток при ошибках API."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"⚠️ {func.__name__} ошибка: {e}. Повтор через {wait_time:.1f}с ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ {func.__name__} не удалось после {max_retries} попыток: {e}")
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class BotConfig:
    """Конфигурация бота."""
    # Bybit API - берём из env или используем дефолт
    # Для Demo Trading на bybit.com: testnet=False, demo=True
    api_key: str = field(default_factory=lambda: os.environ.get('BYBIT_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.environ.get('BYBIT_API_SECRET', ''))
    testnet: bool = False  # False для bybit.com (включая Demo Trading)
    demo: bool = True  # True для Demo Trading режима
    
    # Торговые параметры
    symbol: str = "BTCUSDT"
    category: str = "linear"  # Perpetual futures
    leverage: int = 3  # Плечо 3x
    
    # Размер позиции (100% депозита = $10k)
    position_size_usd: float = 10000.0  # Размер позиции в USD
    
    # Параметры модели (ТОЧНЫЕ значения из validated backtest +4.26% profit)
    horizon: int = 15
    thr_long: float = 0.5943962233115141
    thr_short: float = 0.5452861840022991
    min_confidence: float = 0.05010305677155944
    stop_mult: float = 3.909726253629285
    take_mult: float = 0.6733841570835483
    exit_confidence_drop: float = 0.15  # MODEL_EXIT: порог падения confidence для выхода
    
    # Сессии
    max_trades_per_session: int = 2
    
    # Session detector params
    session_lookback_days: int = 90
    session_percentile: float = 95
    session_prebuffer_min: int = 45
    session_len_min: int = 45
    
    # Безопасность
    max_daily_loss_pct: float = 5.0  # Макс. дневной убыток
    max_position_size_btc: float = 0.5  # Макс. размер позиции в BTC (для $10k @ ~$94k = ~0.1 BTC * 3x leverage)
    
    # ============== ТЕСТОВЫЙ РЕЖИМ ==============
    # Установить True для немедленного теста без ожидания сессии
    # После проверки удалить или установить False
    test_mode: bool = False  # Боевой режим


@dataclass
class Position:
    """Текущая позиция."""
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size: float  # в BTC
    size_usd: float
    stop_loss_price: float
    take_profit_price: float
    order_id: str
    stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    session_id: int = 0
    entry_bar: int = 0
    entry_proba_diff: float = 0.0


# ============================================================================
# REALTIME FEATURE CALCULATOR (из realtime_backtester.py)
# ============================================================================

class RealtimeFeatureCalculator:
    """Расчёт фичей в реальном времени - идентично backtester."""
    
    FEATURE_COLS = [
        'log_return', 'log_return_lag_2', 'hour_sin', 'hour_cos', 'is_weekend',
        'prev_candle_body', 'volume_change', 'volume_change_lag_1',
        'atr_pct', 'rsi_14', 'macd_hist'
    ]
    
    def __init__(self, warmup_bars: int = 100):
        self.warmup_bars = warmup_bars
        self.candles: List[Dict] = []
        self.max_candles = 500
        
    def add_candle(self, candle: Dict):
        """Добавить новую свечу."""
        self.candles.append(candle)
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]
    
    def calculate_features(self) -> Optional[pd.Series]:
        """Рассчитать фичи для текущего момента."""
        if len(self.candles) < self.warmup_bars:
            return None
        
        df = pd.DataFrame(self.candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Базовые фичи
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_lag_2'] = df['log_return'].shift(2)
        
        # Временные фичи
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Свечные паттерны
        df['candle_body'] = (df['close'] - df['open']) / df['open']
        df['prev_candle_body'] = df['candle_body'].shift(1)
        
        # Объём
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_change'] = df['volume'] / df['volume_ma'] - 1
        df['volume_change_lag_1'] = df['volume_change'].shift(1)
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = (df['rsi_14'] - 50) / 50  # Нормализация [-1, 1]
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df['macd_hist'] = (macd - signal) / df['close']
        
        # SMA для тренда (не входит в модель, но нужен для детектора)
        df['sma_14'] = df['close'].rolling(14).mean()
        
        # Берём последнюю строку
        last = df.iloc[-1]
        
        # Проверяем NaN
        features = last[self.FEATURE_COLS]
        if features.isna().any():
            return None
        
        # Добавляем atr_pct и sma_14 для расчёта стопов
        result = features.copy()
        result['atr_pct'] = last['atr_pct']
        result['sma_14'] = last['sma_14']
        
        return result


# ============================================================================
# REALTIME SESSION DETECTOR (ТОЧНАЯ КОПИЯ из realtime_backtester.py)
# ============================================================================

class RealtimeSessionDetector:
    """
    Детектор сессий в реальном времени.
    ИДЕНТИЧЕН логике из phase1_sessions_v3.py:
    1. short_vol = log_return.rolling(20).std() на ПРЕДЫДУЩЕМ баре
    2. threshold = percentile за lookback_days
    3. Старт сессии когда short_vol > threshold И предыдущий short_vol <= threshold
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.lookback_bars = config.session_lookback_days * 24 * 60
        self.volatility_window = 20  # rolling std window
        self.log_returns: List[float] = []
        self._vol_cache: List[float] = []  # rolling std values
        self._session_id = 0
        self.current_session: Optional[Dict] = None
        self.last_session_end: Optional[datetime] = None
        
    def _calculate_rolling_std(self, values: List[float], window: int) -> float:
        """Рассчитать rolling std для последних window значений."""
        if len(values) < window:
            return np.nan
        arr = np.array(values[-window:])
        return float(np.std(arr, ddof=1))
    
    def add_candle(self, timestamp: datetime, close: float, prev_close: float):
        """Добавить новую свечу для расчёта волатильности."""
        if prev_close > 0:
            lr = np.log(close / prev_close)
        else:
            lr = 0.0
        
        self.log_returns.append(lr)
        
        # Рассчитываем rolling std
        if len(self.log_returns) >= self.volatility_window:
            current_vol = self._calculate_rolling_std(self.log_returns, self.volatility_window)
        else:
            current_vol = np.nan
        self._vol_cache.append(current_vol)
        
        # Обрезаем историю
        if len(self.log_returns) > self.lookback_bars + self.volatility_window + 10:
            self.log_returns = self.log_returns[-(self.lookback_bars + self.volatility_window):]
            self._vol_cache = self._vol_cache[-(self.lookback_bars + self.volatility_window):]
    
    def update(self, timestamp: datetime) -> Optional[Dict]:
        """Проверить условия старта сессии."""
        # Проверяем текущую сессию
        if self.current_session:
            if timestamp >= self.current_session['session_end']:
                self.current_session = None
        
        # Если уже в сессии - не создаём новую
        if self.current_session:
            return None
        
        # Проверяем cooldown (30 мин после сессии)
        if self.last_session_end:
            cooldown_end = self.last_session_end + timedelta(minutes=30)
            if timestamp < cooldown_end:
                return None
        
        # Минимум данных
        if len(self._vol_cache) < self.lookback_bars + 2:
            return None
        
        # short_vol на ПРЕДЫДУЩЕМ баре (shift(1))
        short_vol = self._vol_cache[-2] if len(self._vol_cache) >= 2 else np.nan
        short_vol_prev = self._vol_cache[-3] if len(self._vol_cache) >= 3 else np.nan
        
        if np.isnan(short_vol):
            return None
        
        # Percentile за lookback_days (исключаем текущий)
        lookback_data = self._vol_cache[-(self.lookback_bars + 1):-1]
        valid_data = [v for v in lookback_data if not np.isnan(v)]
        
        if len(valid_data) < 1000:
            return None
        
        threshold = np.percentile(valid_data, self.config.session_percentile)
        
        # Проверяем условие старта: high_vol AND NOT high_vol_prev
        high_vol_current = short_vol > threshold
        
        # Порог для предыдущего бара
        lookback_data_prev = self._vol_cache[-(self.lookback_bars + 2):-2] if len(self._vol_cache) >= self.lookback_bars + 2 else []
        valid_data_prev = [v for v in lookback_data_prev if not np.isnan(v)]
        
        if len(valid_data_prev) >= 1000:
            threshold_prev = np.percentile(valid_data_prev, self.config.session_percentile)
            high_vol_prev = short_vol_prev > threshold_prev if not np.isnan(short_vol_prev) else False
        else:
            high_vol_prev = False
        
        # Старт сессии: текущий high И предыдущий NOT high
        if high_vol_current and not high_vol_prev:
            self._session_id += 1
            session = {
                'session_id': self._session_id,
                'prebuffer_start': timestamp - timedelta(minutes=self.config.session_prebuffer_min),
                'session_start': timestamp,
                'session_end': timestamp + timedelta(minutes=self.config.session_len_min),
                'volatility': float(short_vol),
                'threshold': float(threshold),
            }
            self.current_session = session
            self.last_session_end = session['session_end']
            return session
        
        return None
    
    def is_in_session(self, timestamp: datetime) -> Tuple[bool, int, Optional[datetime]]:
        """Проверить, находимся ли в активной сессии."""
        if self.current_session:
            if self.current_session['session_start'] <= timestamp <= self.current_session['session_end']:
                return True, self.current_session['session_id'], self.current_session['session_end']
        return False, -1, None


# ============================================================================
# BYBIT TRADING ENGINE
# ============================================================================

class BybitTradingEngine:
    """Торговый движок для Bybit."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.position: Optional[Position] = None
        self.daily_pnl = 0.0
        self.session_trade_counts: Dict[int, int] = {}
        self.bar_index = 0
        self.trades_log: List[Dict] = []
        # Подключение к Bybit
        logger.info(f"Подключение к Bybit {'DEMO' if config.demo else 'LIVE'}...")
        
        self.http = HTTP(
            testnet=config.testnet,
            api_key=config.api_key,
            api_secret=config.api_secret,
            demo=config.demo,  # Важно для Demo Trading!
            recv_window=20000,  # Увеличенное окно для синхронизации времени
        )
        
        # Загружаем модели
        self._load_models()
        
        # Инициализируем калькулятор фичей и детектор сессий
        self.feature_calculator = RealtimeFeatureCalculator(warmup_bars=100)
        self.session_detector = RealtimeSessionDetector(config)
        
        # Проверяем подключение
        self._check_connection()
        
        # Устанавливаем плечо
        self._set_leverage()
    
    def _load_models(self):
        """Загрузить ML модели."""
        model_path = MODELS_DIR / f"phase2_h{self.config.horizon}_final.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_long = model_data['model_long']
        self.model_short = model_data['model_short']
        logger.info(f"Модели загружены: {model_path.name}")
    
    def _check_connection(self):
        """Проверить подключение к API."""
        try:
            result = self.http.get_wallet_balance(accountType="UNIFIED")
            if result['retCode'] == 0:
                coins = result['result']['list'][0]['coin']
                for coin in coins:
                    if coin['coin'] == 'USDT':
                        balance = float(coin['walletBalance'])
                        logger.info(f"Баланс USDT: {balance:.2f}")
                        return True
            else:
                logger.error(f"Ошибка API: {result['retMsg']}")
                return False
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False
    
    def _set_leverage(self):
        """Установить плечо."""
        try:
            self.http.set_leverage(
                category=self.config.category,
                symbol=self.config.symbol,
                buyLeverage=str(self.config.leverage),
                sellLeverage=str(self.config.leverage),
            )
            logger.info(f"Плечо установлено: {self.config.leverage}x")
        except Exception as e:
            logger.warning(f"Не удалось установить плечо (возможно уже установлено): {e}")
    
    def restore_position_on_startup(self):
        """Проверить и восстановить позицию при старте (после рестарта бота)."""
        try:
            exchange_pos = self.get_position()
            if exchange_pos and float(exchange_pos['size']) > 0:
                direction = "LONG" if exchange_pos['side'] == "Buy" else "SHORT"
                entry_price = float(exchange_pos['avgPrice'])
                size = float(exchange_pos['size'])
                
                # Получаем SL/TP
                stop_loss = float(exchange_pos.get('stopLoss', 0)) or entry_price * 0.97
                take_profit = float(exchange_pos.get('takeProfit', 0)) or entry_price * 1.03
                
                self.position = Position(
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=datetime.now(timezone.utc),
                    size=size,
                    size_usd=size * entry_price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    order_id="restored",
                    session_id=0,
                    entry_bar=self.bar_index,
                    entry_proba_diff=0.5,  # Unknown
                )
                
                logger.warning(f"⚠️ ВОССТАНОВЛЕНА позиция: {direction} {size} BTC @ {entry_price:.2f}")
                logger.warning(f"   SL={stop_loss:.2f}, TP={take_profit:.2f}")
                return True
            else:
                logger.info("✅ Нет открытых позиций на бирже")
                return False
        except Exception as e:
            logger.error(f"Ошибка при восстановлении позиции: {e}")
            return False
    
    def get_current_price(self) -> float:
        """Получить текущую цену."""
        result = self.http.get_tickers(
            category=self.config.category,
            symbol=self.config.symbol
        )
        return float(result['result']['list'][0]['lastPrice'])
    
    def get_position(self) -> Optional[Dict]:
        """Получить текущую позицию с биржи."""
        result = self.http.get_positions(
            category=self.config.category,
            symbol=self.config.symbol
        )
        if result['retCode'] == 0:
            positions = result['result']['list']
            for pos in positions:
                if float(pos['size']) > 0:
                    return pos
        return None
    
    def process_candle(self, candle: Dict) -> Optional[str]:
        """Обработать новую свечу."""
        timestamp = candle['timestamp']
        self.bar_index += 1
        
        # 1. Добавляем в калькулятор фичей
        self.feature_calculator.add_candle(candle)
        
        # 2. Обновляем детектор сессий
        if len(self.feature_calculator.candles) >= 2:
            prev_close = self.feature_calculator.candles[-2]['close']
        else:
            prev_close = candle['close']
        
        self.session_detector.add_candle(timestamp, candle['close'], prev_close)
        
        new_session = self.session_detector.update(timestamp)
        
        if new_session:
            logger.info(f"🔔 Новая сессия #{new_session['session_id']} "
                       f"(vol={new_session['volatility']:.6f}, thr={new_session['threshold']:.6f})")
        
        in_session, session_id, session_end = self.session_detector.is_in_session(timestamp)
        
        # 3. Если не в сессии и есть позиция - закрываем
        if not in_session:
            if self.position:
                logger.info("📤 Выход из сессии - закрываем позицию")
                return self._close_position("SESSION_END")
            return None
        
        # 4. Если есть позиция - проверяем выход
        if self.position:
            action = self._check_exit(candle, timestamp)
            if action:
                return action
        
        # 5. Если нет позиции - проверяем вход
        if self.position is None and in_session:
            return self._check_entry(candle, timestamp, session_id)
        
        return None
    
    def _check_entry(self, candle: Dict, timestamp: datetime, session_id: int) -> Optional[str]:
        """Проверить условия входа."""
        # Лимит сделок на сессию
        if self.session_trade_counts.get(session_id, 0) >= self.config.max_trades_per_session:
            return None
        
        # Рассчитываем фичи
        features = self.feature_calculator.calculate_features()
        if features is None:
            return None
        
        # Предсказание модели
        model_features = features[self.feature_calculator.FEATURE_COLS]
        X = pd.DataFrame([model_features.values], columns=self.feature_calculator.FEATURE_COLS)
        
        proba_long = self.model_long.predict_proba(X)[0, 1]
        proba_short = self.model_short.predict_proba(X)[0, 1]
        proba_diff = proba_long - proba_short
        
        # Определяем сигнал
        signal = 0
        if (proba_long >= self.config.thr_long and 
            proba_short < self.config.thr_short and 
            proba_diff >= self.config.min_confidence):
            signal = 1  # LONG
        elif (proba_short >= self.config.thr_short and 
              proba_long < self.config.thr_long and 
              proba_diff <= -self.config.min_confidence):
            signal = -1  # SHORT
        
        if signal == 0:
            return None
        
        # Рассчитываем стопы
        atr_pct = features.get('atr_pct', 0.01)
        if atr_pct <= 0:
            atr_pct = 0.01
        
        default_stop = 0.005
        base_stop = max(default_stop, atr_pct * 1.5)
        stop_pct = min(base_stop * self.config.stop_mult, 0.03)
        take_pct = stop_pct * self.config.take_mult
        
        price = candle['close']
        direction = 'LONG' if signal == 1 else 'SHORT'
        
        # Рассчитываем уровни SL/TP
        if direction == 'LONG':
            stop_price = price * (1 - stop_pct)
            take_price = price * (1 + take_pct)
        else:
            stop_price = price * (1 + stop_pct)
            take_price = price * (1 - take_pct)
        
        # ОТКРЫВАЕМ ПОЗИЦИЮ НА БИРЖЕ
        return self._open_position(
            direction=direction,
            price=price,
            stop_price=stop_price,
            take_price=take_price,
            timestamp=timestamp,
            session_id=session_id,
            proba_diff=proba_diff
        )
    
    def _open_position(self, direction: str, price: float, stop_price: float, 
                       take_price: float, timestamp: datetime, session_id: int,
                       proba_diff: float) -> Optional[str]:
        """Открыть позицию на бирже."""
        try:
            # Размер в BTC с учётом плеча (Bybit BTCUSDT: min=0.001, step=0.001)
            # $10,000 депозит × 3x плечо = $30,000 позиция
            position_value = self.config.position_size_usd * self.config.leverage
            size_btc = position_value / price
            size_btc = round(size_btc, 3)  # Округляем до 3 знаков (шаг Bybit)
            
            # Ограничение
            if size_btc > self.config.max_position_size_btc:
                size_btc = self.config.max_position_size_btc
            
            side = "Buy" if direction == "LONG" else "Sell"
            
            logger.info(f"🚀 Открываем {direction}: size={size_btc} BTC @ ~{price:.2f} (${position_value:.0f} = ${self.config.position_size_usd:.0f} × {self.config.leverage}x)")
            logger.info(f"   SL={stop_price:.2f}, TP={take_price:.2f}")
            
            # Маркет ордер
            result = self.http.place_order(
                category=self.config.category,
                symbol=self.config.symbol,
                side=side,
                orderType="Market",
                qty=str(size_btc),
                stopLoss=str(round(stop_price, 2)),
                takeProfit=str(round(take_price, 2)),
                tpslMode="Full",
                tpOrderType="Market",
                slOrderType="Market",
            )
            
            if result['retCode'] == 0:
                order_id = result['result']['orderId']
                
                # Получаем реальную цену исполнения
                time.sleep(0.5)  # Ждём исполнения
                exec_price = self.get_current_price()
                
                self.position = Position(
                    direction=direction,
                    entry_price=exec_price,
                    entry_time=timestamp,
                    size=size_btc,
                    size_usd=size_btc * exec_price,
                    stop_loss_price=stop_price,
                    take_profit_price=take_price,
                    order_id=order_id,
                    session_id=session_id,
                    entry_bar=self.bar_index,
                    entry_proba_diff=abs(proba_diff),
                )
                
                self.session_trade_counts[session_id] = self.session_trade_counts.get(session_id, 0) + 1
                
                logger.info(f"✅ {direction} открыт: {size_btc} BTC @ {exec_price:.2f} (order_id={order_id})")
                
                return f'ENTRY_{direction}'
            else:
                logger.error(f"❌ Ошибка открытия: {result['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Исключение при открытии: {e}")
            return None
    
    def _check_exit(self, candle: Dict, timestamp: datetime) -> Optional[str]:
        """Проверить условия выхода (помимо SL/TP на бирже)."""
        if not self.position:
            return None
        
        # Проверяем по горизонту
        holding_bars = self.bar_index - self.position.entry_bar
        if holding_bars >= self.config.horizon:
            logger.info(f"⏰ Horizon exit: {holding_bars} баров")
            return self._close_position("HORIZON_EXIT")
        
        # Проверяем актуальность позиции на бирже
        exchange_pos = self.get_position()
        if exchange_pos is None:
            # Позиция закрылась (SL/TP сработал)
            logger.info("📊 Позиция закрыта на бирже (SL/TP)")
            self._log_trade("SL_OR_TP", candle['close'])
            self.position = None
            return "SL_OR_TP"
        
        # ========== MODEL_EXIT: модель решает выходить ==========
        # Проверяем только если:
        # 1. Держим >= 2 баров
        # 2. Есть минимальная прибыль (>= 30% от тейка)
        current_price = candle['close']
        if self.position.direction == "LONG":
            pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price
        
        # Рассчитываем текущий take_pct (для проверки 30% порога)
        take_pct = abs(self.position.take_profit_price - self.position.entry_price) / self.position.entry_price
        
        if holding_bars >= 2 and pnl_pct >= take_pct * 0.3 and self.config.exit_confidence_drop > 0:
            # Получаем текущие вероятности модели
            features = self.feature_calculator.calculate_features()
            if features is not None:
                model_features = features[self.feature_calculator.FEATURE_COLS]
                X = pd.DataFrame([model_features.values], columns=self.feature_calculator.FEATURE_COLS)
                
                current_proba_long = self.model_long.predict_proba(X)[0, 1]
                current_proba_short = self.model_short.predict_proba(X)[0, 1]
                
                entry_diff = self.position.entry_proba_diff  # Сохранённая разница при входе
                
                if self.position.direction == "LONG":
                    current_diff = current_proba_long - current_proba_short
                    # Выходим если: модель теперь за SHORT или confidence сильно упал
                    if current_diff < 0 or (entry_diff - current_diff) >= self.config.exit_confidence_drop:
                        logger.info(f"🤖 MODEL_EXIT: confidence упал {entry_diff:.3f} -> {current_diff:.3f} (PnL: {pnl_pct*100:.2f}%)")
                        return self._close_position("MODEL_EXIT")
                else:  # SHORT
                    current_diff = current_proba_short - current_proba_long
                    if current_diff < 0 or (entry_diff - current_diff) >= self.config.exit_confidence_drop:
                        logger.info(f"🤖 MODEL_EXIT: confidence упал {entry_diff:.3f} -> {current_diff:.3f} (PnL: {pnl_pct*100:.2f}%)")
                        return self._close_position("MODEL_EXIT")
        # ========== КОНЕЦ MODEL_EXIT ==========
        
        return None
    
    def _close_position(self, reason: str) -> Optional[str]:
        """Закрыть позицию на бирже."""
        if not self.position:
            return None
        
        try:
            side = "Sell" if self.position.direction == "LONG" else "Buy"
            
            logger.info(f"📤 Закрываем {self.position.direction}: {self.position.size} BTC ({reason})")
            
            result = self.http.place_order(
                category=self.config.category,
                symbol=self.config.symbol,
                side=side,
                orderType="Market",
                qty=str(self.position.size),
                reduceOnly=True,
            )
            
            if result['retCode'] == 0:
                time.sleep(0.5)
                exit_price = self.get_current_price()
                
                self._log_trade(reason, exit_price)
                
                logger.info(f"✅ Позиция закрыта @ {exit_price:.2f}")
                self.position = None
                return reason
            else:
                logger.error(f"❌ Ошибка закрытия: {result['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Исключение при закрытии: {e}")
            return None
    
    def _log_trade(self, exit_reason: str, exit_price: float):
        """Записать сделку в лог."""
        if not self.position:
            return
        
        if self.position.direction == 'LONG':
            pnl_pct = exit_price / self.position.entry_price - 1
        else:
            pnl_pct = self.position.entry_price / exit_price - 1
        
        pnl_usd = pnl_pct * self.position.size_usd
        self.daily_pnl += pnl_usd
        
        trade = {
            'direction': self.position.direction,
            'entry_time': self.position.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'size_btc': self.position.size,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
        }
        
        self.trades_log.append(trade)
        
        result = "🟢 WIN" if pnl_pct > 0 else "🔴 LOSE"
        logger.info(f"{result}: {pnl_pct*100:+.2f}% ({pnl_usd:+.2f} USD) | Daily PnL: {self.daily_pnl:+.2f} USD")
        
        # Сохраняем лог сделок
        trades_file = LOGS_DIR / "trades_log.json"
        with open(trades_file, 'w') as f:
            json.dump(self.trades_log, f, indent=2)


# ============================================================================
# WEBSOCKET HANDLER WITH AUTO-RECONNECT
# ============================================================================

class BybitWebSocketHandler:
    """Обработчик WebSocket для получения свечей с автопереподключением."""
    
    def __init__(self, engine: BybitTradingEngine, config: BotConfig):
        self.engine = engine
        self.config = config
        self.ws = None
        self.running = False
        self.last_candle_time: Optional[datetime] = None
        self.current_candle: Optional[Dict] = None
        self.last_message_time: float = time.time()
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # секунд
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Запустить WebSocket."""
        self._connect()
        self.running = True
        
        # Запускаем мониторинг соединения
        self._monitor_thread = threading.Thread(target=self._connection_monitor, daemon=True)
        self._monitor_thread.start()
    
    def _connect(self):
        """Подключиться к WebSocket."""
        logger.info("🔌 Запуск WebSocket...")
        
        try:
            self.ws = WebSocket(
                testnet=self.config.testnet,
                channel_type="linear",
            )
            
            # Подписываемся на kline 1m
            self.ws.kline_stream(
                interval=1,
                symbol=self.config.symbol,
                callback=self._on_kline,
            )
            
            self.last_message_time = time.time()
            self.reconnect_count = 0
            logger.info(f"✅ Подписка на {self.config.symbol} kline 1m")
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения WebSocket: {e}")
            raise
    
    def _connection_monitor(self):
        """Мониторинг соединения и автопереподключение."""
        heartbeat_timeout = 120  # секунд без сообщений = проблема
        
        while self.running:
            time.sleep(10)  # Проверяем каждые 10 секунд
            
            if not self.running:
                break
            
            # Проверяем таймаут
            elapsed = time.time() - self.last_message_time
            if elapsed > heartbeat_timeout:
                logger.warning(f"⚠️ WebSocket тишина {elapsed:.0f}с, переподключение...")
                self._reconnect()
    
    def _reconnect(self):
        """Переподключение WebSocket."""
        self.reconnect_count += 1
        
        if self.reconnect_count > self.max_reconnect_attempts:
            logger.error(f"❌ Превышено максимальное число переподключений ({self.max_reconnect_attempts})")
            self.running = False
            return
        
        logger.info(f"🔄 Попытка переподключения #{self.reconnect_count}...")
        
        # Закрываем старое соединение
        try:
            if self.ws:
                self.ws.exit()
        except:
            pass
        
        # Ждём перед переподключением (экспоненциальный backoff)
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_count - 1)), 60)
        logger.info(f"⏳ Ожидание {delay}с перед переподключением...")
        time.sleep(delay)
        
        # Переподключаемся
        try:
            self._connect()
            logger.info("✅ WebSocket переподключён успешно!")
        except Exception as e:
            logger.error(f"❌ Ошибка переподключения: {e}")
            # Попробуем ещё раз через мониторинг
    
    def _on_kline(self, message: Dict):
        """Обработчик новой свечи."""
        try:
            # Обновляем время последнего сообщения (для мониторинга соединения)
            self.last_message_time = time.time()
            
            if 'data' not in message:
                return
            
            for kline in message['data']:
                # Конвертируем в наш формат
                confirm = kline.get('confirm', False)
                
                timestamp = datetime.fromtimestamp(
                    int(kline['start']) / 1000, 
                    tz=timezone.utc
                )
                
                candle = {
                    'timestamp': timestamp,
                    'open': float(kline['open']),
                    'high': float(kline['high']),
                    'low': float(kline['low']),
                    'close': float(kline['close']),
                    'volume': float(kline['volume']),
                }
                
                # Обновляем текущую свечу
                self.current_candle = candle
                
                # Если свеча закрылась - обрабатываем
                if confirm:
                    if self.last_candle_time != timestamp:
                        self.last_candle_time = timestamp
                        logger.debug(f"📊 Новая свеча: {timestamp} | O={candle['open']:.2f} H={candle['high']:.2f} L={candle['low']:.2f} C={candle['close']:.2f}")
                        
                        # Обрабатываем свечу
                        action = self.engine.process_candle(candle)
                        if action:
                            logger.info(f"🎯 Action: {action}")
                            
        except Exception as e:
            logger.error(f"Ошибка обработки kline: {e}")
    
    def stop(self):
        """Остановить WebSocket."""
        self.running = False
        if self.ws:
            self.ws.exit()
        logger.info("🔌 WebSocket остановлен")


# ============================================================================
# HISTORICAL DATA LOADER
# ============================================================================

def load_historical_candles(http: HTTP, symbol: str, days: int = 100) -> List[Dict]:
    """Загрузить исторические свечи для warmup."""
    logger.info(f"📥 Загрузка исторических данных ({days} дней)...")
    
    all_candles = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    bars_needed = days * 24 * 60  # минутных свечей
    
    while len(all_candles) < bars_needed:
        result = http.get_kline(
            category="linear",
            symbol=symbol,
            interval="1",
            end=end_time,
            limit=1000,
        )
        
        if result['retCode'] != 0:
            logger.error(f"Ошибка загрузки: {result['retMsg']}")
            break
        
        klines = result['result']['list']
        if not klines:
            break
        
        for k in klines:
            ts = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc)
            candle = {
                'timestamp': ts,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            }
            all_candles.append(candle)
        
        end_time = int(klines[-1][0]) - 1
        
        if len(all_candles) % 10000 == 0:
            logger.info(f"  Загружено {len(all_candles):,} свечей...")
    
    # Сортируем по времени
    all_candles.sort(key=lambda x: x['timestamp'])
    
    logger.info(f"✅ Загружено {len(all_candles):,} свечей для warmup")
    return all_candles


# ============================================================================
# MAIN BOT
# ============================================================================

class LiveTradingBot:
    """Основной класс бота."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.engine: Optional[BybitTradingEngine] = None
        self.ws_handler: Optional[BybitWebSocketHandler] = None
        self.running = False
        
    def start(self):
        """Запустить бота."""
        logger.info("=" * 60)
        logger.info("🤖 ЗАПУСК LIVE TRADING BOT")
        logger.info(f"   Symbol: {self.config.symbol}")
        logger.info(f"   Testnet: {self.config.testnet}")
        logger.info(f"   Position size: ${self.config.position_size_usd}")
        logger.info("=" * 60)
        
        try:
            # Создаём движок
            self.engine = BybitTradingEngine(self.config)
            
            # Загружаем исторические данные для warmup
            historical = load_historical_candles(
                self.engine.http, 
                self.config.symbol,
                days=self.config.session_lookback_days + 10
            )
            
            # Прогреваем feature calculator и session detector
            logger.info("🔥 Warmup моделей...")
            for candle in historical:
                self.engine.feature_calculator.add_candle(candle)
                if len(self.engine.feature_calculator.candles) >= 2:
                    prev_close = self.engine.feature_calculator.candles[-2]['close']
                else:
                    prev_close = candle['close']
                self.engine.session_detector.add_candle(
                    candle['timestamp'], 
                    candle['close'], 
                    prev_close
                )
            
            logger.info(f"✅ Warmup завершён: {len(historical):,} свечей")
            
            # Проверяем наличие открытой позиции (восстановление после рестарта)
            self.engine.restore_position_on_startup()
            
            # Запускаем WebSocket
            self.ws_handler = BybitWebSocketHandler(self.engine, self.config)
            self.ws_handler.start()
            
            self.running = True
            
            # Главный цикл
            logger.info("🟢 Бот запущен. Нажмите Ctrl+C для остановки.")
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Получен сигнал остановки...")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Остановить бота."""
        self.running = False
        
        # Закрываем позицию если есть
        if self.engine and self.engine.position:
            logger.info("⚠️ Закрываем открытую позицию...")
            self.engine._close_position("BOT_STOP")
        
        # Останавливаем WebSocket
        if self.ws_handler:
            self.ws_handler.stop()
        
        logger.info("🔴 Бот остановлен")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Точка входа."""
    print("\n" + "=" * 60)
    print("  BYBIT LIVE TRADING BOT (TESTNET)")
    print("=" * 60 + "\n")
    
    # Проверяем API ключи
    api_key = os.environ.get('BYBIT_API_KEY', '')
    api_secret = os.environ.get('BYBIT_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("⚠️  API ключи не найдены в environment variables.")
        print("   Создайте ключи на https://testnet.bybit.com/")
        print("   (API Management -> Create New Key -> System-generated)")
        print()
        api_key = input("Введите API Key: ").strip()
        api_secret = input("Введите API Secret: ").strip()
        
        if not api_key or not api_secret:
            print("❌ Ключи не введены. Выход.")
            return
    
    # Конфигурация
    config = BotConfig()
    config.api_key = api_key
    config.api_secret = api_secret
    
    # Проверяем ключи перед запуском
    print("\n🔑 Проверяем API ключи...")
    try:
        from pybit.unified_trading import HTTP
        http = HTTP(
            testnet=config.testnet, 
            api_key=api_key, 
            api_secret=api_secret,
            demo=config.demo,
            recv_window=20000,
        )
        result = http.get_wallet_balance(accountType="UNIFIED")
        if result['retCode'] == 0:
            coins = result['result']['list'][0]['coin']
            for coin in coins:
                if coin['coin'] == 'USDT':
                    balance = float(coin['walletBalance'])
                    print(f"✅ API ключи валидны! Баланс USDT: {balance:.2f}")
                    break
        else:
            print(f"❌ Ошибка API: {result['retMsg']}")
            print("   Проверьте что ключи созданы на testnet.bybit.com")
            return
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("   Проверьте ключи и интернет-соединение.")
        return
    
    # Запуск
    bot = LiveTradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
