"""
Тест модели на свежих данных за последний месяц.
Загружает СЫРЫЕ свечи с Bybit и прогоняет через СУЩЕСТВУЮЩИЙ realtime бэктестер.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
import time

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np

from phase1_preprocessing import DownloadConfig, download_btc_data
from realtime_backtester import (
    TradingEngine, TradingConfig, SessionConfig,
    RealtimeFeatureCalculator, HistoricalCandleProvider
)

# Директории
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("=" * 60)
    print("ТЕСТ МОДЕЛИ НА СВЕЖИХ ДАННЫХ (1 МЕСЯЦ)")
    print("=" * 60)
    
    # =========================================================================
    # 1. ЗАГРУЗКА МОДЕЛИ
    # =========================================================================
    model_path = MODELS_DIR / "phase2_h15_final.pkl"
    print(f"\n📂 Загрузка модели: {model_path.name}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model_long = model_data['model_long']
    model_short = model_data['model_short']
    feature_cols = model_data['feature_cols']
    horizon = model_data.get('horizon', 15)
    best_params = model_data.get('best_params', {})
    
    print(f"   Horizon: {horizon}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Params: thr_long={best_params.get('thr_long', 0.55):.4f}, "
          f"thr_short={best_params.get('thr_short', 0.55):.4f}")
    
    # =========================================================================
    # 2. ЗАГРУЗКА СЫРЫХ СВЕЧЕЙ: 100 дней прогрева + 30 дней теста
    # =========================================================================
    end_date = datetime.now()
    # Загружаем 130 дней: 100 для прогрева session_detector + 30 для теста
    start_date = end_date - timedelta(days=130)
    test_start_date = end_date - timedelta(days=30)
    
    print(f"\n📥 Загрузка сырых свечей с Bybit...")
    print(f"   Полный период: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')} (130 дней)")
    print(f"   Тестовый период: {test_start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')} (30 дней)")
    
    cfg = DownloadConfig()
    cfg.start_date = start_date
    cfg.end_date = end_date
    cfg.interval = "1m"
    cfg.source = "bybit"
    cfg.symbol = "BTC/USDT:USDT"
    
    try:
        raw_df = download_btc_data(cfg)
    except Exception as e:
        print(f"\n❌ Ошибка загрузки: {e}")
        return
    
    # Разделяем на прогрев и тест
    warmup_df = raw_df[raw_df.index < test_start_date]
    test_df = raw_df[raw_df.index >= test_start_date]
    
    print(f"\n✅ Загружено {len(raw_df):,} свечей (сырые OHLCV)")
    print(f"   Прогрев: {len(warmup_df):,} свечей ({len(warmup_df)//1440} дней)")
    print(f"   Тест: {len(test_df):,} свечей ({len(test_df)//1440} дней)")
    print(f"   Колонки: {list(raw_df.columns)}")
    
    # =========================================================================
    # 3. НАСТРОЙКА ТОРГОВОГО ДВИЖКА
    # =========================================================================
    trading_config = TradingConfig()
    trading_config.horizon = horizon
    trading_config.thr_long = best_params.get('thr_long', 0.55)
    trading_config.thr_short = best_params.get('thr_short', 0.55)
    trading_config.min_confidence = best_params.get('min_confidence', 0.05)
    trading_config.stop_mult = best_params.get('stop_mult', 2.5)
    trading_config.take_mult = best_params.get('take_mult', 1.0)
    
    # Обновляем feature_cols в калькуляторе
    RealtimeFeatureCalculator.FEATURE_COLS = feature_cols
    
    session_config = SessionConfig()
    
    print(f"\n⚙️ Торговые параметры:")
    print(f"   thr_long={trading_config.thr_long:.4f}, thr_short={trading_config.thr_short:.4f}")
    print(f"   stop_mult={trading_config.stop_mult:.2f}, take_mult={trading_config.take_mult:.2f}")
    
    # =========================================================================
    # 4. ПРОГРЕВ НА 100 ДНЯХ + БЭКТЕСТ НА 30 ДНЯХ
    # =========================================================================
    print(f"\n🔧 Инициализация торгового движка...")
    
    engine = TradingEngine(
        model_long=model_long,
        model_short=model_short,
        trading_config=trading_config,
        session_config=session_config,
    )
    engine.reset(10000.0)
    
    # ПРОГРЕВ на 100 днях (для session_detector нужен lookback_days=90)
    print(f"\n🔥 Прогрев на {len(warmup_df):,} барах ({len(warmup_df)//1440} дней)...")
    warmup_provider = HistoricalCandleProvider(warmup_df)
    
    prev_close = None
    for candle in warmup_provider:
        # Добавляем в feature calculator
        engine.feature_calculator.add_candle(candle)
        # Добавляем в session detector
        if prev_close is not None:
            engine.session_detector.add_candle(candle['timestamp'], candle['close'], prev_close)
        prev_close = candle['close']
    
    print(f"   Прогрев завершён. Волатильность: {len(engine.session_detector._vol_cache)} точек")
    
    # БЭКТЕСТ на 30 днях
    print(f"\n🚀 Запуск бэктеста на {len(test_df):,} барах (30 дней)...")
    print("-" * 60)
    
    candle_provider = HistoricalCandleProvider(test_df)
    
    total_bars = len(test_df)
    last_pct = 0
    start_time = time.time()
    session_bars = 0
    
    for i, candle in enumerate(candle_provider):
        engine.process_candle(candle)
        
        # Считаем бары в сессиях
        in_session, _, _ = engine.session_detector.is_in_session(candle['timestamp'])
        if in_session:
            session_bars += 1
        
        # Прогресс каждые 10%
        pct = int((i + 1) / total_bars * 100)
        if pct >= last_pct + 10:
            last_pct = pct
            trades_so_far = len(engine.trades)
            capital = engine.capital
            sessions = len(engine.session_detector.sessions)
            print(f"   [{pct:3d}%] Бар {i+1}/{total_bars} | "
                  f"Сессий: {sessions} | Сделок: {trades_so_far} | Капитал: ${capital:.2f}")
    
    elapsed = time.time() - start_time
    print("-" * 60)
    
    # =========================================================================
    # 5. РЕЗУЛЬТАТЫ
    # =========================================================================
    stats = engine.get_stats()
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ НА СВЕЖИХ ДАННЫХ")
    print("=" * 60)
    
    print(f"\n📈 Период: {test_df.index.min()} - {test_df.index.max()}")
    days_tested = (test_df.index.max() - test_df.index.min()).days
    print(f"   Дней: {days_tested}")
    
    total_pnl = stats.get('total_pnl', 0)
    win_rate = stats.get('win_rate', 0)
    profit_factor = stats.get('profit_factor', 0)
    max_dd = stats.get('max_drawdown', 0)
    n_trades = stats.get('trades', 0)
    
    print(f"\n💰 Прибыль: {total_pnl:+.2f}%")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print(f"📈 Profit Factor: {profit_factor:.2f}")
    print(f"📉 Max Drawdown: {max_dd:.2f}%")
    print(f"🔢 Сделок: {n_trades} ({stats.get('wins', 0)} WIN / {stats.get('losses', 0)} LOSE)")
    print(f"📊 Avg Win: {stats.get('avg_win', 0):+.2f}% | Avg Loss: {stats.get('avg_loss', 0):.2f}%")
    
    print(f"\n📍 Сессий обнаружено: {len(engine.session_detector.sessions)}")
    print(f"   Баров в сессиях: {session_bars} / {total_bars} ({session_bars/total_bars*100:.1f}%)")
    
    print(f"\n🔧 Причины выхода:")
    for reason, count in stats['exit_reasons'].items():
        print(f"   {reason}: {count}")
    
    # Экстраполяция на год
    if days_tested > 0 and total_pnl != 0:
        annual = total_pnl * (365 / days_tested)
        print(f"\n📅 Экстраполяция на год: {annual:.1f}%")
        print(f"   С 3x плечом: ~{annual * 3:.1f}%")
    
    print(f"\n⏱️ Время: {elapsed:.1f} сек")
    
    # Последние сделки
    if engine.trades:
        print(f"\n📝 Сделки ({len(engine.trades)} шт):")
        print("-" * 90)
        for t in engine.trades[-15:]:
            entry = t.entry_time.strftime('%m-%d %H:%M') if t.entry_time else '?'
            exit_t = t.exit_time.strftime('%m-%d %H:%M') if t.exit_time else '?'
            pnl = t.pnl_pct * 100
            icon = "✅" if pnl > 0 else "❌"
            print(f"  {icon} {t.direction:5} | {entry} → {exit_t} | "
                  f"PnL: {pnl:+.3f}% | {t.exit_reason}")
    
    # =========================================================================
    # 6. СОХРАНЕНИЕ
    # =========================================================================
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"fresh_data_test_{timestamp}.json"
    
    save_data = {
        'test_period': {
            'start': str(test_df.index.min()),
            'end': str(test_df.index.max()),
            'days': days_tested,
            'bars': len(test_df)
        },
        'stats': stats,
        'model_params': best_params,
        'trades': [
            {
                'direction': t.direction,
                'entry_time': str(t.entry_time),
                'exit_time': str(t.exit_time),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct * 100,
                'exit_reason': t.exit_reason
            }
            for t in engine.trades
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Сохранено: {results_file.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
