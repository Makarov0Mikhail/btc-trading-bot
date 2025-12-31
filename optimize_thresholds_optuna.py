"""
Оптимизация торговых порогов через Optuna БЕЗ переобучения модели.

Стратегия:
1. Загружаем обученную модель (не трогаем)
2. Предвычисляем предсказания модели ОДИН раз
3. Optuna подбирает торговые параметры на быстром симуляторе
4. Финальная валидация на realtime бэктестере

Параметры для оптимизации:
- thr_long, thr_short: пороги входа
- min_confidence: минимальная разница вероятностей
- stop_mult, take_mult: множители стопа и тейка
"""

import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Добавляем src в path для realtime бэктестера
sys.path.insert(0, str(BASE_DIR / "src"))

# Конфигурация симулятора
COMMISSION = 0.0007
SLIPPAGE = 0.0001
POSITION_PCT = 1.0
MAX_TRADES_PER_SESSION = 2
HORIZON = 15


def load_and_precompute(test_start: str = "2025-09-17", test_end: str = "2025-12-16"):
    """Загружаем данные и предвычисляем предсказания модели."""
    
    print("1. Загружаем данные и модели...")
    
    with open(DATA_DIR / "btc_processed_v3.pkl", 'rb') as f:
        pkg = pickle.load(f)
    
    test_data = pkg['test_data']
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    test_data = test_data[(test_data.index >= test_start_dt) & (test_data.index <= test_end_dt)]
    
    with open(MODELS_DIR / "phase2_h15_final.pkl", 'rb') as f:
        model_pkg = pickle.load(f)
    
    model_long = model_pkg['model_long']
    model_short = model_pkg['model_short']
    feature_cols = model_pkg['feature_cols']
    current_params = model_pkg.get('best_params', {})
    
    print(f"   Период: {test_start} - {test_end}")
    print(f"   Тестовых баров: {len(test_data)}")
    print(f"   Текущие параметры: {current_params}")
    
    # Предвычисляем предсказания
    print("\n2. Предвычисляем предсказания модели...")
    
    X = test_data[feature_cols].values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    
    proba_long = np.zeros(len(test_data))
    proba_short = np.zeros(len(test_data))
    
    proba_long[valid_mask] = model_long.predict_proba(X[valid_mask])[:, 1]
    proba_short[valid_mask] = model_short.predict_proba(X[valid_mask])[:, 1]
    
    # Сессии из данных
    print("3. Извлекаем сессии...")
    in_session = test_data['in_session'].values if 'in_session' in test_data.columns else np.zeros(len(test_data))
    session_id = test_data['session_id'].values if 'session_id' in test_data.columns else np.zeros(len(test_data))
    
    n_sessions = len(np.unique(session_id[session_id > 0]))
    print(f"   Сессий: {n_sessions}")
    
    # Цены и ATR
    close = test_data['Close'].values
    high = test_data['High'].values
    low = test_data['Low'].values
    
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).rolling(14).mean().values
    atr_pct = atr / close
    
    print("4. Предвычисления завершены!")
    
    return {
        'proba_long': proba_long,
        'proba_short': proba_short,
        'in_session': in_session,
        'session_id': session_id,
        'close': close,
        'high': high,
        'low': low,
        'atr_pct': atr_pct,
        'valid_mask': valid_mask,
        'index': test_data.index,
        'n_bars': len(test_data),
        'current_params': current_params,
        'model_pkg': model_pkg,
    }


def simulate_trading(data: dict, params: dict) -> dict:
    """Быстрая симуляция торговли."""
    
    thr_long = params['thr_long']
    thr_short = params['thr_short']
    min_confidence = params['min_confidence']
    stop_mult = params['stop_mult']
    take_mult = params['take_mult']
    
    proba_long = data['proba_long']
    proba_short = data['proba_short']
    in_session = data['in_session']
    session_id = data['session_id']
    close = data['close']
    high = data['high']
    low = data['low']
    atr_pct = data['atr_pct']
    valid_mask = data['valid_mask']
    n_bars = data['n_bars']
    
    # Сигналы
    proba_diff = proba_long - proba_short
    
    # LONG: proba_long >= thr_long AND proba_short < thr_short AND diff >= min_conf
    signal_long = ((proba_long >= thr_long) & 
                   (proba_short < thr_short) & 
                   (proba_diff >= min_confidence))
    
    # SHORT: proba_short >= thr_short AND proba_long < thr_long AND diff <= -min_conf
    signal_short = ((proba_short >= thr_short) & 
                    (proba_long < thr_long) & 
                    (proba_diff <= -min_confidence))
    
    # Только в сессиях и валидные
    signal_long = signal_long & (in_session == 1) & valid_mask
    signal_short = signal_short & (in_session == 1) & valid_mask
    
    # Симуляция
    trades = []
    capital = 10000.0
    max_capital = capital
    min_capital = capital
    session_trades = {}
    
    i = 0
    while i < n_bars - HORIZON:
        sess = session_id[i]
        
        if sess > 0 and session_trades.get(sess, 0) >= MAX_TRADES_PER_SESSION:
            i += 1
            continue
        
        if signal_long[i] or signal_short[i]:
            direction = 'LONG' if signal_long[i] else 'SHORT'
            entry_price = close[i]
            
            atr = atr_pct[i] if not np.isnan(atr_pct[i]) else 0.005
            base_stop = max(0.005, atr * 1.5)
            stop_pct = min(base_stop * stop_mult, 0.03)
            take_pct = stop_pct * take_mult
            
            position_value = capital * POSITION_PCT
            
            # Ищем выход
            exit_price = None
            exit_reason = None
            exit_idx = i + 1
            
            for j in range(i + 1, min(i + HORIZON + 1, n_bars)):
                if direction == 'LONG':
                    max_pnl = high[j] / entry_price - 1
                    min_pnl = low[j] / entry_price - 1
                else:
                    max_pnl = entry_price / low[j] - 1
                    min_pnl = entry_price / high[j] - 1
                
                # Stop loss
                if min_pnl <= -stop_pct:
                    exit_price = entry_price * (1 - stop_pct) if direction == 'LONG' else entry_price * (1 + stop_pct)
                    exit_reason = 'STOP_LOSS'
                    exit_idx = j
                    break
                
                # Take profit
                if max_pnl >= take_pct:
                    exit_price = entry_price * (1 + take_pct) if direction == 'LONG' else entry_price * (1 - take_pct)
                    exit_reason = 'TAKE_PROFIT'
                    exit_idx = j
                    break
                
                # Конец сессии
                if in_session[j] == 0 and j > 0 and in_session[j-1] == 1:
                    exit_price = close[j]
                    exit_reason = 'SESSION_END'
                    exit_idx = j
                    break
            
            # Horizon exit
            if exit_price is None:
                exit_idx = min(i + HORIZON, n_bars - 1)
                exit_price = close[exit_idx]
                exit_reason = 'HORIZON_EXIT'
            
            # PnL
            if direction == 'LONG':
                pnl_pct = exit_price / entry_price - 1
            else:
                pnl_pct = entry_price / exit_price - 1
            
            # Комиссии и slippage
            total_costs = position_value * (COMMISSION * 2 + SLIPPAGE * 2)
            pnl_usd = position_value * pnl_pct - total_costs
            
            trades.append({
                'direction': direction,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'exit_reason': exit_reason,
            })
            
            capital += pnl_usd
            max_capital = max(max_capital, capital)
            min_capital = min(min_capital, capital)
            session_trades[sess] = session_trades.get(sess, 0) + 1
            
            i = exit_idx + 1
        else:
            i += 1
    
    # Статистика
    if not trades:
        return {'trades': 0, 'capital': 10000.0, 'pnl': 0, 'win_rate': 0, 'pf': 0, 'max_dd': 0}
    
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    
    gross_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    max_dd = (max_capital - min_capital) / max_capital * 100 if max_capital > 0 else 0
    
    return {
        'trades': len(trades),
        'capital': capital,
        'pnl': capital - 10000,
        'pnl_pct': (capital - 10000) / 100,
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'pf': pf,
        'max_dd': max_dd,
        'wins': len(wins),
        'losses': len(losses),
    }


def optimize_thresholds(data: dict, n_trials: int = 2000) -> dict:
    """Оптимизация порогов через Optuna."""
    
    print("\n" + "=" * 60)
    print(f"ОПТИМИЗАЦИЯ ПОРОГОВ ({n_trials} trials)")
    print("=" * 60)
    
    current_params = data['current_params']
    
    # Тест с текущими параметрами
    print("\nТекущие параметры:")
    current_stats = simulate_trading(data, current_params)
    print(f"  Сделок: {current_stats['trades']}, Капитал: ${current_stats['capital']:.2f}, "
          f"WR: {current_stats['win_rate']:.1f}%, PF: {current_stats['pf']:.2f}")
    
    def objective(trial):
        params = {
            'thr_long': trial.suggest_float('thr_long', 0.45, 0.80),
            'thr_short': trial.suggest_float('thr_short', 0.45, 0.80),
            'min_confidence': trial.suggest_float('min_confidence', 0.05, 0.35),
            'stop_mult': trial.suggest_float('stop_mult', 0.5, 4.0),
            'take_mult': trial.suggest_float('take_mult', 0.5, 4.0),
        }
        
        stats = simulate_trading(data, params)
        
        # Минимум сделок
        if stats['trades'] < 20:
            return -1000.0
        
        # Целевая функция: прибыль с бонусом за WR и штрафом за DD
        score = stats['pnl']
        
        # Бонус за win rate > 60%
        if stats['win_rate'] > 60:
            score += (stats['win_rate'] - 60) * 5
        
        # Штраф за drawdown > 2%
        if stats['max_dd'] > 2:
            score -= (stats['max_dd'] - 2) * 50
        
        # Бонус за profit factor > 2
        if stats['pf'] > 2:
            score += (stats['pf'] - 2) * 20
        
        return score
    
    sampler = TPESampler(seed=42, n_startup_trials=100)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Добавляем текущие параметры как стартовую точку
    study.enqueue_trial({
        'thr_long': current_params.get('thr_long', 0.6),
        'thr_short': current_params.get('thr_short', 0.6),
        'min_confidence': current_params.get('min_confidence', 0.15),
        'stop_mult': current_params.get('stop_mult', 1.5),
        'take_mult': current_params.get('take_mult', 2.0),
    })
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_stats = simulate_trading(data, best_params)
    
    print("\n" + "=" * 60)
    print("ЛУЧШИЕ ПАРАМЕТРЫ:")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nРезультат:")
    print(f"  Сделок: {best_stats['trades']}")
    print(f"  Капитал: ${best_stats['capital']:.2f} ({best_stats['pnl_pct']:+.2f}%)")
    print(f"  Win Rate: {best_stats['win_rate']:.1f}%")
    print(f"  Profit Factor: {best_stats['pf']:.2f}")
    print(f"  Max Drawdown: {best_stats['max_dd']:.2f}%")
    
    return {
        'best_params': best_params,
        'best_stats': best_stats,
        'current_stats': current_stats,
    }


def validate_on_realtime(best_params: dict, model_pkg: dict):
    """Валидация лучших параметров на realtime бэктестере."""
    
    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ НА REALTIME БЭКТЕСТЕРЕ")
    print("=" * 60)
    
    from realtime_backtester import (
        run_backtest, TradingConfig, SessionConfig, 
        RealtimeFeatureCalculator
    )
    
    # Обновляем модель с новыми параметрами
    model_pkg_updated = model_pkg.copy()
    
    # Сохраняем trailing/breakeven/exit_confidence_drop из текущих
    full_params = model_pkg.get('best_params', {}).copy()
    full_params.update(best_params)
    model_pkg_updated['best_params'] = full_params
    
    # Временно сохраняем для realtime бэктестера
    temp_model_path = MODELS_DIR / "phase2_h15_temp_optimized.pkl"
    with open(temp_model_path, 'wb') as f:
        pickle.dump(model_pkg_updated, f)
    
    print(f"Временная модель: {temp_model_path}")
    print("Запуск realtime бэктеста (может занять несколько минут)...")
    
    result = run_backtest(
        data_path=str(DATA_DIR / "btc_processed_v3.pkl"),
        model_path=str(temp_model_path),
        test_start="2025-09-17",
        test_end="2025-12-16",
        initial_capital=10000.0,
        verbose=True,
    )
    
    return result


def main():
    print("=" * 70)
    print("ОПТИМИЗАЦИЯ ТОРГОВЫХ ПОРОГОВ (БЕЗ ПЕРЕОБУЧЕНИЯ МОДЕЛИ)")
    print("=" * 70)
    
    # 1. Загрузка и предвычисление
    data = load_and_precompute(test_start="2025-09-17", test_end="2025-12-16")
    
    # 2. Оптимизация на быстром симуляторе
    result = optimize_thresholds(data, n_trials=3000)
    
    best_params = result['best_params']
    
    # 3. Спрашиваем про валидацию
    print("\n" + "=" * 60)
    answer = input("Запустить валидацию на realtime бэктестере? (y/n): ").strip().lower()
    
    if answer == 'y':
        realtime_result = validate_on_realtime(best_params, data['model_pkg'])
        
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        fast_stats = result['best_stats']
        rt_stats = realtime_result['stats']
        
        print(f"{'Метрика':<20} {'Fast Sim':>15} {'Realtime':>15}")
        print("-" * 50)
        print(f"{'Сделок':<20} {fast_stats['trades']:>15} {rt_stats['trades']:>15}")
        print(f"{'Капитал':<20} ${fast_stats['capital']:>14.2f} ${rt_stats['final_capital']:>14.2f}")
        print(f"{'Win Rate':<20} {fast_stats['win_rate']:>14.1f}% {rt_stats['win_rate']:>14.1f}%")
        print(f"{'Profit Factor':<20} {fast_stats['pf']:>15.2f} {rt_stats['profit_factor']:>15.2f}")
        print(f"{'Max DD':<20} {fast_stats['max_dd']:>14.2f}% {rt_stats['max_drawdown']:>14.2f}%")
    
    # 4. Спрашиваем про сохранение
    print("\n" + "=" * 60)
    save_answer = input("Сохранить лучшие параметры в модель? (y/n): ").strip().lower()
    
    if save_answer == 'y':
        model_pkg = data['model_pkg'].copy()
        full_params = model_pkg.get('best_params', {}).copy()
        full_params.update(best_params)
        model_pkg['best_params'] = full_params
        model_pkg['params'] = full_params.copy()
        
        with open(MODELS_DIR / "phase2_h15_final.pkl", 'wb') as f:
            pickle.dump(model_pkg, f)
        
        print("Параметры сохранены в phase2_h15_final.pkl!")
    
    # 5. Сохраняем результаты оптимизации
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'best_params': best_params,
        'fast_stats': result['best_stats'],
        'current_params': data['current_params'],
        'current_stats': result['current_stats'],
        'timestamp': datetime.now().isoformat(),
    }
    
    import json
    output_path = RESULTS_DIR / f"threshold_optimization_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nРезультаты сохранены: {output_path}")


if __name__ == "__main__":
    main()
