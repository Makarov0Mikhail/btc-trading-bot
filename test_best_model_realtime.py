"""
Тестирование лучшей модели (phase2_h15_final.pkl) на 3 месяцах 
в максимально реалистичном режиме через realtime_backtester.

Особенности теста:
1. Сырые OHLCV данные (без предрасчитанных фичей)
2. Сессии детектируются в realtime (как на реальной бирже)
3. Фичи рассчитываются посвечно без look-ahead bias
4. Решения модели принимаются на каждой свече
5. Slippage и комиссии учитываются
"""

import sys
import os

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

from realtime_backtester import (
    run_backtest,
    TradingConfig,
    SessionConfig,
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def main():
    print("=" * 70)
    print("ТЕСТ ЛУЧШЕЙ МОДЕЛИ В REALTIME РЕЖИМЕ")
    print("=" * 70)
    
    # Проверяем модель
    model_path = MODELS_DIR / "phase2_h15_final.pkl"
    print(f"\n1. Загружаем модель: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    print(f"   Horizon: {model_pkg.get('horizon', 15)}")
    print(f"   Features: {model_pkg.get('feature_cols', [])}")
    print(f"   Best params: {model_pkg.get('best_params', {})}")
    print(f"   RF params: {model_pkg.get('rf_params', {})}")
    print(f"   AUC Long Test: {model_pkg.get('auc_long_test', 'N/A'):.4f}")
    print(f"   AUC Short Test: {model_pkg.get('auc_short_test', 'N/A'):.4f}")
    
    # Загружаем данные для проверки тестового периода
    print(f"\n2. Проверяем данные...")
    with open(DATA_DIR / "btc_processed_v3.pkl", 'rb') as f:
        data_pkg = pickle.load(f)
    
    test_data = data_pkg['test_data']
    print(f"   Тестовые данные: {test_data.index.min()} - {test_data.index.max()}")
    print(f"   Всего баров: {len(test_data)}")
    
    # 3 месяца тестирования (последние 3 месяца из тестовых данных)
    # Данные: 2025-06-16 до 2025-12-16
    # Берём последние 3 месяца: 2025-09-16 до 2025-12-16
    test_end = test_data.index.max()
    test_start = test_end - timedelta(days=90)  # ~3 месяца
    
    # Округляем до начала дня
    test_start = test_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    print(f"\n3. Тестовый период: {test_start.date()} - {test_end.date()}")
    test_subset = test_data[(test_data.index >= test_start) & (test_data.index <= test_end)]
    print(f"   Баров в периоде: {len(test_subset)}")
    
    # Запускаем бэктест
    print("\n" + "=" * 70)
    print("ЗАПУСК REALTIME БЭКТЕСТА")
    print("=" * 70)
    
    result = run_backtest(
        data_path=str(DATA_DIR / "btc_processed_v3.pkl"),
        model_path=str(model_path),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        initial_capital=10000.0,
        verbose=True,
    )
    
    # Расширенная статистика
    stats = result['stats']
    trades = result['trades']
    
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА")
    print("=" * 70)
    
    # Анализ по направлениям
    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']
    
    if long_trades:
        long_wins = [t for t in long_trades if t['pnl_pct'] > 0]
        long_avg = sum(t['pnl_pct'] for t in long_trades) / len(long_trades)
        print(f"\nLONG сделки: {len(long_trades)}")
        print(f"  Win Rate: {len(long_wins)/len(long_trades)*100:.1f}%")
        print(f"  Avg PnL: {long_avg:+.2f}%")
    
    if short_trades:
        short_wins = [t for t in short_trades if t['pnl_pct'] > 0]
        short_avg = sum(t['pnl_pct'] for t in short_trades) / len(short_trades)
        print(f"\nSHORT сделки: {len(short_trades)}")
        print(f"  Win Rate: {len(short_wins)/len(short_trades)*100:.1f}%")
        print(f"  Avg PnL: {short_avg:+.2f}%")
    
    # Анализ по причинам выхода
    print("\nВыходы по причинам:")
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl_sum': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl_sum'] += t['pnl_pct']
    
    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        avg_pnl = data['pnl_sum'] / data['count'] if data['count'] > 0 else 0
        print(f"  {reason}: {data['count']} ({avg_pnl:+.2f}% avg)")
    
    # Ежемесячная разбивка
    from collections import defaultdict
    monthly = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
    
    for t in trades:
        # Парсим дату
        if isinstance(t['exit_time'], str):
            exit_time = datetime.fromisoformat(t['exit_time'].replace('Z', '+00:00'))
        else:
            exit_time = t['exit_time']
        month_key = exit_time.strftime('%Y-%m')
        monthly[month_key]['trades'] += 1
        monthly[month_key]['pnl'] += t['pnl_pct']
        if t['pnl_pct'] > 0:
            monthly[month_key]['wins'] += 1
    
    print("\nПо месяцам:")
    for month, data in sorted(monthly.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        print(f"  {month}: {data['trades']} сделок, {data['pnl']:+.2f}% PnL, {wr:.1f}% WR")
    
    # Итого
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 70)
    print(f"Начальный капитал: $10,000.00")
    print(f"Конечный капитал: ${stats['final_capital']:,.2f}")
    print(f"Общая прибыль: ${stats['final_capital'] - 10000:+,.2f} ({(stats['final_capital']/10000-1)*100:+.2f}%)")
    print(f"")
    print(f"Всего сделок: {stats['trades']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    print("=" * 70)
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f"best_model_realtime_test_{timestamp}.json"
    
    output = {
        'model': 'phase2_h15_final.pkl',
        'test_period': {
            'start': str(test_start.date()),
            'end': str(test_end.date()),
            'bars': len(test_subset),
        },
        'model_params': model_pkg.get('best_params', {}),
        'rf_params': model_pkg.get('rf_params', {}),
        'stats': stats,
        'trades': trades,
        'monthly_breakdown': dict(monthly),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nРезультаты сохранены: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
