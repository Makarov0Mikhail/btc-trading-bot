"""
Переобучение модели с новыми гиперпараметрами RF.
Модель обучается ОДИН раз, потом быстрый подбор порогов.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Новые гиперпараметры RF (максимально глубокие)
NEW_RF_PARAMS = {
    'n_estimators': 400,      # Много деревьев
    'max_depth': 25,          # Очень глубоко
    'min_samples_split': 5,   # Минимум
    'min_samples_leaf': 3,    # Минимум
    'class_weight': None,
    'n_jobs': -1,
    'random_state': 42,
}

HORIZON = 15


def load_data():
    """Загрузка данных."""
    print("1. Загружаем данные...")
    
    with open(DATA_DIR / "btc_processed_v3.pkl", 'rb') as f:
        pkg = pickle.load(f)
    
    train_data = pkg['train_data']
    val_data = pkg.get('val_data', pd.DataFrame())
    test_data = pkg['test_data']
    
    print(f"   Train: {len(train_data)} баров")
    print(f"   Val: {len(val_data)} баров")
    print(f"   Test: {len(test_data)} баров")
    
    return train_data, val_data, test_data


def prepare_features(df: pd.DataFrame, feature_cols: list, horizon: int):
    """Подготовка фичей и таргетов."""
    
    # Таргет: цена через horizon баров выше/ниже текущей
    df = df.copy()
    future_return = df['Close'].shift(-horizon) / df['Close'] - 1
    
    # Для LONG: цена выросла на > 0.1%
    # Для SHORT: цена упала на > 0.1%
    target_long = (future_return > 0.001).astype(int)
    target_short = (future_return < -0.001).astype(int)
    
    # Убираем последние horizon строк (нет будущего)
    df = df.iloc[:-horizon].copy()
    target_long = target_long.iloc[:-horizon]
    target_short = target_short.iloc[:-horizon]
    
    # Фичи
    X = df[feature_cols].values
    
    # Убираем NaN и Inf
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[valid_mask]
    y_long = target_long.values[valid_mask]
    y_short = target_short.values[valid_mask]
    
    # Заменяем оставшиеся большие значения
    X = np.clip(X, -1e10, 1e10)
    
    return X, y_long, y_short


def train_model(X_train, y_train, rf_params):
    """Обучение модели."""
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, name=""):
    """Оценка модели."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5
    acc = accuracy_score(y, pred)
    
    print(f"   {name}: AUC={auc:.4f}, Acc={acc:.4f}, Pos={y.mean()*100:.1f}%")
    return auc, acc


def main():
    print("=" * 60)
    print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ С НОВЫМИ ГИПЕРПАРАМЕТРАМИ")
    print("=" * 60)
    
    # 1. Загружаем данные
    train_data, val_data, test_data = load_data()
    
    # 2. Загружаем текущую модель для получения feature_cols
    print("\n2. Загружаем текущую конфигурацию...")
    with open(MODELS_DIR / "phase2_h15_final.pkl", 'rb') as f:
        old_pkg = pickle.load(f)
    
    feature_cols = old_pkg['feature_cols']
    print(f"   Фичей: {len(feature_cols)}")
    print(f"   {feature_cols}")
    
    # 3. Подготавливаем данные
    print("\n3. Подготавливаем данные...")
    
    # Train + Val для обучения
    if len(val_data) > 0:
        train_full = pd.concat([train_data, val_data])
    else:
        train_full = train_data
    
    X_train, y_long_train, y_short_train = prepare_features(train_full, feature_cols, HORIZON)
    X_test, y_long_test, y_short_test = prepare_features(test_data, feature_cols, HORIZON)
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Long positives (train): {y_long_train.mean()*100:.1f}%")
    print(f"   Short positives (train): {y_short_train.mean()*100:.1f}%")
    
    # 4. Обучаем модели
    print("\n4. Обучаем модели с новыми гиперпараметрами...")
    print(f"   RF params: {NEW_RF_PARAMS}")
    
    start_time = datetime.now()
    
    print("\n   Обучение модели LONG...")
    model_long = train_model(X_train, y_long_train, NEW_RF_PARAMS)
    
    print("   Обучение модели SHORT...")
    model_short = train_model(X_train, y_short_train, NEW_RF_PARAMS)
    
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\n   Время обучения: {train_time:.1f} сек")
    
    # 5. Оценка
    print("\n5. Оценка моделей:")
    print("   LONG:")
    evaluate_model(model_long, X_train, y_long_train, "Train")
    auc_long_test, _ = evaluate_model(model_long, X_test, y_long_test, "Test")
    
    print("   SHORT:")
    evaluate_model(model_short, X_train, y_short_train, "Train")
    auc_short_test, _ = evaluate_model(model_short, X_test, y_short_test, "Test")
    
    # 6. Сохраняем новую модель
    print("\n6. Сохраняем новую модель...")
    
    new_pkg = {
        'model_long': model_long,
        'model_short': model_short,
        'feature_cols': feature_cols,
        'target_col': f'target_h{HORIZON}',
        'horizon': HORIZON,
        'rf_params': NEW_RF_PARAMS,
        'best_params': {
            'thr_long': 0.55,  # Начальные пороги (потом оптимизируем)
            'thr_short': 0.55,
            'min_confidence': 0.10,
            'stop_mult': 1.5,
            'take_mult': 2.0,
            'trailing_mult': 0.5,
            'breakeven_mult': 0.5,
            'exit_confidence_drop': 0.15,
        },
        'params': {},
        'train_time': train_time,
        'auc_long_test': auc_long_test,
        'auc_short_test': auc_short_test,
    }
    new_pkg['params'] = new_pkg['best_params'].copy()
    
    with open(MODELS_DIR / "phase2_h15_final.pkl", 'wb') as f:
        pickle.dump(new_pkg, f)
    
    print("   Модель сохранена!")
    
    # 7. Быстрый подбор порогов
    print("\n" + "=" * 60)
    print("7. БЫСТРЫЙ ПОДБОР ПОРОГОВ")
    print("=" * 60)
    
    from optimize_thresholds_fast import load_and_precompute, simulate_trading
    import optuna
    from optuna.samplers import TPESampler
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    data = load_and_precompute()
    
    print("\n   Тестируем с начальными порогами...")
    initial_stats = simulate_trading(data, new_pkg['best_params'])
    print(f"   Сделок: {initial_stats['trades']}, Капитал: ${initial_stats['capital']:.2f}")
    
    print("\n   Запускаем оптимизацию порогов (5000 trials)...")
    
    def objective(trial):
        params = {
            'thr_long': trial.suggest_float('thr_long', 0.45, 0.85),
            'thr_short': trial.suggest_float('thr_short', 0.45, 0.85),
            'min_confidence': trial.suggest_float('min_confidence', 0.01, 0.30),
            'stop_mult': trial.suggest_float('stop_mult', 0.3, 3.0),
            'take_mult': trial.suggest_float('take_mult', 0.5, 5.0),
        }
        
        stats = simulate_trading(data, params)
        
        if stats['trades'] < 3:
            return -1000.0
        
        return stats['capital'] - 10000
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=5000, show_progress_bar=True)
    
    best_params = study.best_params
    
    print(f"\n   Лучшие пороги:")
    for k, v in best_params.items():
        print(f"     {k}: {v:.4f}")
    
    # Финальный тест
    final_stats = simulate_trading(data, best_params)
    
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print("=" * 60)
    print(f"Сделок: {final_stats['trades']}")
    print(f"Капитал: ${final_stats['capital']:.2f} ({(final_stats['capital']-10000)/100:+.2f}%)")
    print(f"Win Rate: {final_stats['win_rate']:.1f}%")
    print(f"Profit Factor: {final_stats['pf']:.2f}")
    print(f"Max Drawdown: {final_stats['max_dd']:.2f}%")
    
    # Сохраняем лучшие параметры
    full_params = new_pkg['best_params'].copy()
    full_params.update(best_params)
    new_pkg['best_params'] = full_params
    new_pkg['params'] = full_params.copy()
    
    with open(MODELS_DIR / "phase2_h15_final.pkl", 'wb') as f:
        pickle.dump(new_pkg, f)
    
    print("\n   Параметры сохранены!")
    print("=" * 60)


if __name__ == "__main__":
    main()
