# BTC Trading Bot - Bybit Demo

Автоматический торговый бот для BTC/USDT на Bybit Demo Trading.

## Быстрый старт

### 1. Установка (первый раз)
```bash
# Создать виртуальное окружение
python -m venv btc_trading_env

# Активировать (Windows)
btc_trading_env\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### 2. Настройка API ключей
1. Зайти на [bybit.com](https://bybit.com) → Demo Trading
2. Создать API ключи (с разрешениями на торговлю)
3. Установить переменные окружения:

**Windows (PowerShell):**
```powershell
$env:BYBIT_API_KEY = "ваш_ключ"
$env:BYBIT_API_SECRET = "ваш_секрет"
```

**Или отредактировать `start_bot.bat`**

### 3. Запуск
```bash
# PowerShell
$env:BYBIT_API_KEY = "ключ"; $env:BYBIT_API_SECRET = "секрет"; python live_trading_bot.py

# Или через bat файл (предварительно настроить ключи)
start_bot.bat
```

## Параметры

Настройки в `live_trading_bot.py` (class BotConfig):

| Параметр | Значение | Описание |
|----------|----------|----------|
| position_size_usd | 10000 | Размер позиции в USD |
| leverage | 3 | Плечо |
| horizon | 15 | Горизонт модели (минут) |
| thr_long | 0.5944 | Порог для LONG |
| thr_short | 0.5453 | Порог для SHORT |

## Что делает бот

1. **Session Detector** - определяет моменты высокой волатильности (95-й перцентиль за 90 дней)
2. **ML Model** - RandomForest предсказывает направление на 15 минут вперёд
3. **Risk Management** - SL/TP на основе ATR, MODEL_EXIT при падении confidence

## Результаты бэктеста

- Период: 30 дней свежих данных
- Профит: **+4.26%** (при депозите $10k × 3x = $1278)
- Win Rate: **92.3%** (12/13 сделок)
- Max Drawdown: -0.53%

## Файлы

```
live_trading_bot.py  - Основной бот
requirements.txt     - Зависимости
start_bot.bat        - Скрипт запуска (Windows)
models/              - Обученные ML модели
logs/                - Логи торговли
```

## Безопасность

⚠️ Это Demo Trading - используйте только с демо аккаунтом!

Для реальной торговли:
- Измените `demo: bool = False` в BotConfig
- Проверьте все параметры риска
- Начните с минимальной суммы
