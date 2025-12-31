"""Валидация ордеров на Bybit Demo."""
from pybit.unified_trading import HTTP
import os

api_key = os.environ.get('BYBIT_API_KEY', 'ZVtFkU98lwei7fhuo4')
api_secret = os.environ.get('BYBIT_API_SECRET', 'f4ruVkXNP5vIkVkqc0iuj76ze0DbQqe3pI6y')

http = HTTP(testnet=False, api_key=api_key, api_secret=api_secret, demo=True, recv_window=20000)

print('=' * 60)
print('  ВАЛИДАЦИЯ ОРДЕРОВ НА BYBIT DEMO')
print('=' * 60)

# История ордеров
print('\n📋 ИСТОРИЯ ОРДЕРОВ (последние):')
orders = http.get_order_history(category='linear', symbol='BTCUSDT', limit=10)
if orders['retCode'] == 0:
    for o in orders['result']['list'][:5]:
        print('Order ID:', o['orderId'])
        print('  Side:', o['side'], '| Type:', o['orderType'])
        print('  Qty:', o['qty'], 'BTC')
        print('  AvgPrice:', o.get('avgPrice', 'N/A'))
        print('  Status:', o['orderStatus'])
        print('  StopLoss:', o.get('stopLoss', 'N/A'))
        print('  TakeProfit:', o.get('takeProfit', 'N/A'))
        print('  Created:', o['createdTime'])
        print()

# История закрытых позиций (PnL)
print('\n💰 ИСТОРИЯ P&L (closed trades):')
pnl = http.get_closed_pnl(category='linear', symbol='BTCUSDT', limit=5)
if pnl['retCode'] == 0:
    for p in pnl['result']['list'][:3]:
        print('Order ID:', p['orderId'])
        print('  Side:', p['side'])
        print('  Qty:', p['qty'], 'BTC')
        print('  Entry:', p['avgEntryPrice'])
        print('  Exit:', p['avgExitPrice'])
        print('  PnL:', p['closedPnl'], 'USDT')
        print()

# Текущий баланс
print('\n💵 ТЕКУЩИЙ БАЛАНС:')
bal = http.get_wallet_balance(accountType='UNIFIED')
if bal['retCode'] == 0:
    for coin in bal['result']['list'][0]['coin']:
        if float(coin.get('walletBalance', 0)) > 0:
            print(' ', coin['coin'], ':', coin['walletBalance'])
