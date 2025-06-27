import time
from datetime import datetime, timedelta
import pandas as pd
from pybit.unified_trading import HTTP
import yaml
from logger_module import log_and_notify

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

BYBIT_API_KEY = config['bybit']['api_key']
BYBIT_API_SECRET = config['bybit']['api_secret']
TESTNET = config['bybit']['testnet']

symbol = config['trading']['symbol']
qty = config['trading']['qty']

SMA_SHORT = 50
SMA_LONG = 200

TP_PCT = config['strategy']['tp_pct']
SL_PCT = config['strategy']['sl_pct']
COOLDOWN_MINUTES = config['strategy']['cooldown_minutes']

STRATEGY_NAME = 'SMA50/200'

session = HTTP(
    testnet=TESTNET,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

def get_ohlcv(symbol, interval, limit):
    candles = session.get_kline(symbol=symbol, interval=interval, limit=limit)['result']['list']
    df = pd.DataFrame(candles)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['open'] = df['open'].astype(float)
    df['start'] = pd.to_datetime(df['start'], unit='s')
    return df

def get_sma_signals(df):
    df['sma_short'] = df['close'].rolling(SMA_SHORT).mean()
    df['sma_long'] = df['close'].rolling(SMA_LONG).mean()
    buy_signal = (df['sma_short'].iloc[-2] < df['sma_long'].iloc[-2]) and (df['sma_short'].iloc[-1] >= df['sma_long'].iloc[-1])
    sell_signal = (df['sma_short'].iloc[-2] > df['sma_long'].iloc[-2]) and (df['sma_short'].iloc[-1] <= df['sma_long'].iloc[-1])
    return buy_signal, sell_signal

def place_order(side, qty):
    order = session.place_order(
        category="linear",
        symbol=symbol,
        side="Buy" if side == "buy" else "Sell",
        order_type="Market",
        qty=qty,
        reduce_only=False,
    )
    return order

def close_order(position_side, qty):
    session.place_order(
        category="linear",
        symbol=symbol,
        side="Sell" if position_side == "Buy" else "Buy",
        order_type="Market",
        qty=qty,
        reduce_only=True,
    )

def main_loop(log_and_notify):
    position = None
    entry_price = None
    tp = None
    sl = None
    entry_time = None
    last_loss_time = None

    while True:
        try:
            df = get_ohlcv(symbol, 'D', SMA_LONG + 5)
            price = df['close'].iloc[-1]
            buy_signal, sell_signal = get_sma_signals(df)
            now = datetime.now()

            if last_loss_time and now < last_loss_time + timedelta(minutes=COOLDOWN_MINUTES):
                print('Waiting after loss...')
                time.sleep(60)
                continue

            if not position and buy_signal:
                place_order('buy', qty)
                position = 'buy'
                entry_price = price
                tp = entry_price * (1 + TP_PCT / 100)
                sl = entry_price * (1 - SL_PCT / 100)
                entry_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{now} Entry BUY at {entry_price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")

            elif not position and sell_signal:
                place_order('sell', qty)
                position = 'sell'
                entry_price = price
                tp = entry_price * (1 - TP_PCT / 100)
                sl = entry_price * (1 + SL_PCT / 100)
                entry_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{now} Entry SELL at {entry_price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")

            elif position:
                hit_tp = (price >= tp if position == 'buy' else price <= tp)
                hit_sl = (price <= sl if position == 'buy' else price >= sl)
                reverse_signal = False
                if position == 'buy':
                    _, sell_signal_now = get_sma_signals(df)
                    reverse_signal = sell_signal_now
                elif position == 'sell':
                    buy_signal_now, _ = get_sma_signals(df)
                    reverse_signal = buy_signal_now

                pnl = (price - entry_price) / entry_price * 100 if position == 'buy' else (entry_price - price) / entry_price * 100
                should_exit = hit_tp or hit_sl or reverse_signal

                if should_exit:
                    close_order('Buy' if position == 'buy' else 'Sell', qty)
                    exit_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    trade = {
                        'strategy': STRATEGY_NAME,
                        'side': position,
                        'position_size': qty,
                        'entry': entry_price,
                        'exit': price,
                        'pnl': round(pnl, 3),
                        'status': 'closed',
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'duration_minutes': '',  # auto-посчитается в logger
                        'tp': round(tp, 2),
                        'sl': round(sl, 2),
                        'fee': '',               # сюда можно добавить расчёт комиссии, если знаешь формулу
                        'comment': (
                            f"{'TP' if hit_tp else ('SL' if hit_sl else 'Reverse signal')}"
                        ),
                        'time': exit_time
                    }
                    print(f"{now} Exit {position.upper()} at {price:.2f} | PNL: {pnl:.3f}%")
                    log_and_notify(trade)
                    if hit_sl:
                        last_loss_time = now
                    position = None
                    entry_price = None
                    tp = None
                    sl = None
                    entry_time = None

            time.sleep(60 * 60 * 24)  # Проверять раз в день

        except Exception as e:
            print("Error:", e)
            time.sleep(60 * 60 * 6)

if __name__ == '__main__':
    main_loop(log_and_notify)
