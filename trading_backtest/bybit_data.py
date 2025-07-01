from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta
import time

session = HTTP(testnet=False)

def ms(ts):
    "Datetime → миллисекунды"
    return int(time.mktime(ts.timetuple()) * 1000)

def interval_to_minutes(interval):
    # Bybit "interval" → минут
    mapping = {
        '1': 1, '3': 3, '5': 5, '15': 15, '30': 30, '60': 60,
        '120': 120, '240': 240, '360': 360, '720': 720,
        'D': 1440,
        'W': 10080,
        'M': 43200
    }
    if str(interval) in mapping:
        return mapping[str(interval)]
    else:
        raise ValueError(f"Unknown interval: {interval}")

def get_klines(category, symbol, interval, start: datetime, end: datetime, limit=1000):
    klines = []
    current = start
    minutes_per_candle = interval_to_minutes(interval)
    while current < end:
        next_ts = min(current + timedelta(minutes=minutes_per_candle * limit), end)
        resp = session.get_kline(
            category=category,
            symbol=symbol,
            interval=str(interval),
            start=ms(current),
            end=ms(next_ts),
            limit=limit
        )
        if resp['retCode'] != 0:
            print("API Error:", resp)
            break
        data = resp['result']['list']
        if not data:
            break
        klines += data
        # API отдаёт свечи в обратном порядке
        current = datetime.fromtimestamp(int(data[0][0]) // 1000) + timedelta(minutes=minutes_per_candle)
        if len(data) < limit:
            break
    df = pd.DataFrame(klines, columns=[
        "startTime", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["startTime"] = pd.to_datetime(df["startTime"].astype('int64'), unit='ms')
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values("startTime").reset_index(drop=True)
    return df
