import pandas as pd
import numpy as np

def ema(df, period, price_col='close'):
    return df[price_col].ewm(span=period).mean()

def backtest_strategy(df_4h: pd.DataFrame, df_1d: pd.DataFrame):
    # EMA на 1D
    df_1d['EMA50'] = ema(df_1d, 50)
    df_1d['EMA200'] = ema(df_1d, 200)

    # EMA на 4H
    df_4h['EMA21'] = ema(df_4h, 21)
    df_4h['EMA50'] = ema(df_4h, 50)
    df_4h['EMA200'] = ema(df_4h, 200)
    df_4h['SMAvol9'] = df_4h['volume'].rolling(9).mean()

    # Присоединить к каждой 4H-свече значения EMA50/200 с 1D
    df_4h = pd.merge_asof(
        df_4h.sort_values('startTime'), 
        df_1d[['startTime', 'EMA50', 'EMA200']].sort_values('startTime'), 
        left_on='startTime', right_on='startTime', 
        direction='backward', suffixes=('', '_1d')
    )

    signals = []
    for i in range(200, len(df_4h) - 2):
        if df_4h['EMA50_1d'].iloc[i] > df_4h['EMA200_1d'].iloc[i]:
            candle = df_4h.iloc[i]
            if (candle['EMA50'] > candle['EMA200']) and \
               (candle['low'] <= candle['EMA21']) and \
               (candle['close'] >= candle['EMA21']) and \
               (candle['volume'] > candle['SMAvol9']):
                entry = df_4h.iloc[i + 1]['open']
                stop = min(candle['low'], candle['EMA21']) * 0.995
                take = entry + 2 * (entry - stop)
                outcome = None
                for j in range(i + 1, min(i + 30, len(df_4h) - 1)):
                    bar = df_4h.iloc[j]
                    if bar['low'] <= stop:
                        exit_price = stop
                        outcome = 'stop'
                        break
                    if bar['high'] >= take:
                        exit_price = take
                        outcome = 'take'
                        break
                if outcome:
                    signals.append({
                        'date': df_4h.iloc[i+1]['startTime'],
                        'entry': entry,
                        'stop': stop,
                        'take': take,
                        'exit': exit_price,
                        'outcome': outcome
                    })
    results = pd.DataFrame(signals)
    total_trades = len(results)
    winrate = round(100*sum(results['outcome']=='take')/total_trades,1) if total_trades else 0

    # Calculate profit: (Take = +1R, Stop = -1R) — you can scale R to $, %, or anything you want.
    if not results.empty:
        results['profit'] = np.where(results['outcome']=='take', results['take']-results['entry'], results['exit']-results['entry'])
        total_profit = results['profit'].sum()
    else:
        total_profit = 0

    return {
        'total_trades': total_trades,
        'winrate': winrate,
        'results': results,
        'total_profit': total_profit
    }
