import numpy as np
import pandas as pd
import pandas_ta as ta

def ema(df, period, price_col='close'):
    return df[price_col].ewm(span=period).mean()

def backtest_strategy(
    df_4h, df_1d, deposit, risk_percent, stop_percent, take_percent,
    min_adx_day=20, min_adx_4h=20
):
    import pandas_ta as ta

    # EMA и ADX на 1D
    df_1d['EMA50'] = ema(df_1d, 50)
    df_1d['EMA200'] = ema(df_1d, 200)
    df_1d['ADX'] = ta.adx(df_1d['high'], df_1d['low'], df_1d['close'], length=14)['ADX_14']

    # EMA и ADX на 4H
    df_4h['EMA21'] = ema(df_4h, 21)
    df_4h['EMA50'] = ema(df_4h, 50)
    df_4h['EMA200'] = ema(df_4h, 200)
    df_4h['SMAvol9'] = df_4h['volume'].rolling(9).mean()
    df_4h['ADX'] = ta.adx(df_4h['high'], df_4h['low'], df_4h['close'], length=14)['ADX_14']

    df_4h = pd.merge_asof(
        df_4h.sort_values('startTime'),
        df_1d[['startTime', 'EMA50', 'EMA200', 'ADX']].sort_values('startTime'),
        left_on='startTime', right_on='startTime',
        direction='backward', suffixes=('', '_1d')
    )

    signals = []
    for i in range(201, len(df_4h) - 2):
        candle = df_4h.iloc[i]
        prev_candle = df_4h.iloc[i-1]

        # ==== LONG ====
        if (
            candle['EMA50_1d'] > candle['EMA200_1d']
            and candle['ADX_1d'] >= min_adx_day
            and candle['ADX'] >= min_adx_4h
        ):
            bullish_engulfing = (
                candle['close'] > candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                candle['open'] < prev_candle['close'] and
                candle['close'] > prev_candle['open']
            )
            body = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            pinbar = lower_shadow >= 2 * body and lower_shadow > upper_shadow

            if (candle['EMA50'] > candle['EMA200']) and \
               (candle['low'] <= candle['EMA21']) and \
               (candle['close'] >= candle['EMA21']) and \
               (candle['volume'] > candle['SMAvol9']) and \
               (bullish_engulfing or pinbar):

                entry = df_4h.iloc[i + 1]['open']
                stop = entry * (1 - stop_percent/100.0)
                take = entry * (1 + take_percent/100.0)
                pos_risk = deposit * (risk_percent/100.0)
                volume = pos_risk / abs(entry - stop)
                outcome = None
                for j in range(i + 1, min(i + 30, len(df_4h) - 1)):
                    bar = df_4h.iloc[j]
                    if bar['low'] <= stop:
                        exit_price = stop
                        outcome = 'stop'
                        profit = (exit_price - entry) * volume
                        profit_asset = (exit_price - entry) / entry * volume
                        break
                    if bar['high'] >= take:
                        exit_price = take
                        outcome = 'take'
                        profit = (exit_price - entry) * volume
                        profit_asset = (exit_price - entry) / entry * volume
                        break
                if outcome:
                    signals.append({
                        'date': df_4h.iloc[i+1]['startTime'],
                        'side': 'long',
                        'entry': entry,
                        'stop': stop,
                        'take': take,
                        'exit': exit_price,
                        'outcome': outcome,
                        'risk_usd': pos_risk,
                        'volume': volume,
                        'profit_usd': profit,
                        'profit_asset': profit,
                        'adx_4h': candle['ADX'],
                        'adx_1d': candle['ADX_1d'],
                        'pattern': 'bullish_engulfing' if bullish_engulfing else ('pinbar' if pinbar else '')
                    })

        # ==== SHORT ====
        if (
            candle['EMA50_1d'] < candle['EMA200_1d']
            and candle['ADX_1d'] >= min_adx_day
            and candle['ADX'] >= min_adx_4h
        ):
            # Медвежье поглощение
            bearish_engulfing = (
                candle['close'] < candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                candle['open'] > prev_candle['close'] and
                candle['close'] < prev_candle['open']
            )
            # Пин-бар с длинным ВЕРХНИМ хвостом
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            pinbar = upper_shadow >= 2 * body and upper_shadow > lower_shadow

            if (candle['EMA50'] < candle['EMA200']) and \
               (candle['high'] >= candle['EMA21']) and \
               (candle['close'] <= candle['EMA21']) and \
               (candle['volume'] > candle['SMAvol9']) and \
               (bearish_engulfing or pinbar):

                entry = df_4h.iloc[i + 1]['open']
                stop = entry * (1 + stop_percent/100.0)
                take = entry * (1 - take_percent/100.0)
                pos_risk = deposit * (risk_percent/100.0)
                volume = pos_risk / abs(entry - stop)
                outcome = None
                for j in range(i + 1, min(i + 30, len(df_4h) - 1)):
                    bar = df_4h.iloc[j]
                    if bar['high'] >= stop:
                        exit_price = stop
                        outcome = 'stop'
                        profit = (entry - exit_price) * volume
                        profit_asset = (entry - exit_price) / entry * volume
                        break
                    if bar['low'] <= take:
                        exit_price = take
                        outcome = 'take'
                        profit = (entry - exit_price) * volume
                        profit_asset = (entry - exit_price) / entry * volume
                        break
                if outcome:
                    signals.append({
                        'date': df_4h.iloc[i+1]['startTime'],
                        'side': 'short',
                        'entry': entry,
                        'stop': stop,
                        'take': take,
                        'exit': exit_price,
                        'outcome': outcome,
                        'risk_usd': pos_risk,
                        'volume': volume,
                        'profit_usd': profit,
                        'profit_asset': profit,
                        'adx_4h': candle['ADX'],
                        'adx_1d': candle['ADX_1d'],
                        'pattern': 'bearish_engulfing' if bearish_engulfing else ('pinbar' if pinbar else '')
                    })
    results = pd.DataFrame(signals)
    total_trades = len(results)
    winrate = round(100*sum(results['outcome']=='take')/total_trades,1) if total_trades else 0
    total_profit_usd = results['profit_usd'].sum() if not results.empty else 0
    total_profit_asset = results['profit_asset'].sum() if not results.empty else 0
    return {
        'total_trades': total_trades,
        'winrate': winrate,
        'results': results,
        'total_profit_usd': total_profit_usd,
        'total_profit_asset': total_profit_asset
    }
