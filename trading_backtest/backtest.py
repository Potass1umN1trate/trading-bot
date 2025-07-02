import numpy as np
import pandas as pd
import pandas_ta as ta
import logging

def ema(df, period, price_col='close'):
    return df[price_col].ewm(span=period).mean()

def backtest_strategy(
    df_4h, df_1d, deposit, risk_percent, stop_percent, take_percent,
    min_adx_day=20, min_adx_4h=20, trailing_percent=0, adaptive_tp=True
):
    # Setup logging to file
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="backtest.log",
        filemode="w"
    )
    logger = logging.getLogger("backtest")

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
                candle['open'] <= prev_candle['close'] and
                candle['close'] > prev_candle['open']
            )
            body = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            pinbar = lower_shadow >= 2 * body and lower_shadow > upper_shadow

            # if (candle['EMA50'] > candle['EMA200']) and \
            #    (candle['low'] <= candle['EMA21']) and \
            #    (candle['close'] >= candle['EMA21']) and \
            #    (candle['volume'] > candle['SMAvol9']) and \
            #    (bullish_engulfing or pinbar):
                
            
            if (candle['EMA50'] > candle['EMA200']) and \
               (candle['close'] >= candle['EMA21']) and \
               (candle['volume'] > candle['SMAvol9']) and \
               (bullish_engulfing or pinbar):

                entry = df_4h.iloc[i + 1]['open']
                stop = entry * (1 - stop_percent/100.0)
                take = entry * (1 + take_percent/100.0)
                pos_risk = deposit * (risk_percent/100.0)
                volume = pos_risk / abs(entry - stop)
                highest_price = entry
                trailing_stop = stop
                outcome = None
                adaptive_exit = None
                for j in range(i + 1, min(i + 30, len(df_4h) - 1)):
                    bar = df_4h.iloc[j]
                    # Update trailing stop
                    if trailing_percent is not None:
                        if bar['high'] > highest_price:
                            highest_price = bar['high']
                            new_trailing_stop = highest_price * (1 - trailing_percent/100.0)
                            if new_trailing_stop > trailing_stop:
                                trailing_stop = new_trailing_stop
                    # Check exit
                    if bar['low'] <= trailing_stop:
                        exit_price = trailing_stop
                        outcome = 'trailing stop' if trailing_stop > stop else 'stop'
                        profit = (exit_price - entry) * volume
                        profit_asset = (exit_price - entry) / entry * volume
                        break
                    if adaptive_tp:
                        # Bearish engulfing (выход из лонга)
                        bearish_engulfing = (
                            bar['close'] < bar['open'] and
                            df_4h.iloc[j-1]['close'] > df_4h.iloc[j-1]['open'] and
                            bar['open'] >= df_4h.iloc[j-1]['close'] and
                            bar['close'] < df_4h.iloc[j-1]['open']
                        )
                        if bearish_engulfing or bar['close'] < bar['EMA21']:
                            exit_price = bar['close']
                            outcome = 'adaptive_tp'
                            profit = (exit_price - entry) * volume
                            profit_asset = (exit_price - entry) / entry * volume
                            adaptive_exit = True
                            break
                    elif bar['high'] >= take:
                        exit_price = take
                        outcome = 'take'
                        profit = (exit_price - entry) * volume
                        profit_asset = (exit_price - entry) / entry * volume
                        break
                if outcome:
                    trade_info = {
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
                        'profit_asset': profit_asset,
                        'adx_4h': candle['ADX'],
                        'adx_1d': candle['ADX_1d'],
                        'pattern': 'bullish_engulfing' if bullish_engulfing else ('pinbar' if pinbar else ''),
                        'adaptive_exit': ("None" if not adaptive_exit else adaptive_exit)
                    }
                    logger.debug(f"Processed LONG trade: {trade_info}")
                    signals.append(trade_info)
            else:
                logger.debug({
                    'date': candle['startTime'],
                    'side': 'long',
                    'skipped': True,
                    'reason': 'Entry conditions not met',
                    'EMA50': candle['EMA50'],
                    'EMA200': candle['EMA200'],
                    'EMA21': candle['EMA21'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume'],
                    'SMAvol9': candle['SMAvol9'],
                    'bullish_engulfing': bullish_engulfing,
                    'pinbar': pinbar
                })
        else:
            logger.debug({
                'date': candle['startTime'],
                'side': 'long',
                'skipped': True,
                'reason': '1D trend/ADX conditions not met',
                'EMA50_1d': candle['EMA50_1d'],
                'EMA200_1d': candle['EMA200_1d'],
                'ADX_1d': candle['ADX_1d'],
                'ADX_4h': candle['ADX']
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
                prev_candle['close'] >= prev_candle['open'] and
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
                lowest_price = entry
                trailing_stop = stop
                adaptive_exit = None
                outcome = None
                for j in range(i + 1, min(i + 30, len(df_4h) - 1)):
                    bar = df_4h.iloc[j]
                    # Update trailing stop
                    if trailing_percent is not None:
                        if bar['low'] < lowest_price:
                            lowest_price = bar['low']
                            new_trailing_stop = lowest_price * (1 + trailing_percent/100.0)
                            if new_trailing_stop < trailing_stop:
                                trailing_stop = new_trailing_stop
                    # Check exit
                    if bar['high'] >= trailing_stop:
                        exit_price = trailing_stop
                        outcome = 'trailing stop' if trailing_stop > stop else 'stop'
                        profit = (entry - exit_price) * volume
                        profit_asset = (entry - exit_price) / entry * volume
                        break
                    if adaptive_tp:
                        # Bullish engulfing (выход из шорта)
                        bullish_engulfing = (
                            bar['close'] > bar['open'] and
                            df_4h.iloc[j-1]['close'] < df_4h.iloc[j-1]['open'] and
                            bar['open'] <= df_4h.iloc[j-1]['close'] and
                            bar['close'] > df_4h.iloc[j-1]['open']
                        )
                        if bullish_engulfing or bar['close'] > bar['EMA21']:
                            exit_price = bar['close']
                            outcome = 'adaptive_tp'
                            profit = (entry - exit_price) * volume
                            profit_asset = (entry - exit_price) / entry * volume
                            adaptive_exit = True
                            break
                    elif bar['low'] <= take:
                        exit_price = take
                        outcome = 'take'
                        profit = (entry - exit_price) * volume
                        profit_asset = (entry - exit_price) / entry * volume
                        break
                if outcome:
                    trade_info = {
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
                        'profit_asset': profit_asset,
                        'adx_4h': candle['ADX'],
                        'adx_1d': candle['ADX_1d'],
                        'pattern': 'bearish_engulfing' if bearish_engulfing else ('pinbar' if pinbar else ''),
                        'adaptive_exit': adaptive_exit
                    }
                    logger.debug(f"Processed SHORT trade: {trade_info}")
                    signals.append(trade_info)
            else:
                logger.debug({
                    'date': candle['startTime'],
                    'side': 'short',
                    'skipped': True,
                    'reason': 'Entry conditions not met',
                    'EMA50': candle['EMA50'],
                    'EMA200': candle['EMA200'],
                    'EMA21': candle['EMA21'],
                    'high': candle['high'],
                    'close': candle['close'],
                    'volume': candle['volume'],
                    'SMAvol9': candle['SMAvol9'],
                    'bearish_engulfing': bearish_engulfing,
                    'pinbar': pinbar
                })
        else:
            logger.debug({
                'date': candle['startTime'],
                'side': 'short',
                'skipped': True,
                'reason': '1D trend/ADX conditions not met',
                'EMA50_1d': candle['EMA50_1d'],
                'EMA200_1d': candle['EMA200_1d'],
                'ADX_1d': candle['ADX_1d'],
                'ADX_4h': candle['ADX']
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
