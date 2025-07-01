import pandas as pd

# Загрузить 4H и 1D данные
df_4h = pd.read_csv('BTCUSDT_4h.csv', parse_dates=['date']).sort_values('date')
df_1d = pd.read_csv('BTCUSDT_1d.csv', parse_dates=['date']).sort_values('date')

# EMA на 1D
df_1d['EMA50'] = df_1d['close'].ewm(span=50).mean()
df_1d['EMA200'] = df_1d['close'].ewm(span=200).mean()
# Допустим, тут же можно считать ADX, если нужно

# EMA на 4H
df_4h['EMA21'] = df_4h['close'].ewm(span=21).mean()
df_4h['EMA50'] = df_4h['close'].ewm(span=50).mean()
df_4h['EMA200'] = df_4h['close'].ewm(span=200).mean()
df_4h['SMAvol9'] = df_4h['volume'].rolling(9).mean()

# Присоединить к каждой 4H-свече значения EMA50/200 c последней прошедшей 1D-свечи
df_4h = pd.merge_asof(
    df_4h, 
    df_1d[['date', 'EMA50', 'EMA200']], 
    left_on='date', 
    right_on='date', 
    direction='backward', 
    suffixes=('', '_1d')
)

signals = []

for i in range(200, len(df_4h) - 2):
    # Глобальный фильтр: только лонг, если EMA50_1d > EMA200_1d
    if df_4h['EMA50_1d'].iloc[i] > df_4h['EMA200_1d'].iloc[i]:
        # Локальный тренд на 4H
        candle = df_4h.iloc[i]
        if (candle['EMA50'] > candle['EMA200']) and \
           (candle['low'] <= candle['EMA21']) and \
           (candle['close'] >= candle['EMA21']) and \
           (candle['volume'] > candle['SMAvol9']):
            # Вход на следующей свече
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
                    'date': df_4h.iloc[i+1]['date'],
                    'entry': entry,
                    'stop': stop,
                    'take': take,
                    'exit': exit_price,
                    'outcome': outcome
                })

results = pd.DataFrame(signals)
print(results['outcome'].value_counts())
print(f"Total trades: {len(results)}")
print(f"Win rate: {round(100*sum(results['outcome']=='take')/len(results),1)}%")
print(results.tail())
