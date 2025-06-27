# logger_module.py

import csv
import asyncio
from aiogram import Bot
import yaml
from datetime import datetime

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
CSV_FILE = 'trades.csv'
TELEGRAM_TOKEN = config['telegram']['token']
TELEGRAM_CHAT_ID = config['telegram']['chat_id']

def log_to_csv(trade):
    header = [
        'strategy',
        'side',
        'position_size',
        'entry',
        'exit',
        'pnl',
        'status',
        'entry_time',
        'exit_time',
        'duration_minutes',
        'tp',
        'sl',
        'fee',
        'time',        # дублируем exit_time для обратной совместимости
        'comment'
    ]
    # добавим автозаполнение duration_minutes если не задано
    if 'entry_time' in trade and 'exit_time' in trade and not trade.get('duration_minutes'):
        try:
            t1 = datetime.strptime(trade['entry_time'], "%Y-%m-%d %H:%M:%S")
            t2 = datetime.strptime(trade['exit_time'], "%Y-%m-%d %H:%M:%S")
            trade['duration_minutes'] = round((t2 - t1).total_seconds() / 60, 2)
        except Exception:
            trade['duration_minutes'] = ''

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({k: trade.get(k, '') for k in header})

async def send_to_telegram(trade):
    bot = Bot(token=TELEGRAM_TOKEN)
    msg = (
        f"BTCUSDT {trade.get('side','').upper()} {trade.get('strategy','')}\n"
        f"Entry: {trade.get('entry')}\n"
        f"Exit: {trade.get('exit')}\n"
        f"PNL: {trade.get('pnl',0):.3f}%\n"
        f"Duration: {trade.get('duration_minutes','?')} min\n"
        f"Entry Time: {trade.get('entry_time','')}\n"
        f"Exit Time: {trade.get('exit_time','')}\n"
        f"Status: {trade.get('status','')}\n"
        f"TP: {trade.get('tp','')}\n"
        f"SL: {trade.get('sl','')}\n"
        f"Comment: {trade.get('comment','')}"
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    await bot.session.close()

def log_and_notify(trade):
    log_to_csv(trade)
    asyncio.run(send_to_telegram(trade))

if __name__ == '__main__':
    # Пример
    now = datetime.now().replace(second=0, microsecond=0)
    test_trade = {
        'strategy': 'SMA50/200',
        'side': 'buy',
        'position_size': 0.01,
        'entry': 65000,
        'exit': 65325,
        'pnl': 0.5,
        'status': 'closed',
        'entry_time': (now - pd.Timedelta(minutes=1400)).strftime("%Y-%m-%d %H:%M:%S"),
        'exit_time': now.strftime("%Y-%m-%d %H:%M:%S"),
        'tp': 65450,
        'sl': 64700,
        'fee': 2.5,
        'comment': 'Test signal',
        'time': now.strftime("%Y-%m-%d %H:%M:%S")   # для обратной совместимости
    }
    log_and_notify(test_trade)
