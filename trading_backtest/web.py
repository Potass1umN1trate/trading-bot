from flask import Flask, request, render_template_string, send_file, session
from datetime import datetime
import traceback
from bybit_data import get_klines
from backtest import backtest_strategy
import io

app = Flask(__name__)
app.secret_key = 'supersecret'

HTML_FORM = '''
<!doctype html>
<title>Backtest Bybit EMA-Strategy</title>
<h2>Run Backtest with Bybit Data</h2>
<form method=post>
  Symbol: <input type=text name=symbol value="{{ symbol }}"><br>
  Category: <select name=category>
    <option value="linear" {% if category == 'linear' %}selected{% endif %}>linear</option>
    <option value="spot" {% if category == 'spot' %}selected{% endif %}>spot</option>
    <option value="inverse" {% if category == 'inverse' %}selected{% endif %}>inverse</option>
  </select><br>
  Start Date (YYYY-MM-DD): <input type=text name=start value="{{ start }}"><br>
  End Date (YYYY-MM-DD): <input type=text name=end value="{{ end }}"><br>
  Deposit: <input type="number" name="deposit" step="any" value="{{ deposit }}"><br>
  Risk percent: <input type="number" name="risk" step="any" value="{{ risk }}"><br>
  Stop percent: <input type="number" name="stop_percent" step="any" value="{{ stop_percent }}"><br>
  Take percent: <input type="number" name="take_percent" step="any" value="{{ take_percent }}"><br>
  <b>Trailing stop (%)</b>: <input type="number" name="trailing_percent" step="any" value="{{ trailing_percent }}"><br>
  <b>Min ADX (1D)</b>: <input type="number" name="min_adx_day" step="any" value="{{ min_adx_day }}"><br>
  <b>Min ADX (4H)</b>: <input type="number" name="min_adx_4h" step="any" value="{{ min_adx_4h }}"><br>
  <label>
      <input type="checkbox" name="adaptive_tp" {% if adaptive_tp %}checked{% endif %}>
      Adaptive Take Profit
  </label><br>
  <input type=submit value="Run Backtest">
</form>
<hr>
{% if error %}
  <h3 style="color:red;">ERROR:</h3>
  <pre>{{ error }}</pre>
  <hr>
{% endif %}
{% if result %}
  <h3>Results:</h3>
  <b>Total trades:</b> {{ result.total_trades }}<br>
  <b>Win rate:</b> {{ result.winrate }}%<br>
  <b>Total profit ($):</b> {{ result.total_profit_usd }}<br>
  <b>Total profit (asset):</b> {{ result.total_profit_asset }}<br>
  <a href="/download_csv" target="_blank">Download All Trades (CSV)</a>
  <hr>
  <b>All trades:</b><br>
  {{ result.trades|safe }}
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    # Default values
    symbol = 'BTCUSDT'
    category = 'linear'
    start = '2023-01-01'
    end = '2023-07-01'
    deposit = 1000
    risk = 1
    stop_percent = 5
    take_percent = 10
    trailing_percent = 0
    min_adx_day = 17
    min_adx_4h = 20
    adaptive_tp = False

    if request.method == 'POST':
        try:
            symbol = request.form.get('symbol', symbol).upper()
            category = request.form.get('category', category)
            start = request.form.get('start', start)
            end = request.form.get('end', end)
            deposit = float(request.form.get('deposit', deposit))
            risk = float(request.form.get('risk', risk))
            stop_percent = float(request.form.get('stop_percent', stop_percent))
            take_percent = float(request.form.get('take_percent', take_percent))
            trailing_percent = request.form.get("trailing_percent")
            trailing_percent = float(trailing_percent) if trailing_percent not in [None, ""] else None
            min_adx_day = float(request.form.get("min_adx_day", min_adx_day))
            min_adx_4h = float(request.form.get("min_adx_4h", min_adx_4h))
            adaptive_tp = request.form.get("adaptive_tp") == 'on'

            df_1d = get_klines(category, symbol, "D", datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d'))
            df_4h = get_klines(category, symbol, "240", datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d'))
            bt = backtest_strategy(df_4h, df_1d, deposit, risk, stop_percent, take_percent, 
                                  min_adx_day=min_adx_day, min_adx_4h=min_adx_4h, trailing_percent=trailing_percent, adaptive_tp=adaptive_tp)
            session['trades_csv'] = bt['results'].to_csv(index=False)
            result = {
                'total_trades': bt['total_trades'],
                'winrate': bt['winrate'],
                'total_profit_usd': round(bt['total_profit_usd'], 2),
                'total_profit_asset': round(bt['total_profit_asset'], 6),
                'trades': bt['results'].to_html(index=False)
            }
        except Exception:
            error = traceback.format_exc()
    else:
        # GET-запрос — сохранить дефолтные или последние введённые значения
        pass

    return render_template_string(HTML_FORM,
                                 result=result,
                                 error=error,
                                 symbol=symbol,
                                 category=category,
                                 start=start,
                                 end=end,
                                 deposit=deposit,
                                 risk=risk,
                                 stop_percent=stop_percent,
                                 take_percent=take_percent,
                                 trailing_percent=trailing_percent,
                                 min_adx_day=min_adx_day,
                                 min_adx_4h=min_adx_4h,
                                 adaptive_tp=adaptive_tp
                                 )

@app.route('/download_csv')
def download_csv():
    csv = session.get('trades_csv', None)
    if not csv:
        return "No data available."
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='trades.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
