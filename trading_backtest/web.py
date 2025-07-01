from flask import Flask, request, render_template_string, send_file, session
import io
from datetime import datetime
import traceback

from bybit_data import get_klines
from backtest import backtest_strategy

app = Flask(__name__)
app.secret_key = 'supersecret'

HTML_FORM = '''
<!doctype html>
<title>Backtest Bybit EMA-Strategy</title>
<h2>Run Backtest with Bybit Data</h2>
<form method=post>
  Symbol: <input type=text name=symbol value="BTCUSDT"><br>
  Category: <select name=category>
    <option value="linear">linear</option>
    <option value="spot">spot</option>
    <option value="inverse">inverse</option>
  </select><br>
  Start Date (YYYY-MM-DD): <input type=text name=start value="2023-01-01"><br>
  End Date (YYYY-MM-DD): <input type=text name=end value="2023-07-01"><br>
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
  <b>Total profit (in asset):</b> {{ result.total_profit }}<br>
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
    if request.method == 'POST':
        try:
            symbol = request.form.get('symbol', 'BTCUSDT').upper()
            category = request.form.get('category', 'linear')
            start = datetime.strptime(request.form.get('start', '2023-01-01'), '%Y-%m-%d')
            end = datetime.strptime(request.form.get('end', '2023-07-01'), '%Y-%m-%d')

            df_1d = get_klines(category, symbol, "D", start, end)
            df_4h = get_klines(category, symbol, "240", start, end)
            bt = backtest_strategy(df_4h, df_1d)
            # Save DataFrame to session for download
            session['trades_csv'] = bt['results'].to_csv(index=False)
            result = {
                'total_trades': bt['total_trades'],
                'winrate': bt['winrate'],
                'total_profit': round(bt['total_profit'], 2),
                'trades': bt['results'].to_html(index=False)
            }
        except Exception as e:
            import traceback
            error = traceback.format_exc()
    return render_template_string(HTML_FORM, result=result, error=error)

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
