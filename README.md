# Trading Bot

This is a trading bot that uses AI to predict price movements and execute trades on Bybit.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/trading-bot.git
    cd trading-bot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Configure your API keys and trading parameters:
    - Copy `config.yaml.example` to `config.yaml` and update the values:
        ```sh
        cp config.yaml.example config.yaml
        ```

## Running the Bot

To start the trading bot, run:
```sh
python main.py
```

## Configuration

The bot uses a `config.yaml` file for configuration. Here is an example:

```yaml
bybit:
  url: "https://api-demo.bybit.com"
  api_key: your_api_key
  api_secret: your_api_secret

trading:
  symbol: "BTCUSDT"
  category: "linear"
  order_value: 100.00
  upward_trend_threshold: 1.50
  dip_threshold: -2.25
  profit_threshold: 1.25
  stop_loss_threshold: -2.00
  initial_price: 100.00
  trading_interval: "60"

ai_model:
  lookback_period: 24
  prediction_horizon: 1
  features: ['close', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
  training_data_limit: 2000
```

## .gitignore

The `.gitignore` file includes the following entries to avoid committing sensitive information and unnecessary files:

```ignore
.gitignore
config.yaml
.venv
*.log
```

## License

This project is licensed under the MIT License.
