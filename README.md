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

## Docker

To build and run the trading bot using Docker Compose:

1. Build the Docker image:
    ```sh
    sudo docker compose build
    ```

2. Run the Docker container:
    ```sh
    sudo docker compose up -d
    ```

3. To stop the Docker container:
    ```sh
    sudo docker compose down
    ```

4. To modify the application, make changes to the code and then rebuild and restart the container:
    ```sh
    sudo docker compose build
    sudo docker compose up -d
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
  quantity_step: 0.001  # Add quantity_step parameter
  risk_per_trade: 0.01  # Add risk_per_trade parameter

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
__pycache__
config.yaml
.venv
*.log
TODO.txt
*.joblib
```

## License

This project is licensed under the MIT License.
