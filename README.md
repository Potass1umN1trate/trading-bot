
# Trading Bot

This project is an AI-powered trading bot that uses machine learning to predict market trends and execute trades on the Bybit exchange.

## Features

- Fetches historical market data from Bybit
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trains a RandomForestClassifier model to predict price direction
- Executes trades based on AI predictions and predefined thresholds
- Logs trading activities and errors

## Requirements

- Python 3.7+
- Bybit API credentials
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/trading-bot.git
    cd trading-bot
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a `config.ini` file with your Bybit API credentials and trading parameters:
    ```ini
    [bybit]
    api_key = your_api_key
    api_secret = your_api_secret
    url = https://api.bybit.com

    [trading]
    upward_trend_threshold = 1.0
    dip_threshold = -1.0
    profit_threshold = 2.0
    stop_loss_threshold = -1.0
    initial_price = 50000.0
    ```

## Usage

1. Run the trading bot:
    ```sh
    python main.py
    ```

2. The bot will start fetching market data, making predictions, and executing trades based on the AI model and predefined thresholds.

## Logging

The bot logs its activities to `trading_bot.log`. You can monitor this file to see the bot's actions and any errors that occur.

## Disclaimer

This trading bot is for educational purposes only. Trading cryptocurrencies involves significant risk, and you should only trade with money you can afford to lose. The authors are not responsible for any financial losses incurred while using this bot.

## License

This project is licensed under the MIT License.
