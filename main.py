#!/usr/bin/env python3

import os
import sys
import logging
import yaml
import shutil  # Added for copying config example
from enhanced_trading_bot import TradingBot  # Assuming your modified bot class is here

# Basic logging setup for the main script itself
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Check if config exists
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        if os.path.exists('config.yaml.example'):
            try:
                shutil.copy('config.yaml.example', config_path)
                logging.warning(f"Config not found. Copied default from config.yaml.example to {config_path}.")
                logging.warning("IMPORTANT: Please update config.yaml with your API credentials and ensure ai_model.features lists raw columns like ['open', 'high', 'low', 'close', 'volume'].")
                # Optional: exit here to force user configuration
                # sys.exit("Please configure config.yaml before running.")
            except Exception as e:
                logging.error(f"Could not copy config.yaml.example: {e}")
                sys.exit(1)
        else:
            logging.error(f"Config file {config_path} not found, and no config.yaml.example exists.")
            sys.exit(1)

    # Read configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error reading config file {config_path}: {e}")
        sys.exit(1)

    # --- IMPORTANT ---
    # Verify that the config reflects the use of raw features.
    # The bot now expects features like ['open', 'high', 'low', 'close', 'volume']
    # Ensure your config.yaml's ai_model.features section is updated accordingly!
    logging.info(f"Reading features from config: {config.get('ai_model', {}).get('features', 'Not Set')}")
    logging.warning("Ensure the features listed above are raw OHLCV columns for the modified bot.")
    # ---

    # Make sure models directory exists (Optional; bot currently saves to root)
    # os.makedirs('models', exist_ok=True)

    try:
        # Initialize and run trading bot
        trading_bot = TradingBot(
            bybit_api_key=config['bybit']['api_key'],
            bybit_api_secret=config['bybit']['api_secret'],
            bybit_url=config['bybit']['url'],
            symbol=config['trading']['symbol'],
            category=config['trading']['category'],
            # order_value is no longer used
            # upward_trend_threshold is removed
            # dip_threshold is removed
            profit_threshold=float(config['trading']['profit_threshold']),
            stop_loss_threshold=float(config['trading']['stop_loss_threshold']),
            # initial_price is fetched live; no longer provided
            lookback_period=int(config['ai_model']['lookback_period']),
            prediction_horizon=int(config['ai_model']['prediction_horizon']),
            # --- Ensure config['ai_model']['features'] lists raw features ---
            features=config['ai_model']['features'],
            interval_seconds=int(config['trading']['interval_seconds']),
            trading_interval=str(config['trading']['trading_interval']),
            training_data_limit=int(config['ai_model']['training_data_limit']),
            quantity_step=float(config['trading']['quantity_step']),
            risk_per_trade=float(config['trading']['risk_per_trade']),
            trailing_stop_loss=float(config['trading']['trailing_stop_loss']),
            retrain_buffer_size=int(config['ai_model']['retrain_buffer_size'])  # Add this line
        )
        trading_bot.run()

    except KeyError as e:
        logging.error(f"Missing configuration key: {e}. Please check your {config_path}.")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid value type in configuration: {e}. Please check your {config_path}.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during bot initialization or run: {e}", exc_info=True)
        sys.exit(1)