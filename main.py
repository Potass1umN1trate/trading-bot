#!/usr/bin/env python3

import os
import sys
import logging
import yaml
from enhanced_trading_bot import TradingBot

if __name__ == "__main__":
    # Check if config exists
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        if os.path.exists('config.yaml.example'):
            import shutil
            shutil.copy('config.yaml.example', config_path)
            logging.warning(f"Config not found. Created default {config_path} - please update with your credentials.")
        else:
            logging.error(f"Config file {config_path} not found.")
            sys.exit(1)
    
    # Read configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Make sure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run trading bot
    trading_bot = TradingBot(
        bybit_api_key=config['bybit']['api_key'],
        bybit_api_secret=config['bybit']['api_secret'],
        bybit_url=config['bybit']['url'],
        symbol=config['trading']['symbol'],
        category=config['trading']['category'],
        order_value=float(config['trading']['order_value']),
        upward_trend_threshold=float(config['trading']['upward_trend_threshold']),
        dip_threshold=float(config['trading']['dip_threshold']),
        profit_threshold=float(config['trading']['profit_threshold']),
        stop_loss_threshold=float(config['trading']['stop_loss_threshold']),
        initial_price=float(config['trading']['initial_price']),
        lookback_period=int(config['ai_model']['lookback_period']),
        prediction_horizon=int(config['ai_model']['prediction_horizon']),
        features=config['ai_model']['features'],
        interval_seconds=config['trading']['interval_seconds'],
        trading_interval=config['trading']['trading_interval'],
        training_data_limit=int(config['ai_model']['training_data_limit']),
        quantity_step=float(config['trading']['quantity_step']),  # Add quantity_step parameter
        risk_per_trade=float(config['trading']['risk_per_trade']),  # Add risk_per_trade parameter
        trailing_stop_loss=float(config['trading']['trailing_stop_loss'])  # Add trailing_stop_loss parameter
    )
    trading_bot.run()