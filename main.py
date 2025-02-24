import logging
import configparser
from pybit.unified_trading import HTTP

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize the Bybit client
client = HTTP(testnet=False,
              api_key=config['bybit']['api_key'],
              api_secret=config['bybit']['api_secret'],
              demo=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Using Bybit API URL: https://api.bybit.com")

# Constants
UPWARD_TREND_THRESHOLD = float(config['trading']['upward_trend_threshold'])
DIP_THRESHOLD = float(config['trading']['dip_threshold'])
PROFIT_THRESHOLD = float(config['trading']['profit_threshold'])
STOP_LOSS_THRESHOLD = float(config['trading']['stop_loss_threshold'])

# Variables
lastOpPrice = float(config['trading']['initial_price'])

# Get the orderbook of the USDT Perpetual, BTCUSDT
client.get_orderbook(category="linear", symbol="BTCUSDT")

# Submit the orders in bulk.
print(client.place_order(
    category="spot",
    symbol="BTCUSDT",
    side="Buy",
    orderType="Limit",
    qty="0.1",
    price="15600",
    timeInForce="PostOnly",
    orderLinkId="spot-test-postonly",
    isLeverage=0,
    orderFilter="Order"
))
