import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os.path

class TradingBot:
    def __init__(
        self,
        bybit_api_key,
        bybit_api_secret,
        bybit_url,
        symbol,
        category,
        profit_threshold,
        stop_loss_threshold,
        lookback_period,
        prediction_horizon,
        features,  # Expecting raw features: ['open', 'high', 'low', 'close', 'volume']
        interval_seconds,
        training_data_limit,
        trading_interval,
        quantity_step,
        risk_per_trade=0.01,
        trailing_stop_loss=0.075
    ):
        # Initialize logger
        self.setup_logger()

        # Initialize the Bybit client
        self.client = HTTP(
            testnet=False,  # Set to True for testnet trading
            api_key=bybit_api_key,
            api_secret=bybit_api_secret,
            demo=True  # Set to False for live trading if testnet=False
        )
        self.logger.info(f"Using Bybit API URL: {self.client.endpoint}")

        # Trading parameters
        self.symbol = symbol
        self.category = category
        self.position_size = 0
        self.current_position = None

        # Load trading thresholds
        self.profit_threshold = profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.trailing_stop_loss = trailing_stop_loss

        # AI model parameters
        self.skipped_trades = 0
        self.max_skipped_trades = 3
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        # --- Use raw features ---
        self.features = features
        self.logger.info(f"Using features: {self.features}")
        self.model = None
        self.scaler = None

        self.interval_seconds = interval_seconds
        self.training_data_limit = training_data_limit
        self.trading_interval = trading_interval
        self.quantity_step = quantity_step
        self.risk_per_trade = risk_per_trade

        # Initialize or load AI model
        self.initialize_model()
        self.stop_price = None  # Stores TSL value

    def setup_logger(self):
        """Configure logging with timestamps and rotation"""
        self.logger = logging.getLogger('trading_bot')
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler('trading_bot.log')
            file_handler.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def initialize_model(self):
        """Initialize or load AI model"""
        model_path = 'trading_model_raw.joblib'
        scaler_path = 'scaler_raw.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.logger.info(f"Loading existing RAW AI model from {model_path} and scaler from {scaler_path}")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            self.logger.info("No existing raw model found. Training new RAW AI model.")
            self.train_model()

    def fetch_market_data(self, interval=None, limit=96):
        """Fetch historical kline data from Bybit"""
        if interval is None:
            interval = self.trading_interval
        try:
            response = self.client.get_kline(
                category=self.category,
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            if response['retCode'] != 0:
                self.logger.error(f"Error fetching kline data: {response['retMsg']}")
                return None

            klines = response['result']['list']
            if not klines:
                self.logger.warning("Fetched 0 klines from Bybit.")
                return None

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.logger.debug(f"Fetched {len(df)} klines from Bybit. Last timestamp: {df['timestamp'].iloc[-1]}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None

    def create_labels(self, df):
        """Create target labels for AI training based on future price movement"""
        df = df.copy()
        df['future_price'] = df['close'].shift(-self.prediction_horizon)
        df['price_direction'] = np.where(df['future_price'] > df['close'], 1, 0)
        df.dropna(subset=['future_price'], inplace=True)
        df.dropna(subset=self.features + ['price_direction'], inplace=True)
        self.logger.debug(f"Labels created. DataFrame shape after labeling and dropna: {df.shape}")
        return df

    def train_model(self):
        """Train the AI prediction model using RAW data"""
        model_path = 'trading_model_raw.joblib'
        scaler_path = 'scaler_raw.joblib'
        self.logger.info(f"Fetching data for initial RAW model training (limit={self.training_data_limit})...")
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.training_data_limit)
        if df is None or len(df) < self.lookback_period + self.prediction_horizon + 50:
            self.logger.error(f"Not enough data to train the RAW model. Fetched {len(df) if df is not None else 0} rows.")
            return False
        df = self.create_labels(df)
        if df is None or df.empty or len(df) < 50:
            self.logger.error("Failed to prepare sufficient training data after labeling.")
            return False
        self.logger.info(f"Preparing features: {self.features}")
        X = df[self.features]
        y = df['price_direction'].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info(f"Scaler fitted on {X_scaled.shape[0]} samples.")
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1,
            warm_start=True,
            random_state=42,
            learning_rate_init=0.001
        )
        self.logger.info("Starting initial partial_fit on MLPClassifier...")
        self.model.partial_fit(X_scaled, y, classes=np.array([0, 1]))
        self.logger.info("Initial partial_fit complete.")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"RAW Model saved to {model_path}, Scaler saved to {scaler_path}")
        return True

    def incremental_retrain(self, X_new, y_new):
        """Incrementally retrain the model with new RAW data"""
        model_path = 'trading_model_raw.joblib'
        if self.model is None or self.scaler is None:
            self.logger.error("Model or scaler not initialized for incremental retrain.")
            return
        y_new = np.asarray(y_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        unique_classes = np.unique(y_new)
        classes_to_fit = np.array([0, 1])
        if len(unique_classes) == 0:
            self.logger.warning("incremental_retrain called with empty y_new. Skipping.")
            return
        elif len(unique_classes) == 1:
            self.logger.warning(f"incremental_retrain called with only one class ({unique_classes[0]}) in y_new. Still fitting with classes [0, 1].")
        try:
            X_new_scaled = self.scaler.transform(X_new)
            self.model.partial_fit(X_new_scaled, y_new, classes=classes_to_fit)
            joblib.dump(self.model, model_path)
            self.logger.info(f"Incremental partial_fit completed. Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error during incremental retrain: {e}")

    def predict_price_direction(self):
        """Use the trained model to predict price direction using RAW data"""
        if self.model is None or self.scaler is None:
            self.logger.error("Model or Scaler not initialized for prediction.")
            return None
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.lookback_period)
        if df is None or df.empty:
            self.logger.warning("Failed to fetch data for prediction or data was empty.")
            return None
        if len(df) == 0:
            self.logger.warning("DataFrame is empty after fetching, cannot get latest data for prediction.")
            return None
        try:
            latest_data_row = df.iloc[-1]
            if latest_data_row[self.features].isnull().any():
                self.logger.warning(f"Latest data row contains NaN in features: {latest_data_row[self.features]}. Skipping prediction.")
                return None
            latest_data = latest_data_row[self.features].values.reshape(1, -1)
            scaled_data = self.scaler.transform(latest_data)
            prediction = self.model.predict(scaled_data)[0]
            probability = self.model.predict_proba(scaled_data)[0]
            prediction_confidence = probability[1] if prediction == 1 else probability[0]
            self.logger.info(f"RAW AI prediction: {'UP' if prediction == 1 else 'DOWN'} with {prediction_confidence:.2f} confidence")
            return {
                'direction': 'up' if prediction == 1 else 'down',
                'confidence': prediction_confidence,
                'timestamp': datetime.now().isoformat()
            }
        except IndexError:
            self.logger.error("IndexError: Could not access iloc[-1]. DataFrame might be smaller than expected.")
            return None
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return None

    def get_wallet_balance(self):
        """Get available balance from wallet"""
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if response['retCode'] != 0:
                self.logger.error(f"Error getting wallet balance: {response['retMsg']}")
                return None
            balance = None
            if response['result'] and response['result']['list']:
                unified_account = response['result']['list'][0]
                if unified_account['coin']:
                    for coin_info in unified_account['coin']:
                        if coin_info['coin'] == 'USDT':
                            balance = float(coin_info['walletBalance'])
                            break
            if balance is not None:
                self.logger.debug(f"Current wallet balance: {balance} USDT")
                return balance
            else:
                self.logger.error("Could not find USDT balance in the response.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting wallet balance: {str(e)}")
            return None

    def get_current_price(self):
        """Get the current price of the trading symbol"""
        try:
            response = self.client.get_tickers(category=self.category, symbol=self.symbol)
            if response['retCode'] != 0:
                self.logger.error(f"Error getting ticker: {response['retMsg']}")
                return None
            if response['result'] and response['result']['list']:
                price = float(response['result']['list'][0]['lastPrice'])
                self.logger.info(f"Current {self.symbol} price: {price}")
                return price
            else:
                self.logger.error(f"Ticker data not found in response for {self.symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return None

    def check_open_positions(self):
        """Check for any open positions"""
        try:
            response = self.client.get_positions(
                category=self.category,
                symbol=self.symbol
            )
            if response['retCode'] != 0:
                if "position not found" in response['retMsg']:
                    self.logger.info("No open positions found.")
                    self.current_position = None
                    return None
                self.logger.error(f"Error checking positions: {response['retMsg']}")
                return None
            positions = response['result']['list']
            if not positions or not positions[0] or float(positions[0]['size']) == 0:
                self.logger.info("No open positions.")
                self.current_position = None
                return None
            pos_data = positions[0]
            position = {
                'size': float(pos_data.get('size', 0)),
                'side': pos_data.get('side'),
                'entry_price': float(pos_data.get('avgPrice', 0)),
                'unrealized_pnl': float(pos_data.get('unrealisedPnl', 0)),
                'mark_price': float(pos_data.get('markPrice', 0))
            }
            if not position['side'] or position['entry_price'] == 0:
                self.logger.warning(f"Incomplete position data received: {pos_data}")
                self.current_position = None
                return None
            self.current_position = position
            self.logger.info(f"Current position: {position['size']} {self.symbol} {position['side']} at {position['entry_price']:.4f}, PnL: {position['unrealized_pnl']:.4f}")
            return position
        except Exception as e:
            self.logger.error(f"Error checking positions: {str(e)}")
            self.current_position = None
            return None

    def place_order(self, side, quantity, order_type="Market", price=None):
        """Place an order on Bybit"""
        if quantity <= 0:
            self.logger.warning(f"Attempted to place order with zero or negative quantity: {quantity}. Skipping.")
            return None
        try:
            order_params = {
                "category": self.category,
                "symbol": self.symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(quantity)
            }
            if order_type == "Limit":
                if price is None:
                    self.logger.error("Price must be provided for Limit orders.")
                    return None
                order_params["price"] = str(price)
            self.logger.info(f"Placing order with params: {order_params}")
            response = self.client.place_order(**order_params)
            if response['retCode'] != 0:
                self.logger.error(f"Order placement error: {response['retMsg']} (Code: {response['retCode']})")
                if 'result' in response and response['result']:
                    self.logger.error(f"Order Response Result: {response['result']}")
                return None
            order_id = response['result'].get('orderId', 'N/A')
            order_link_id = response['result'].get('orderLinkId', 'N/A')
            self.logger.info(f"Order placed successfully: {side} {quantity} {self.symbol} {order_type} at {price if price else 'market price'}. ID: {order_id}, LinkID: {order_link_id}")
            if (side == "Sell" and self.current_position and self.current_position['side'] == "Buy") or \
               (side == "Buy" and self.current_position and self.current_position['side'] == "Sell"):
                self.stop_price = None
                self.logger.info("Reset TSL price after closing position.")
            return order_id
        except Exception as e:
            self.logger.error(f"Exception during order placement: {str(e)}")
            return None

    def calculate_risk_based_quantity(self, current_price):
        """Calculate quantity based on risk percentage and fixed stop loss distance"""
        if current_price is None or current_price <= 0:
            self.logger.error("Invalid current_price for quantity calculation.")
            return None
        balance = self.get_wallet_balance()
        if balance is None or balance <= 0:
            self.logger.error("Invalid balance for quantity calculation.")
            return None
        risk_capital = balance * self.risk_per_trade
        if risk_capital <= 0:
            self.logger.warning(f"Calculated risk capital is zero or negative ({risk_capital:.4f}). Cannot calculate quantity.")
            return None
        self.logger.debug(f"Risk capital: {risk_capital:.4f} USDT")
        stop_distance_pct = self.stop_loss_threshold
        stop_distance = current_price * (stop_distance_pct / 100.0)
        if stop_distance <= 0:
            self.logger.error(f"Calculated stop distance is zero or negative ({stop_distance:.4f}) based on SL threshold {stop_distance_pct}%. Cannot calculate quantity.")
            return None
        self.logger.debug(f"Stop distance (price units): {stop_distance:.4f}")
        quantity = risk_capital / stop_distance
        self.logger.debug(f"Raw quantity calculated: {quantity:.8f}")
        if self.quantity_step <= 0:
            self.logger.error("Invalid quantity_step (must be > 0).")
            return None
        quantity_adjusted = (quantity // self.quantity_step) * self.quantity_step
        self.logger.info(f"Calculated risk-based quantity: {quantity_adjusted:.8f} {self.symbol[:-4]} (Step: {self.quantity_step})")
        if quantity_adjusted <= 0:
            self.logger.warning(f"Adjusted quantity is zero or less ({quantity_adjusted:.8f}) after applying step {self.quantity_step}. Cannot place trade.")
            return None
        return quantity_adjusted

    def update_trailing_stop_value(self, side, current_price, entry_price):
        """
        Update trailing stop loss based on the current price.
        Only updates if TSL is not set or if the new value improves the stop level.
        Requires the position to be profitable relative to entry price.
        """
        if current_price <= 0:
            return False

        is_profitable = (side == "Buy" and current_price > entry_price) or \
                        (side == "Sell" and current_price < entry_price)
        if not is_profitable:
            return False

        if side == "Buy":
            new_tsl = current_price * (1 - self.trailing_stop_loss / 100.0)
            if self.stop_price is None or new_tsl > self.stop_price:
                if new_tsl > entry_price:
                    old_tsl = self.stop_price
                    self.stop_price = new_tsl
                    self.logger.info(f"Updated TSL for LONG from {old_tsl} to: {self.stop_price:.4f} (Current: {current_price:.4f})")
                    return True
                else:
                    return False
        elif side == "Sell":
            new_tsl = current_price * (1 + self.trailing_stop_loss / 100.0)
            if self.stop_price is None or new_tsl < self.stop_price:
                if new_tsl < entry_price:
                    old_tsl = self.stop_price
                    self.stop_price = new_tsl
                    self.logger.info(f"Updated TSL for SHORT from {old_tsl} to: {self.stop_price:.4f} (Current: {current_price:.4f})")
                    return True
                else:
                    return False
        return False

    def execute_trade_strategy(self):
        """Execute trading logic based on RAW data AI predictions"""
        self.logger.info("--- Executing Trade Strategy Cycle ---")
        current_price = self.get_current_price()
        if current_price is None:
            self.logger.error("Could not get current price. Skipping cycle.")
            return False

        df = self.fetch_market_data(interval=self.trading_interval, limit=self.lookback_period + 5)
        if df is None or df.empty:
            self.logger.warning("Could not fetch sufficient recent data. Skipping cycle.")
            return False

        prediction = self.predict_price_direction()
        if prediction is None:
            self.logger.warning("Could not get AI prediction. Skipping cycle.")
            return False

        direction = prediction['direction']
        confidence = prediction['confidence']

        position = self.check_open_positions()

        if position:
            self.skipped_trades = 0
            entry_price = position['entry_price']
            side = position['side']
            position_size = position['size']
            price_change_pct = 0 if entry_price == 0 else ((current_price - entry_price) / entry_price) * 100
            self.logger.info(f"Managing {side} position. Entry: {entry_price:.4f}, Current: {current_price:.4f}, Change: {price_change_pct:.2f}%, TSL: {self.stop_price}")
            try:
                latest_features_raw = df.iloc[-1][self.features].values
            except (IndexError, KeyError) as e:
                self.logger.error(f"Failed to get latest raw features for retraining: {e}. Cannot close/retrain.")
                return False

            close_position_flag = False
            retrain_label = None

            if self.stop_price is not None:
                if side == 'Buy' and current_price <= self.stop_price:
                    self.logger.info(f"Trailing Stop Loss hit for LONG at {self.stop_price:.4f}. Closing position.")
                    close_position_flag = True
                    retrain_label = 1
                elif side == 'Sell' and current_price >= self.stop_price:
                    self.logger.info(f"Trailing Stop Loss hit for SHORT at {self.stop_price:.4f}. Closing position.")
                    close_position_flag = True
                    retrain_label = 0

            if not close_position_flag:
                if side == 'Buy' and price_change_pct >= self.profit_threshold:
                    self.logger.info(f"Take Profit threshold ({self.profit_threshold:.2f}%) hit for LONG. Closing position.")
                    close_position_flag = True
                    retrain_label = 1
                elif side == 'Sell' and price_change_pct <= -self.profit_threshold:
                    self.logger.info(f"Take Profit threshold ({self.profit_threshold:.2f}%) hit for SHORT. Closing position.")
                    close_position_flag = True
                    retrain_label = 0

            if not close_position_flag and self.stop_price is None:
                if side == 'Buy' and price_change_pct <= -self.stop_loss_threshold:
                    self.logger.info(f"Stop Loss threshold ({self.stop_loss_threshold:.2f}%) hit for LONG. Closing position.")
                    close_position_flag = True
                    retrain_label = 0
                elif side == 'Sell' and price_change_pct >= self.stop_loss_threshold:
                    self.logger.info(f"Stop Loss threshold ({self.stop_loss_threshold:.2f}%) hit for SHORT. Closing position.")
                    close_position_flag = True
                    retrain_label = 1

            if close_position_flag:
                close_side = "Sell" if side == "Buy" else "Buy"
                order_id = self.place_order(close_side, position_size)
                if order_id and retrain_label is not None:
                    self.logger.info(f"Position closed. Retraining model with label: {retrain_label}")
                    self.incremental_retrain(latest_features_raw, [retrain_label])
                else:
                    self.logger.error("Failed to place closing order or retrain label missing. Manual check needed.")
                self.current_position = None
                self.stop_price = None
                return True
            else:
                self.update_trailing_stop_value(side, current_price, entry_price)
        else:
            self.logger.info(f"No open position. AI Prediction: {direction.upper()} (Confidence: {confidence:.2f})")
            CONFIDENCE_THRESHOLD = 0.75
            open_trade_flag = False
            trade_side = None
            if direction == 'up' and confidence >= CONFIDENCE_THRESHOLD:
                self.logger.info(f"AI suggests LONG with sufficient confidence ({confidence:.2f}).")
                trade_side = "Buy"
                open_trade_flag = True
            elif direction == 'down' and confidence >= CONFIDENCE_THRESHOLD:
                self.logger.info(f"AI suggests SHORT with sufficient confidence ({confidence:.2f}).")
                trade_side = "Sell"
                open_trade_flag = True

            if open_trade_flag:
                quantity = self.calculate_risk_based_quantity(current_price)
                if quantity and quantity > 0:
                    self.logger.info(f"Calculated quantity: {quantity}. Placing {trade_side} order.")
                    order_id = self.place_order(trade_side, quantity)
                    if order_id:
                        self.skipped_trades = 0
                        self.stop_price = None
                        return True
                    else:
                        self.logger.error("Failed to place opening order.")
                else:
                    self.logger.warning("Calculated quantity was zero or invalid. Cannot open trade.")
            else:
                self.logger.info("AI prediction confidence below threshold or no signal. No trade.")
                self.skipped_trades += 1
                self.logger.info(f"Skipped trades count: {self.skipped_trades}")

            if self.skipped_trades >= self.max_skipped_trades:
                self.logger.warning(f"Reached max skipped trades ({self.max_skipped_trades}). Checking for penalty.")
                try:
                    start_index = max(0, len(df) - 1 - self.prediction_horizon)
                    price_then = df['close'].iloc[start_index]
                    price_now = df['close'].iloc[-1]
                    if price_then > 0:
                        actual_change_pct = ((price_now - price_then) / price_then) * 100
                        self.logger.info(f"Actual price change over ~{self.prediction_horizon} steps: {actual_change_pct:.2f}%")
                        punishment_threshold_pct = self.profit_threshold
                        if abs(actual_change_pct) >= punishment_threshold_pct:
                            self.logger.warning(f"Significant price move ({actual_change_pct:.2f}%) missed! Penalizing model for inaction.")
                            penalty_label = 1 if actual_change_pct > 0 else 0
                            latest_features_raw = df.iloc[-1][self.features].values
                            self.logger.info(f"Retraining inactive model with label: {penalty_label}")
                            self.incremental_retrain(latest_features_raw, [penalty_label])
                            self.skipped_trades = 0
                        else:
                            self.logger.info("No significant price movement during inactivity. No penalty.")
                    else:
                        self.logger.warning("Could not calculate actual price change due to zero start price.")
                except IndexError:
                    self.logger.warning("Not enough data points in df to calculate actual change for penalty.")
                except Exception as e:
                    self.logger.error(f"Error during penalty calculation: {e}")
        return True

    def run(self):
        """Run the trading bot in a loop"""
        self.logger.info(f"Starting RAW data trading bot for {self.symbol} on {self.trading_interval} min interval.")
        self.check_open_positions()
        self.get_wallet_balance()
        while True:
            start_time = time.time()
            try:
                success = self.execute_trade_strategy()
                if not success:
                    self.logger.warning("Trade strategy execution reported an issue. Continuing loop.")
            except KeyboardInterrupt:
                self.logger.info("Trading bot stopped by user (KeyboardInterrupt).")
                break
            except Exception as e:
                self.logger.error(f"!!! Critical Error in main loop: {str(e)}", exc_info=True)
            end_time = time.time()
            elapsed = end_time - start_time
            sleep_time = self.interval_seconds - elapsed
            if sleep_time < 0:
                self.logger.warning(f"Strategy execution ({elapsed:.2f}s) took longer than interval ({self.interval_seconds}s). Running next cycle immediately.")
                sleep_time = 0
            self.logger.info(f"--- Cycle End --- Sleeping for {sleep_time:.2f} seconds ---")
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    import os
    API_KEY = os.environ.get('BYBIT_API_KEY', 'your_api_key')
    API_SECRET = os.environ.get('BYBIT_API_SECRET', 'your_api_secret')
    if API_KEY == 'your_api_key' or API_SECRET == 'your_api_secret':
        print("WARNING: Using default API keys. Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables.")
    bot = TradingBot(
        bybit_api_key=API_KEY,
        bybit_api_secret=API_SECRET,
        bybit_url='https://api.bybit.com',
        symbol='BTCUSDT',
        category='linear',
        profit_threshold=0.8,       # Take profit %
        stop_loss_threshold=0.4,    # Stop loss %
        trailing_stop_loss=0.3,     # Trailing stop loss % activation/distance
        lookback_period=30,         # How much data to fetch for context (e.g., needed for 30 steps)
        prediction_horizon=1,       # Predict 1 interval step ahead
        features=['open', 'high', 'low', 'close', 'volume'],
        interval_seconds=60,        # Check every 60 seconds
        training_data_limit=2000,   # For initial training
        trading_interval='1',       # Use 1-minute klines
        quantity_step=0.001,        # Min BTC quantity step for BTCUSDT linear
        risk_per_trade=0.01         # Risk 1% of balance per trade
    )
    bot.run()