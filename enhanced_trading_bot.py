import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
import joblib
import os.path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Use the last time step
        return self.fc(out)

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
        trailing_stop_loss=0.075,
        retrain_buffer_size=100, # New parameter for buffer size
        hidden_size=64,
        num_layers=2,
        dropout_prob=0.2,
        learning_rate=0.001,
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

        # --- Retraining Buffer ---
        self.buffer_size = retrain_buffer_size
        self.retrain_buffer = deque(maxlen=self.buffer_size)
        self.logger.info(f"Initialized retraining buffer with max size: {self.buffer_size}")
        # ---

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.num_features_per_step = len(self.features)
        self.model_path = 'trading_model_lstm.pth'
        # Initialize LSTM model
        self.model = LSTMModel(
            input_size=self.num_features_per_step,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout_prob=self.dropout_prob
        ).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize or load AI model
        self.initialize_model()
        self.stop_price = None  # Stores TSL value

    def setup_logger(self):
        """Configure logging with timestamps and rotation"""
        self.logger = logging.getLogger('trading_bot')
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler('trading_bot.log')
            file_handler.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.debug("Logger setup complete.")
        self.logger.info("Logger setup complete.")
        self.logger.warning("Logger setup complete.")
        self.logger.error("Logger setup complete.")

    def initialize_model(self):
        """Initialize or load AI model"""
        model_path = 'trading_model_lstm.pth'
        scaler_path = 'scaler_raw.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.logger.info(f"Loading existing LSTM model from {model_path} and scaler from {scaler_path}")
            self.model.load_state_dict(torch.load(model_path))
            self.scaler = joblib.load(scaler_path)
        else:
            self.logger.info("No existing model found. Training new LSTM model.")
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

    def create_sequence_features_and_labels(self, df):
        """Prepare sequence data for LSTM."""
        df['future_price'] = df['close'].shift(-self.prediction_horizon)
        df['price_direction'] = np.where(df['future_price'] > df['close'], 1, 0)
        df.dropna(subset=self.features + ['price_direction'], inplace=True)
        scaled_features = self.scaler.transform(df[self.features])
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback_period):
            X.append(scaled_features[i:i + self.lookback_period])
            y.append(df['price_direction'].iloc[i + self.lookback_period - 1])
        return np.array(X), np.array(y)

    def train_model(self):
        """Train the LSTM model."""
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.training_data_limit)
        if df is None or len(df) < self.lookback_period + self.prediction_horizon:
            self.logger.error("Not enough data to train the model.")
            return False
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.features])
        X, y = self.create_sequence_features_and_labels(df)
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model.train()
        for epoch in range(10):  # Example: 10 epochs
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
        torch.save(self.model.state_dict(), self.model_path)
        self.logger.info(f"Model trained and saved to {self.model_path}")

    def incremental_retrain(self, X_batch, y_batch):
        """Incrementally retrain the LSTM model."""
        X_tensor = torch.FloatTensor(X_batch).to(device)
        y_tensor = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        torch.save(self.model.state_dict(), self.model_path)
        self.logger.info("Model incrementally retrained and saved.")

    def predict_price_direction(self):
        """Predict price direction using the LSTM model."""
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.lookback_period)
        if df is None or len(df) < self.lookback_period:
            self.logger.warning("Not enough data for prediction.")
            return None
        scaled_features = self.scaler.transform(df[self.features].tail(self.lookback_period))
        X_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probability = torch.sigmoid(logits).item()
        direction = 'up' if probability >= 0.5 else 'down'
        confidence = probability if direction == 'up' else 1 - probability
        return {'direction': direction, 'confidence': confidence, 'timestamp': datetime.now().isoformat()}

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
                if latest_features_raw.ndim > 1:
                    latest_features_raw = latest_features_raw.flatten()

            except (IndexError, KeyError) as e:
                self.logger.error(f"Failed to get latest raw features for retraining: {e}. Cannot close/retrain.")
                least_features_raw = None

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
                 if order_id:
                     # --- Add to buffer instead of calling retrain directly ---
                     if retrain_label is not None and latest_features_raw is not None:
                         self.retrain_buffer.append((latest_features_raw, retrain_label))
                         self.logger.info(f"Added sample (label {retrain_label}) to retrain buffer. Buffer size: {len(self.retrain_buffer)}")
                     elif latest_features_raw is None:
                          self.logger.warning("Could not add sample to buffer: features unavailable.")
                     # --------------------------------------------------------
                 else:
                      self.logger.error("Failed to place closing order. Manual check needed.")
                 self.current_position = None # Ensure position state is updated
                 self.stop_price = None # Reset TSL after closing
                 self._check_and_trigger_retrain() # Check buffer after adding sample
                 return True # End cycle after closing position
        else:
            self.logger.info(f"No open position. AI Prediction: {direction.upper()} (Confidence: {confidence:.2f})")
            CONFIDENCE_THRESHOLD = 0.70
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
                                try:
                                    latest_features_raw = df.iloc[-1][self.features].values
                                    if latest_features_raw.ndim > 1:
                                        latest_features_raw = latest_features_raw.flatten()
                                    # --- Add penalty sample to buffer ---
                                    self.retrain_buffer.append((latest_features_raw, penalty_label))
                                    self.logger.info(f"Added penalty sample (label {penalty_label}) to retrain buffer. Buffer size: {len(self.retrain_buffer)}")
                                    # ------------------------------------
                                except (IndexError, KeyError) as e:
                                    self.logger.error(f"Failed to get latest raw features for penalty buffer sample: {e}")
                                except Exception as e:
                                    self.logger.error(f"Error during penalty handling: {e}")
                            else:
                                self.logger.info("No significant price movement during inactivity. No penalty.")
                        else:
                            self.logger.warning("Could not calculate actual price change due to zero start price.")
                    except IndexError:
                        self.logger.warning("Not enough data points in df to calculate actual change for penalty.")
                    except Exception as e:
                        self.logger.error(f"Error during penalty calculation: {e}")

        self._check_and_trigger_retrain() # Check buffer after processing
        self.logger.info("Trade strategy cycle completed.")
        return True

    def _check_and_trigger_retrain(self):
        """
        Checks the retraining buffer and triggers incremental retraining
        if the buffer contains samples of both classes (0 and 1) and
        meets a minimum size requirement.
        """
        MIN_BUFFER_FOR_RETRAIN = min(2, self.buffer_size // 10) # Require at least 10 samples or 10% of buffer size

        self.logger.debug(f"Checking retrain buffer (Current size: {len(self.retrain_buffer)}).")

        if len(self.retrain_buffer) < MIN_BUFFER_FOR_RETRAIN:
            self.logger.debug(f"Buffer size ({len(self.retrain_buffer)}) is less than minimum required ({MIN_BUFFER_FOR_RETRAIN}). Not retraining.")
            return

        labels_in_buffer = [item[1] for item in self.retrain_buffer]
        unique_labels = np.unique(labels_in_buffer)

        if len(unique_labels) == 2: # Check if both 0 and 1 are present
            self.logger.info(f"Retrain buffer contains both classes (0 and 1) and meets size requirement ({len(self.retrain_buffer)} >= {MIN_BUFFER_FOR_RETRAIN}). Triggering retrain.")

            # Prepare data batches
            X_batch = np.array([item[0] for item in self.retrain_buffer])
            y_batch = np.array(labels_in_buffer)

            # Perform incremental retraining
            success = self.incremental_retrain(X_batch, y_batch)

            if success:
                # Clear the buffer only if retraining was successful
                self.retrain_buffer.clear()
                self.skipped_trades = 0 # Reset counter after retraining
                self.logger.info("Retraining successful. Buffer cleared.")
            else:
                # Optional: Decide what to do if retrain fails. Keep buffer for next attempt?
                # Or clear buffer anyway to avoid getting stuck? Let's keep it for now.
                self.logger.error("Incremental retrain failed. Buffer content preserved for next attempt.")

        else:
            self.logger.debug(f"Buffer does not contain samples for both classes yet. Unique labels found: {unique_labels}. Not retraining.")

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