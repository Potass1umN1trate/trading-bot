import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # Replace RandomForest with MLPClassifier
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
        order_value,
        upward_trend_threshold,
        dip_threshold,
        profit_threshold,
        stop_loss_threshold,
        initial_price,
        lookback_period,
        prediction_horizon,
        features,
        interval_seconds,
        training_data_limit,
        trading_interval,
        quantity_step,  # Add quantity_step parameter
        risk_per_trade=0.01,  # Add risk_per_trade parameter
        trailing_stop_loss=0.075
    ):
        # Initialize logger
        self.setup_logger()
        
        # Initialize the Bybit client
        self.client = HTTP(
            testnet=False,
            api_key=bybit_api_key,
            api_secret=bybit_api_secret,
            demo=True
        )
        self.logger.info(f"Using Bybit API URL: {bybit_url}")
        
        # Trading parameters
        self.symbol = symbol
        self.category = category
        self.order_value = order_value
        self.position_size = 0
        self.current_position = None
        
        # Load trading thresholds
        self.upward_trend_threshold = upward_trend_threshold
        self.dip_threshold = dip_threshold
        self.profit_threshold = profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.trailing_stop_loss = trailing_stop_loss
        self.last_price = initial_price
        
        # AI model parameters
        self.skipped_trades = 0  # Track how many times the bot skips trading
        self.max_skipped_trades = 3  # Define a threshold for inaction punishment
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.model = None
        self.scaler = None

        self.interval_seconds = interval_seconds
        self.training_data_limit = training_data_limit
        self.trading_interval = trading_interval
        self.quantity_step = quantity_step  # Initialize quantity_step
        self.risk_per_trade = risk_per_trade  # Initialize risk_per_trade
        
        # Initialize or load AI model
        self.initialize_model()
        self.stop_price = None  # NEW: STORES TSL VALUE

    def setup_logger(self):
        """Configure logging with timestamps and rotation"""
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(logging.DEBUG)
        
        # Create a file handler that logs even debug messages
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def initialize_model(self):
        """Initialize or load AI model"""
        model_path = 'trading_model.joblib'
        scaler_path = 'scaler.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.logger.info("Loading existing AI model and scaler")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            self.logger.info("Training new AI model")
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
            
            # Convert to DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp in ascending order
            df = df.sort_values('timestamp')
            
            self.logger.info(f"Fetched {len(df)} klines from Bybit")
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None

    def get_indicator_settings(self):
        """
        Dynamically adjusts indicator parameters based on the selected trading interval.
        """
        interval = int(self.trading_interval)  # Convert to integer
        if interval == 1:
            return {"rsi_period": 7, "macd_short": 6, "macd_long": 13, "macd_signal": 5, "bollinger_window": 10}
        elif interval <= 15:
            return {"rsi_period": 14, "macd_short": 12, "macd_long": 26, "macd_signal": 9, "bollinger_window": 20}
        elif interval <= 60:
            return {"rsi_period": 21, "macd_short": 26, "macd_long": 50, "macd_signal": 9, "bollinger_window": 30}
        else:
            return {"rsi_period": 21, "macd_short": 50, "macd_long": 100, "macd_signal": 9, "bollinger_window": 50}

    def calculate_indicators(self, df):
        """
        Calculate technical indicators using adaptive settings based on trading interval.
        """
        if df is None or len(df) == 0:
            self.logger.warning("calculate_indicators: Received empty or None dataframe.")
            return None
        
        df = df.copy()
        settings = self.get_indicator_settings()

        self.logger.info(f"Calculating indicators with settings: {settings}")

        # RSI Calculation
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=settings["rsi_period"], min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=settings["rsi_period"], min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))

        #self.logger.debug(f"RSI calculated: Last 5 values:\n{df['rsi'].tail()}")

        # MACD Calculation
        df['macd'] = df['close'].ewm(span=settings["macd_short"], adjust=False).mean() - df['close'].ewm(span=settings["macd_long"], adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=settings["macd_signal"], adjust=False).mean()

        #self.logger.debug(f"MACD calculated: Last 5 values:\n{df[['macd', 'macd_signal']].tail()}")

        # Bollinger Bands
        df['sma'] = df['close'].rolling(window=settings["bollinger_window"]).mean()
        df['std'] = df['close'].rolling(window=settings["bollinger_window"]).std()
        df['bollinger_upper'] = df['sma'] + (df['std'] * 2)
        df['bollinger_lower'] = df['sma'] - (df['std'] * 2)

        #self.logger.debug(f"Bollinger Bands calculated: Last 5 values:\n{df[['bollinger_upper', 'bollinger_lower']].tail()}")

        # VWAP
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()

        #self.logger.debug(f"VWAP calculated: Last 5 values:\n{df['vwap'].tail()}")

        # ATR (Average True Range for volatility)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()

        #self.logger.debug(f"ATR calculated: Last 5 values:\n{df['atr'].tail()}")

        # Momentum & Price Change
        df['momentum'] = df['close'].diff(4)
        df['price_change_1m'] = df['close'].pct_change(1) * 100
        df['price_change_5m'] = df['close'].pct_change(5) * 100
        df['price_change_15m'] = df['close'].pct_change(15) * 100
        df['volume_change'] = df['volume'].pct_change() * 100

        #self.logger.debug(f"Price changes calculated: Last 5 values:\n{df[['price_change_1m', 'price_change_5m', 'price_change_15m']].tail()}")

        df.dropna(inplace=True)

        self.logger.info(f"Indicator calculation complete. DataFrame shape: {df.shape}")

        return df

    def create_labels(self, df):
        """Create target labels for AI training based on future price movement"""
        # Calculate future price movement (1 minute ahead)
        df['future_price'] = df['close'].shift(-self.prediction_horizon)
        df['price_direction'] = (df['future_price'] > df['close']).astype(int)
        
        # Remove rows with NaN values (last rows won't have future data)
        df.dropna(inplace=True)
        
        return df

    def train_model(self):
        """Train the AI prediction model"""
        # Fetch historical data
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.training_data_limit)  # Get more data for training
        
        if df is None or len(df) < 100:
            self.logger.error("Not enough data to train the model")
            return False
        
        # Calculate indicators and create labels
        df = self.calculate_indicators(df)
        df = self.create_labels(df)
        
        if df is None or len(df) == 0:
            self.logger.error("Failed to prepare training data")
            return False
        
        # Prepare features and target
        X = df[self.features]
        y = df['price_direction'].values  # Convert to NumPy array without index
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model using MLPClassifier with partial_fit
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Two hidden layers: 64 neurons, then 32
            activation='relu',            # ReLU activation
            solver='adam',                # Adam optimizer
            max_iter=1,                   # We'll train in small steps
            warm_start=True,              # Keep the model state between fits
            random_state=42               # For reproducibility
        )
        self.model.partial_fit(X_scaled, y, classes=[0, 1])
        
        # Save model and scaler
        joblib.dump(self.model, 'trading_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        
        self.logger.info("Model partially fit on initial dataset.")
        return True

    def incremental_retrain(self, X_new, y_new):
        """Incrementally retrain the model with new data"""
        if self.model is None or self.scaler is None:
            self.logger.error("Model or scaler not initialized.")
            return

        # temproary fix for the case when only one class is present in y_new
        # Ensure y_new contains both classes
        if len(np.unique(y_new)) == 1:  # Only one class present
            y_fake = np.array([1 - y_new[0]])  # Add missing class
            X_fake = np.zeros((1, X_new.shape[1]))  # Dummy input
            X_new = np.vstack([X_new, X_fake])  # Stack fake X
            y_new = np.append(y_new, y_fake)  # Append fake y

        # Scale the new data with the existing scaler
        X_new_scaled = self.scaler.transform(X_new)
        # Call partial_fit with classes=[0,1]
        self.model.partial_fit(X_new_scaled, y_new, classes=[0, 1])
        # Save the updated model
        joblib.dump(self.model, 'trading_model.joblib')
        self.logger.info("Incremental partial_fit completed.")

    def predict_price_direction(self):
        """Use the trained model to predict price direction"""
        if self.model is None or self.scaler is None:
            self.logger.error("Model not initialized")
            return None
        
        # Fetch recent data
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.lookback_period)
        
        if df is None or len(df) == 0:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        if df is None or len(df) < 5:  # Need enough data for features
            return None
        
        # Get latest data point
        latest_data = df.iloc[-1][self.features].values.reshape(1, -1)
        
        # Scale features
        scaled_data = self.scaler.transform(latest_data)
        
        # Make prediction (1 = up, 0 = down)
        prediction = self.model.predict(scaled_data)[0]
        probability = self.model.predict_proba(scaled_data)[0]
        
        prediction_confidence = probability[1] if prediction == 1 else probability[0]
        
        self.logger.info(f"AI prediction: {'UP' if prediction == 1 else 'DOWN'} with {prediction_confidence:.2f} confidence")
        
        return {
            'direction': 'up' if prediction == 1 else 'down',
            'confidence': prediction_confidence,
            'timestamp': datetime.now().isoformat()
        }

    def get_wallet_balance(self):
        """Get available balance from wallet"""
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            
            if response['retCode'] != 0:
                self.logger.error(f"Error getting wallet balance: {response['retMsg']}")
                return None
            
            balance = float(response['result']['list'][0]['coin'][0]['walletBalance'])
            self.logger.info(f"Current wallet balance: {balance} USDT")
            return balance
            
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
            
            price = float(response['result']['list'][0]['lastPrice'])
            self.logger.info(f"Current {self.symbol} price: {price}")
            return price
            
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
                self.logger.error(f"Error checking positions: {response['retMsg']}")
                return None
            
            positions = response['result']['list']
            
            if not positions or float(positions[0]['size']) == 0:
                self.logger.info("No open positions")
                self.current_position = None
                return None
            
            position = {
                'size': float(positions[0]['size']),
                'side': positions[0]['side'],
                'entry_price': float(positions[0]['avgPrice']),
                'unrealized_pnl': float(positions[0]['unrealisedPnl'])
            }
            
            self.current_position = position
            self.logger.info(f"Current position: {position['size']} {self.symbol} {position['side']} " +
                           f"at {position['entry_price']}, PnL: {position['unrealized_pnl']}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error checking positions: {str(e)}")
            return None

    def place_order(self, side, quantity, order_type="Market", price=None):
        """Place an order on Bybit"""
        try:
            order_params = {
                "category": self.category,
                "symbol": self.symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(quantity)
            }
            
            if price and order_type == "Limit":
                order_params["price"] = str(price)
                order_params["timeInForce"] = "PostOnly"
            else:
                order_params["timeInForce"] = "GTC"
            
            response = self.client.place_order(**order_params)
            
            if response['retCode'] != 0:
                self.logger.error(f"Order error: {response['retMsg']}")
                return None
            
            order_id = response['result']['orderId']
            self.logger.info(f"Order placed: {side} {quantity} {self.symbol} at {price if price else 'market price'}, ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None

    def calculate_trade_quantity(self, price, usd_amount):
        """Calculate the quantity to trade based on USD amount"""
        if not price:
            return None
        
        # Calculate quantity and round to the nearest quantity step
        quantity = round(usd_amount / price / self.quantity_step) * self.quantity_step
        return quantity

    def calculate_risk_based_quantity(self, current_price):
        # 1) Get wallet balance
        balance = self.get_wallet_balance()
        if balance is None:
            return None
        
        # 2) Risk capital = balance * self.risk_per_trade
        risk_capital = balance * self.risk_per_trade
        
        # 3) Decide a stop-loss distance, e.g. use ATR or a fixed fraction:
        df = self.fetch_market_data(interval=self.trading_interval, limit=30)
        df = self.calculate_indicators(df)
        if df is None or 'atr' not in df.columns:
            self.logger.warning('No ATR available, fallback to fixed stop distance.')
            stop_distance = current_price * 0.01  # e.g. 1% of price
        else:
            # Use last ATR value
            stop_distance = df.iloc[-1]['atr']
        
        # 4) Position size = risk_capital / (stop_distance)
        quantity = risk_capital / stop_distance

        # Round to nearest quantity step
        quantity = round(quantity / self.quantity_step) * self.quantity_step
        return quantity

    def should_open_long(self, df, prediction_confidence):
        """
        Decide if we should open a LONG position based on:
        - AI confidence
        - MACD > MACD signal
        - RSI above 50
        """
        # Basic checks
        if prediction_confidence < 0.70:
            return False  # not confident enough
        
        # Confluence checks
        latest_row = df.iloc[-1]
        if latest_row['macd'] <= latest_row['macd_signal']:
            return False
        if latest_row['rsi'] <= 50:
            return False
        
        return True

    def should_open_short(self, df, prediction_confidence):
        """
        Decide if we should open a SHORT position based on:
        - AI confidence
        - MACD < MACD signal
        - RSI below 50
        """
        if prediction_confidence < 0.70:
            return False
        
        latest_row = df.iloc[-1]
        if latest_row['macd'] >= latest_row['macd_signal']:
            return False
        if latest_row['rsi'] >= 50:
            return False
        
        return True

    def place_atr_based_stop(self, side, entry_price, df):
        if df is None or 'atr' not in df.columns:
            return None, None
        
        last_atr = df.iloc[-1]['atr']
        if side == 'Buy':
            stop_loss = entry_price - 1.5 * last_atr
            take_profit = entry_price + 2.0 * last_atr
        else:  # side == 'Sell'
            stop_loss = entry_price + 1.5 * last_atr
            take_profit = entry_price - 2.0 * last_atr
        
        return stop_loss, take_profit

    def update_trailing_stop(self, side, entry_price, current_price, trailing_stop_price, trail_gap):
        """
        Move trailing stop up if current_price - trailing_stop_price > trail_gap for a long.
        """
        if side == 'Buy':
            if (current_price - trailing_stop_price) >= trail_gap:
                new_stop = current_price - trail_gap
                return max(new_stop, entry_price)  # ensure not below entry
        else:
            # Opposite for short
            if (trailing_stop_price - current_price) >= trail_gap:
                new_stop = current_price + trail_gap
                return min(new_stop, entry_price)
        
        return trailing_stop_price

    def execute_trade_strategy(self):
        current_price = self.get_current_price()
        if not current_price:
            return False
        
        position = self.check_open_positions()
        
        # [IMPROVEMENT] Fetch fresh data for confluence
        df = self.fetch_market_data(interval=self.trading_interval, limit=self.lookback_period)
        df = self.calculate_indicators(df)
        if df is None or len(df) < self.prediction_horizon:
            self.logger.info(f"Not enough data to make a confluence-based decision. Data length: {len(df)}")
            return False
        
        # [IMPROVEMENT] AI Prediction
        prediction = self.predict_price_direction()
        if not prediction:
            return False
        
        direction = prediction['direction']  # 'up' or 'down'
        confidence = prediction['confidence']
        
        if position:
            # We already have a position open
            self.skipped_trades = 0

            entry_price = position['entry_price']
            side = position['side']  # 'Buy' or 'Sell'
            unrealized_pnl = position['unrealized_pnl']
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
            self.logger.info(f"Current position: {side} {position['size']} at {entry_price}, PnL: {unrealized_pnl:.2f} USDT")
            self.logger.info(f"Current price change: {price_change_pct:.2f}%")

            if unrealized_pnl > 1:
                if self.stop_price is None:
                    self.stop_price = current_price * (1 - self.trailing_stop_loss/100) if side == "Buy" else current_price * (1 + self.trailing_stop_loss/100)
                    self.logger.info(f"Initialized TSL at: {self.stop_price}")
            else:
                self.stop_price = None
                # self.stop_price = entry_price * (1 - self.stop_loss_threshold/100) if side == "Buy" else entry_price * (1 + self.stop_loss_threshold/100)
                # self.logger.info(f"Initialized SL at: {self.stop_price}")
            
            # [IMPROVEMENT] Check if trailing stop or partial close is triggered
            # For demonstration, keep it simple:
            
            if side == 'Buy':
                # If price went up enough, or new signal is strongly down, close
                if (self.stop_price and current_price <= self.stop_price) or (not self.stop_price and price_change_pct <= -self.profit_threshold):
                    self.logger.info(f"Closing LONG position. Price change: {price_change_pct:.2f}%, AI: {direction} ({confidence:.2f})")
                    self.place_order("Sell", position['size'])
                    #self.skipped_trades = 0
                    # Incremental retrain with new data
                    X_new = df.iloc[-1][self.features].values.reshape(1, -1)
                    y_new = np.array([1 if price_change_pct >= self.profit_threshold else 0], dtype=int)
                    self.logger.info(f"Incremental retrain with new data. Input shape: {X_new}, target shape: {y_new}")
                    self.incremental_retrain(X_new, y_new)
                    return True
                elif self.stop_price and current_price > self.stop_price/(1 - self.trailing_stop_loss/100):
                    self.stop_price = current_price * (1 - self.trailing_stop_loss/100)
                    self.logger.info(f"Updated TSL to: {self.stop_price}")
            else:
                # side == 'Sell'
                if (self.stop_price and current_price >= self.stop_price) or (not self.stop_price and price_change_pct >= self.profit_threshold):
                    self.logger.info(f"Closing SHORT position. Price change: {price_change_pct:.2f}%, AI: {direction} ({confidence:.2f})")
                    self.place_order("Buy", position['size'])
                    #self.skipped_trades = 0
                    # Incremental retrain with new data
                    X_new = df.iloc[-1][self.features].values.reshape(1, -1)
                    y_new = np.array([0 if -price_change_pct >= self.profit_threshold else 1], dtype=int)
                    self.logger.info(f"Incremental retrain with new data. Input shape: {X_new}, target shape: {y_new}")
                    self.incremental_retrain(X_new, y_new)
                    return True
                elif self.stop_price and current_price < self.stop_price/(1 + self.trailing_stop_loss/100):
                    self.stop_price = current_price * (1 + self.trailing_stop_loss/100)
                    self.logger.info(f"Updated TSL to: {self.stop_price}")
            
        else:
            # No open positions
            # [IMPROVEMENT] Evaluate confluence
            if direction == 'up':
                if confidence > 0.75: #self.should_open_long(df, confidence):
                    quantity = self.calculate_risk_based_quantity(current_price)
                    if quantity and quantity > 0:
                        self.logger.info(f"Opening LONG. Confidence: {confidence:.2f}")
                        self.skipped_trades = 0
                        return self.place_order("Buy", quantity)
            elif direction == 'down':
                if confidence > 0.75: #self.should_open_short(df, confidence):
                    quantity = self.calculate_risk_based_quantity(current_price)
                    if quantity and quantity > 0:
                        self.logger.info(f"Opening SHORT. Confidence: {confidence:.2f}")
                        self.skipped_trades = 0
                        return self.place_order("Sell", quantity)
        
        self.logger.info("No trade signals at this time.")

        self.skipped_trades += 1
        # If price moves significantly while bot stays inactive, penalize it
        if self.skipped_trades >= self.max_skipped_trades:
            last_movement = abs(df.iloc[-1]['price_change_15m'])  # Check 15m price change
            punishment_threshold = self.profit_threshold  # Adjusted from 1.5% to 0.3% (can be tweaked)

            self.logger.info(f"Last price movement: {last_movement:.2f}%")

            if last_movement > punishment_threshold:
                self.logger.warning(f"Missed a {last_movement:.2f}% price move! Penalizing model for inaction.")
                
                # The bot was inactive, so we assume it should have traded
                X_penalty = df.iloc[-1][self.features].values.reshape(1, -1)

                # If price moved UP, encourage an "UP" prediction next time
                y_penalty = np.array([1]) if df.iloc[-1]['price_change_15m'] > 0 else np.array([0])

                self.incremental_retrain(X_penalty, y_penalty)
            else:
                self.logger.info("No significant price movement while bot was inactive.")

        return True

    def run(self):
        """Run the trading bot in a loop"""
        self.logger.info(f"Starting trading bot for {self.symbol}")
        
        while True:
            try:
                self.execute_trade_strategy()
                
                # Sleep until next iteration
                self.logger.info(f"Sleeping for {self.interval_seconds} seconds")
                time.sleep(self.interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Trading bot stopped by user")
                break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    bot = TradingBot(
        bybit_api_key='your_api_key',
        bybit_api_secret='your_api_secret',
        bybit_url='https://api.bybit.com',
        symbol='BTCUSDT',
        category='linear',
        order_value=100,
        upward_trend_threshold=0.5,
        dip_threshold=0.5,
        profit_threshold=1.0,
        stop_loss_threshold=0.5,
        initial_price=50000,
        lookback_period=14,
        prediction_horizon=1,
        features=['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'price_change_1h', 'price_change_4h', 'price_change_24h', 'volume_change'],
        interval_seconds=300,
        training_data_limit=2000,
        trading_interval='60',
        quantity_step=0.001,  # Add quantity_step parameter
        risk_per_trade=0.01  # Add risk_per_trade parameter
    )
    bot.run()