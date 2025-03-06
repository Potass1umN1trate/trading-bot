import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
        quantity_step  # Add quantity_step parameter
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
        self.last_price = initial_price
        
        # AI model parameters
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.model = None
        self.scaler = None

        self.interval_seconds = interval_seconds
        self.training_data_limit = training_data_limit
        self.trading_interval = trading_interval
        self.quantity_step = quantity_step  # Initialize quantity_step
        
        # Initialize or load AI model
        self.initialize_model()

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

    def calculate_indicators(self, df):
        """Calculate technical indicators for the dataset"""
        if df is None or len(df) == 0:
            return None
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma'] + (df['std'] * 2)
        df['bollinger_lower'] = df['sma'] - (df['std'] * 2)
        
        # Percentage change features
        df['price_change_1h'] = df['close'].pct_change(1) * 100
        df['price_change_4h'] = df['close'].pct_change(4) * 100
        df['price_change_24h'] = df['close'].pct_change(24) * 100
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change() * 100
        
        # Drop NaN values resulting from calculations
        df.dropna(inplace=True)
        
        return df

    def create_labels(self, df):
        """Create target labels for AI training based on future price movement"""
        # Calculate future price movement (1 hour ahead)
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
        y = df['price_direction']
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(self.model, 'trading_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        
        # Log feature importance
        feature_importance = pd.DataFrame(
            self.model.feature_importances_,
            index=self.features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        self.logger.info("Model trained successfully")
        self.logger.info(f"Feature importance:\n{feature_importance}")
        
        return True

    def predict_price_direction(self):
        """Use the trained model to predict price direction"""
        if self.model is None or self.scaler is None:
            self.logger.error("Model not initialized")
            return None
        
        # Fetch recent data
        df = self.fetch_market_data(interval='60', limit=30)
        
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

    def execute_trade_strategy(self):
        """Execute the AI-powered trading strategy"""
        # Get current market state
        current_price = self.get_current_price()
        if not current_price:
            return False
        
        # Check current positions
        position = self.check_open_positions()
        
        # Get AI prediction
        prediction = self.predict_price_direction()
        if not prediction:
            return False
        
        # Trading logic based on AI prediction and existing positions
        if position:
            # We have an open position, check if we should close it
            entry_price = position['entry_price']
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # For long positions
            if position['side'] == 'Buy':
                # Take profit or stop loss
                if (price_change_pct >= self.profit_threshold or 
                    price_change_pct <= self.stop_loss_threshold or
                    prediction['direction'] == 'down' and prediction['confidence'] > 0.75 ):
                    
                    self.logger.info(f"Closing LONG position. Price change: {price_change_pct:.2f}%, " +
                                   f"AI prediction: {prediction['direction']} ({prediction['confidence']:.2f})")
                    
                    # Place sell order to close the position
                    return self.place_order("Sell", position['size'])
            
            # For short positions
            elif position['side'] == 'Sell':
                # Take profit or stop loss (for shorts, profit when price goes down)
                if (price_change_pct <= -self.profit_threshold or 
                    price_change_pct >= self.stop_loss_threshold or
                    prediction['direction'] == 'up' and prediction['confidence'] > 0.75):
                    
                    self.logger.info(f"Closing SHORT position. Price change: {price_change_pct:.2f}%, " +
                                   f"AI prediction: {prediction['direction']} ({prediction['confidence']:.2f})")
                    
                    # Place buy order to close the position
                    return self.place_order("Buy", position['size'])
        
        else:
            # No open positions, check if we should open one
            if prediction['confidence'] > 65: # Only trade when confidence is high
                # Calculate quantity based on order value
                quantity = self.calculate_trade_quantity(current_price, self.order_value)
                
                if prediction['direction'] == 'up':
                    self.logger.info(f"Opening LONG position based on AI prediction with {prediction['confidence']:.2f} confidence")
                    return self.place_order("Buy", quantity)
                else:
                    self.logger.info(f"Opening SHORT position based on AI prediction with {prediction['confidence']:.2f} confidence")
                    return self.place_order("Sell", quantity)
        
        # No trade executed
        self.logger.info("No trade signals at this time")
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
        quantity_step=0.001  # Add quantity_step parameter
    )
    bot.run()