# model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tiingo import TiingoClient
from xgboost import XGBRegressor
from datetime import timedelta
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    def __init__(self, api_key, symbol, start_date, end_date, frequency):
        self.api_key = api_key
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.client = TiingoClient({'api_key': self.api_key, 'session': True})
        self.df = None
        self.scaler_X = MinMaxScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rolling_std_train = None
        self.rolling_std_test = None
        self.test_indices = None
        self.features = ['lag1_return', 'lag2_return', 'mean_lag_5']

    def fetch_data(self):
        try:
            historical_prices = self.client.get_crypto_price_history(
                tickers=[self.symbol],
                startDate=self.start_date,
                endDate=self.end_date,
                resampleFreq=self.frequency
            )
            self.df = pd.DataFrame(historical_prices[0]['priceData'])
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
            self.df['close'] = self.df['close'].astype(float)

            self.df['log_return'] = np.log(self.df['close'].shift(-1) / self.df['close'])
            self.df['lag1_return'] = self.df['log_return'].shift(1)
            self.df['lag2_return'] = self.df['log_return'].shift(2)
            self.df['mean_lag_5'] = self.df['log_return'].shift(1).rolling(window=5).mean()
            self.df['rolling_std_100'] = self.df['log_return'].shift(1).rolling(window=100).std()

            self.df.loc[self.df['rolling_std_100'] == 0, 'rolling_std_100'] = 1e-6
            self.df = self.df.dropna(subset=['log_return', 'lag1_return', 'lag2_return', 'mean_lag_5', 'rolling_std_100'])

            min_points = 100
            if len(self.df) < min_points:
                raise ValueError(f"Insufficient data points ({len(self.df)}) fetched. Need at least {min_points}.")
            
            logger.info(f"Data fetched and preprocessed successfully. Fetched {len(self.df)} points "
                        f"from {self.start_date} to {self.end_date}")
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            self.df = None

    def preprocess_data(self, split_ratio=config.MODEL["SPLIT_RATIO"]):
        if self.df is None or self.df.empty:
            raise ValueError("No data available. Fetch data first.")

        X = self.df[self.features].values
        y = self.df['log_return'].values
        rolling_std = self.df['rolling_std_100'].values

        X_scaled = self.scaler_X.fit_transform(X)
        train_size = int(len(X_scaled) * split_ratio)
        self.X_train, self.X_test = X_scaled[:train_size], X_scaled[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        self.rolling_std_train, self.rolling_std_test = rolling_std[:train_size], rolling_std[train_size:]
        self.test_indices = self.df.index[train_size:]

        logger.info(f"Data preprocessed. Shapes - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

    def custom_ztae_objective(self, y_true, y_pred):
        grad = np.sign(y_pred - y_true) / self.rolling_std_train
        hess = np.ones_like(y_true) / self.rolling_std_train
        return grad, hess

    def train_model(self, n_estimators=config.MODEL["N_ESTIMATORS"],
                    learning_rate=config.MODEL["LEARNING_RATE"],
                    max_depth=config.MODEL["MAX_DEPTH"]):
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Preprocess data first.")

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            objective=self.custom_ztae_objective
        )
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model trained successfully")

        importances = pd.DataFrame({'Feature': self.features, 'Importance': self.model.feature_importances_})
        logger.info(f"Feature Importances:\n{importances.sort_values(by='Importance', ascending=False)}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def evaluate_model(self):
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained or no test data available.")

        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        train_ztae = np.mean(np.abs(self.y_train - train_pred) / self.rolling_std_train)
        test_ztae = np.mean(np.abs(self.y_test - test_pred) / self.rolling_std_test)
        corr_coeff, _ = pearsonr(self.y_test, test_pred)

        logger.info(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        logger.info(f"Train ZTAE: {train_ztae:.6f}, Test ZTAE: {test_ztae:.6f}")
        if test_mse <= train_mse * 1.5:
            logger.info("No overfitting detected")
        else:
            logger.warning("Potential overfitting detected")

        return train_mse, test_mse, train_ztae, test_ztae, corr_coeff

    def rolling_forecast(self, forecast_hours=24):
        if self.model is None:
            raise ValueError("Model not trained or loaded yet.")
        if self.df is None or len(self.df) < 50:
            raise ValueError(f"Insufficient data for forecast. Need at least 50 points, got {len(self.df) if self.df is not None else 0}.")
        if not hasattr(self.scaler_X, 'feature_range'):
            raise ValueError("Scaler not fitted. Preprocess data first.")

        logger.info(f"Starting rolling forecast with {len(self.df)} data points")
        predictions = []
        latest_data = self.df.tail(50).copy()
        current_price = self.df['close'].iloc[-1]
        current_time = self.df.index[-1]
        logger.info(f"Initial state - current_price: {current_price}, current_time: {current_time}")

        for i in range(forecast_hours):
            logger.debug(f"Forecast step {i+1}/{forecast_hours}")
            try:
                latest_scaled = self.scaler_X.transform(latest_data[self.features].values[-1].reshape(1, -1))
                prediction = self.model.predict(latest_scaled)[0]
                new_price = current_price * np.exp(prediction)

                start_time = current_time + timedelta(hours=i)
                end_time = start_time + timedelta(hours=1)

                predictions.append({
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'current_price': float(current_price),
                    'log_return': float(prediction),
                    'new_price': float(new_price)
                })

                new_row = pd.DataFrame([[prediction, latest_data['lag1_return'].iloc[-1],
                                        latest_data['mean_lag_5'].iloc[-1]]],
                                      columns=self.features, index=[end_time])
                latest_data = pd.concat([latest_data.iloc[1:], new_row])
                current_price = new_price
                logger.debug(f"Step {i+1} completed - new_price: {new_price}")
            except Exception as e:
                logger.error(f"Forecast step {i+1} failed: {str(e)}")
                raise

        logger.info("Rolling forecast completed")
        return predictions

    def save_model(self, filename_json=config.MODEL["SAVE_JSON"]):
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(filename_json)
        logger.info("Model saved successfully")

    def load_model(self, filename_json=config.MODEL["SAVE_JSON"]):
        self.model = XGBRegressor()
        self.model.load_model(filename_json)
        logger.info("Model loaded successfully")