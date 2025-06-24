# app.py
import pandas as pd
import numpy as np
from flask import Flask, jsonify, Response
from model import CryptoPricePredictor
from datetime import datetime
import config
import logging
from flask import request

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

predictor = None

def initialize_app():
    global predictor
    try:
        predictor = CryptoPricePredictor(
            api_key=config.TIINGO["API_KEY"],
            symbol=config.TIINGO["SYMBOL"],
            start_date=config.TIINGO["START_DATE"],
            end_date=config.TIINGO["END_DATE"],
            frequency=config.TIINGO["FREQUENCY"]
        )
        logger.info(f"CryptoPricePredictor initialized successfully. "
                    f"Date range: {config.TIINGO['START_DATE']} to {config.TIINGO['END_DATE']}")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        predictor = None

initialize_app()

if predictor is None:
    logger.error("Predictor initialization failed.")

@app.route(config.API["ENDPOINTS"]["HEALTH"], methods=['GET'])
def health():
    try:
        status = {
            "status": "OK",
            "predictor_initialized": predictor is not None,
            "model_trained": predictor is not None and predictor.model is not None
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response("Health check failed", status=500)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Crypto Price Prediction API", "endpoints": config.API["ENDPOINTS"]})

@app.route(config.API["ENDPOINTS"]["FETCH_DATA"], methods=['GET'])
def fetch_data():
    try:
        logger.info("Fetching data from Tiingo")
        if predictor is None:
            return jsonify({"error": "Predictor not initialized"}), 500
        predictor.fetch_data()
        if predictor.df is None or predictor.df.empty:
            return jsonify({"error": "No data fetched from Tiingo"}), 500
        data_head = predictor.df.head().reset_index().to_dict(orient='records')
        logger.info("Data fetched successfully")
        return jsonify({"message": "Data fetched successfully", "data_head": data_head})
    except Exception as e:
        logger.error(f"Fetch data failed: {str(e)}")
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["TRAIN"], methods=['GET'])
def train():
    try:
        logger.info("Starting model training")
        if predictor is None:
            return jsonify({"error": "Predictor not initialized"}), 500
        if predictor.df is None or predictor.df.empty:
            return jsonify({"error": "Please fetch data first"}), 400
        predictor.preprocess_data(split_ratio=config.MODEL["SPLIT_RATIO"])
        predictor.train_model(
            n_estimators=config.MODEL["N_ESTIMATORS"],
            learning_rate=config.MODEL["LEARNING_RATE"],
            max_depth=config.MODEL["MAX_DEPTH"]
        )
        logger.info("Model trained successfully")
        return jsonify({"message": "Model trained successfully"})
    except ValueError as ve:
        logger.error(f"ValueError during training: {str(ve)}")
        return jsonify({"error": f"Training failed: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return jsonify({"error": f"Unexpected error during training: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["PREDICT"], methods=['GET'])
def predict():
    try:
        logger.info("Making prediction")
        if predictor is None or predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        if predictor.X_test is None:
            return jsonify({"error": "No test data available; train the model first"}), 400
        test_features = predictor.X_test[-1].reshape(1, -1)
        prediction = predictor.predict(test_features)
        current_price = predictor.df['close'].iloc[-1]
        predicted_price = current_price * np.exp(prediction[0])
        logger.info("Prediction generated successfully")
        return jsonify({
            "log_return": float(prediction[0]),
            "current_price": float(current_price),
            "predicted_price": float(predicted_price)
        })
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": f"Failed to predict: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["EVALUATE"], methods=['GET'])
def evaluate():
    try:
        logger.info("Evaluating model")
        if predictor is None or predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        train_mse, test_mse, train_ztae, test_ztae, corr_coeff = predictor.evaluate_model()
        logger.info("Model evaluation completed")
        return jsonify({
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_ztae": float(train_ztae),
            "test_ztae": float(test_ztae),
            "correlation_coefficient": float(corr_coeff)
        })
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return jsonify({"error": f"Failed to evaluate: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["SAVE_MODEL"], methods=['GET'])
def save_model():
    try:
        logger.info("Saving model")
        if predictor is None:
            logger.error("Predictor not initialized")
            return jsonify({"error": "Predictor not initialized"}), 500
        predictor.save_model(filename_json=config.MODEL["SAVE_JSON"])
        logger.info("Model saved successfully")
        return jsonify({"message": "Model saved successfully"}), 200
    except Exception as e:
        logger.error(f"Save model failed: {str(e)}")
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["LOAD_MODEL"], methods=['GET'])
def load_model():
    try:
        logger.info("Loading model")
        if predictor is None:
            logger.error("Predictor not initialized")
            return jsonify({"error": "Predictor not initialized"}), 500
        predictor.load_model(filename_json=config.MODEL["SAVE_JSON"])
        logger.info("Model loaded successfully")
        return jsonify({"message": "Model loaded successfully"}), 200
    except Exception as e:
        logger.error(f"Load model failed: {str(e)}")
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["FORECAST"], methods=['GET'])
def forecast():
    try:
        logger.info("Generating forecast")
        if predictor is None or predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        forecast_data = predictor.rolling_forecast(forecast_hours=24)
        logger.info("Forecast generated successfully")
        return jsonify({"message": "Forecast generated successfully", "forecast": forecast_data})
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        return jsonify({"error": f"Failed to generate forecast: {str(e)}"}), 500

@app.route('/inference/<token>', methods=['GET'])
def inference_token(token):
    try:
        logger.info(f"Inference requested for token: {token}")
        if predictor is None:
            logger.error("Predictor not initialized")
            return jsonify({"error": "Predictor not initialized"}), 500
        if predictor.df is None or predictor.df.empty:
            logger.info("Fetching data on first inference")
            predictor.fetch_data()
            if predictor.df is None or predictor.df.empty:
                return jsonify({"error": "Failed to fetch initial data"}), 500
        if predictor.model is None:
            logger.info("Training model on first inference")
            predictor.preprocess_data(split_ratio=config.MODEL["SPLIT_RATIO"])
            predictor.train_model(
                n_estimators=config.MODEL["N_ESTIMATORS"],
                learning_rate=config.MODEL["LEARNING_RATE"],
                max_depth=config.MODEL["MAX_DEPTH"]
            )
        forecast = predictor.rolling_forecast(forecast_hours=1)  # Single-hour prediction
        logger.info(f"Inference generated successfully for token: {token}")
        return jsonify({
            "token": token,
            "prediction": forecast[0]  # Return 1-hour forecast
        })
    except Exception as e:
        logger.error(f"Inference failed for token {token}: {str(e)}")
        return jsonify({"error": f"Failed to generate inference for token {token}: {str(e)}"}), 500

# Added for Allora worker compatibility
@app.route('/inference', methods=['GET'])
def inference():
    try:
        logger.info("Inference requested for Allora")
        if predictor is None:
            logger.error("Predictor not initialized")
            return jsonify({"error": "Predictor not initialized"}), 500
        if predictor.df is None or predictor.df.empty:
            logger.info("Fetching data on first inference")
            predictor.fetch_data()
            if predictor.df is None or predictor.df.empty:
                return jsonify({"error": "Failed to fetch initial data"}), 500
        if predictor.model is None:
            logger.info("Training model on first inference")
            predictor.preprocess_data(split_ratio=config.MODEL["SPLIT_RATIO"])
            predictor.train_model(
                n_estimators=config.MODEL["N_ESTIMATORS"],
                learning_rate=config.MODEL["LEARNING_RATE"],
                max_depth=config.MODEL["MAX_DEPTH"]
            )
        forecast = predictor.rolling_forecast(forecast_hours=1)  # Single-hour prediction
        logger.info("Inference generated successfully")
        print("Predicted log return:", forecast[0]['log_return'])
        return Response(str(forecast[0]['log_return']), status=200)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        print(f"error Failed to generate inference: {str(e)}")
        return jsonify({"error": f"Failed to generate inference: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask app on {config.API['HOST']}:{config.API['PORT']}")
    app.run(host=config.API["HOST"], port=config.API["PORT"], debug=False)