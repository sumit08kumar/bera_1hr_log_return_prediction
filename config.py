# config.py
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

default_end_date = datetime.now().strftime('%Y-%m-%d')
default_start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

TIINGO = {
    "API_KEY": os.getenv("TIINGO_API_KEY"),
    "SYMBOL": os.getenv("TIINGO_SYMBOL"),
    "START_DATE": os.getenv("TIINGO_START_DATE", default_start_date),
    "END_DATE": os.getenv("TIINGO_END_DATE", default_end_date),
    "FREQUENCY": os.getenv("TIINGO_FREQUENCY")
}

MODEL = {
    "SPLIT_RATIO": float(os.getenv("MODEL_SPLIT_RATIO", 0.8)),
    "OPTUNA_TRIALS": int(os.getenv("MODEL_OPTUNA_TRIALS", 50)),
    "N_ESTIMATORS": int(os.getenv("MODEL_N_ESTIMATORS", 100)),
    "LEARNING_RATE": float(os.getenv("MODEL_LEARNING_RATE", 0.1)),
    "MAX_DEPTH": int(os.getenv("MODEL_MAX_DEPTH", 3)),
    "SAVE_JSON": os.getenv("MODEL_SAVE_JSON"),
    "SAVE_PKL": os.getenv("MODEL_SAVE_PKL")
}

API = {
    "HOST": os.getenv("API_HOST", "0.0.0.0"),
    "PORT": int(os.getenv("API_PORT", 8000)),
    "BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000"),
    "ENDPOINTS": {
        "FETCH_DATA": "/fetch_data",
        "TRAIN": "/train",
        "PREDICT": "/predict",
        "EVALUATE": "/evaluate",
        "SAVE_MODEL": "/save_model",
        "LOAD_MODEL": "/load_model",
        "HEALTH": "/health",
        "FORECAST": "/forecast",
        "INFERENCE": "/inference/<token>"
    }
}