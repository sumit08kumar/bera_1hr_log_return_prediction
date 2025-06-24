# api_client.py
import requests
import logging
from typing import Dict, Any
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlaskAPIClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or config.API["BASE_URL"]).rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        self.session = requests.Session()
        logger.info(f"Initialized FlaskAPIClient with base_url: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        logger.info(f"Requesting {method.upper()} {url}")
        try:
            response = self.session.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            logger.info(f"Request to {url} successful")
            return response.json() if response.content else {}
        except requests.RequestException as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["HEALTH"])

    def fetch_data(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["FETCH_DATA"])

    def train_model(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["TRAIN"])

    def predict(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["PREDICT"])

    def evaluate_model(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["EVALUATE"])

    def save_model(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["SAVE_MODEL"])

    def load_model(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["LOAD_MODEL"])

    def forecast(self) -> Dict[str, Any]:
        return self._make_request("GET", config.API["ENDPOINTS"]["FORECAST"])

    def get_inference(self, token: str) -> str:
        endpoint = f"/inference/{token}"
        logger.info(f"Requesting GET {self.base_url}{endpoint}")
        try:
            response = self.session.get(self.base_url + endpoint, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Inference for {token} successful")
            return response.text
        except requests.RequestException as e:
            logger.error(f"Inference for {token} failed: {str(e)}")
            return str(e)

    def close(self):
        self.session.close()
        logger.info("Session closed")

def test_client(base_url: str = "http://localhost:8000"):
    client = FlaskAPIClient(base_url)
    print("Checking health:", client.health_check())
    print("\nFetching data:", client.fetch_data())
    print("\nTraining model:", client.train_model())
    print("\nSaving model:", client.save_model())
    print("\nLoading model:", client.load_model())
    print("\nForecasting:", client.forecast())
    print("\nGetting inference for BERA:", client.get_inference("BERA"))
    client.close()

if __name__ == "__main__":
    test_client()