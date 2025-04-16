# worker.py
import json
import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlloraWorker:
    def __init__(self, config_path="/app/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.worker_config = self.config["worker"][0]  # Assuming one worker
        self.inference_endpoint = self.worker_config["parameters"]["InferenceEndpoint"]
        self.token = self.worker_config["parameters"]["Token"]
        self.loop_seconds = self.worker_config["loopSeconds"]
        self.headers = {'Content-Type': 'application/json'}

    def run_inference(self):
        try:
            # Prepare the payload for the POST request
            payload = {"token": self.token}  # Adjust the payload as needed
            response = requests.post(self.inference_endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info(f"Worker: Inference for {self.token} successful: {response.text}")
            # Simulate submitting to Allora network (placeholder)
            logger.info(f"Worker: Simulated submission to topicId {self.worker_config['topicId']}")
        except Exception as e:
            logger.error(f"Worker: Failed to run inference for {self.token}: {str(e)}")
    def run(self):
        logger.info("Worker: Starting Allora worker")
        while True:
            self.run_inference()
            logger.info(f"Worker: Sleeping for {self.loop_seconds} seconds")
            time.sleep(self.loop_seconds)

if __name__ == "__main__":
    worker = AlloraWorker()
    worker.run()