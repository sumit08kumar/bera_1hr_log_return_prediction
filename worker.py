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
        self.topic_id = self.worker_config["topicId"]
        self.wallet_address = self.config["wallet"]["addressKeyName"]

    def run_inference(self):
        try:
            # Prepare the payload for the POST request
            payload = {"token": self.token}
            logger.info(f"Worker: Sending inference request to {self.inference_endpoint}")
            response = requests.post(self.inference_endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info(f"Worker: Inference for {self.token} successful: {response.text}")

        # Simulate building the worker payload
            block_height = 3475517  # Replace with actual block height logic
            worker_payload = {
                "worker": self.wallet_address,
                "nonce": {"block_height": block_height},
                "topic_id": self.topic_id,
                "inference_forecasts_bundle": {
                    "inference": {
                    "topic_id": self.topic_id,
                    "block_height": block_height,
                    "inferer": self.wallet_address,
                    "value": "-0.003"  # Replace with actual inference value
                    }
                },
                "inferences_forecasts_bundle_signature": "...",  # Replace with actual signature logic
                "pubkey": "..."  # Replace with actual public key logic
            }

            logger.info(f"Worker: Building worker payload for topicId {self.topic_id}")
            logger.info(f"Worker: Payload: {json.dumps(worker_payload)}")

            # Simulate sending the payload to the chain
            logger.info(f"Worker: Sending InsertWorkerPayload to chain")
            # Add actual logic to send the payload to the chain here

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