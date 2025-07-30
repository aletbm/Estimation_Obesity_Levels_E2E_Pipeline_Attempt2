from mlflow.tracking import MlflowClient
from pprint import pprint

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MyCatBoostClassifier"
ARTIFACT_DIR = "monitoring/artifacts"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def list_all_models():
    for rm in client.search_registered_models():
        for alias in rm.aliases:
            pprint(alias)

if __name__ == "__main__":
    list_all_models()