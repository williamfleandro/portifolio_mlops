import os
import mlflow

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

print("Tracking URI:", mlflow.get_tracking_uri())