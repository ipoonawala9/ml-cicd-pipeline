"""Minimal prediction script used inside the Docker image."""
import json
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    X, y = load_breast_cancer(return_X_y=True)
    sample = X[:5]
    preds = model.predict(sample).tolist()
    print(json.dumps({"predictions": preds, "actual": y[:5].tolist()}))


if __name__ == "__main__":
    main()
    sys.exit(0)
