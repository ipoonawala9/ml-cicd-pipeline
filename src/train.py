"""Train a small classifier and enforce an accuracy quality gate.

Exit code 0 = gate passed, 1 = gate failed (this fails the CI job).
"""
import json
import math
import os
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DEFAULT_THRESHOLD = 0.90


def get_threshold() -> float:
    return float(os.getenv("ACCURACY_THRESHOLD", DEFAULT_THRESHOLD))


def build_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train_and_evaluate(seed: int = 42):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = build_model().fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def quality_gate(accuracy: float, threshold: float) -> bool:
    return accuracy >= threshold


def main() -> int:
    threshold = get_threshold()
    model, accuracy = train_and_evaluate()
    passed = quality_gate(accuracy, threshold)
    MODEL_DIR.mkdir(exist_ok=True)
    metrics = {"accuracy": round(accuracy, 4), "threshold": threshold, "passed": passed}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Accuracy: {accuracy:.4f} | Threshold: {threshold:.2f}")
    if not passed:
        print("QUALITY GATE FAILED: accuracy is below the threshold")
        return 1
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print("QUALITY GATE PASSED: model saved to model/model.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
