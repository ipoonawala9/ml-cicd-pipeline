"""Print model metrics as a Markdown table for the GitHub Actions job summary."""
import json
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "model" / "metrics.json"


def main() -> None:
    m = json.loads(METRICS.read_text())
    status = "PASSED" if m["passed"] else "FAILED"
    print("## Model quality gate")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Accuracy | {m['accuracy']} |")
    print(f"| Threshold | {m['threshold']} |")
    print(f"| Gate | {status} |")


if __name__ == "__main__":
    main()
