import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score


def _seed_from_timestamp(timestamp: str) -> int:
    h = hashlib.sha256(timestamp.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def make_synthetic_dataset_from_timestamp(
    timestamp: str,
    *,
    n_features: int = 6,
    n_informative: int = 3,
    min_samples: int = 100,
    max_samples: int = 2000,
):
    seed = _seed_from_timestamp(timestamp)
    rng = random.Random(seed)
    n_samples = rng.randint(min_samples, max_samples)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=True,
    )
    return X, y


def evaluate_model_and_save(
    timestamp: str,
    *,
    base_dir: Optional[Path] = None,
):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    models_dir = base_dir / "models"
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"model_{timestamp}_dt_model.joblib"
    model = joblib.load(model_path)

    X, y = make_synthetic_dataset_from_timestamp(timestamp)
    y_predict = model.predict(X)

    metrics = {"F1_Score": f1_score(y, y_predict)}
    metrics_path = metrics_dir / f"{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics, str(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    metrics, metrics_path = evaluate_model_and_save(args.timestamp, base_dir=None)
    print(f"Saved metrics to {metrics_path}: {metrics}")


if __name__ == "__main__":
    main()
