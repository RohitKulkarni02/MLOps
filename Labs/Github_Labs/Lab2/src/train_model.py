import argparse
import datetime
import hashlib
import random
import pickle
from pathlib import Path
from typing import Optional

from joblib import dump
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def _seed_from_timestamp(timestamp: str) -> int:
    # Deterministic seed so scheduled workflows produce stable artifacts for a given timestamp.
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


def train_model_and_save(
    timestamp: str,
    *,
    base_dir: Optional[Path] = None,
    use_mlflow: bool = True,
    rf_random_state: int = 0,
):
    """
    Trains a RandomForest model on a deterministic synthetic dataset and saves artifacts into:
      - data/
      - models/
      - mlruns/ (when use_mlflow=True)
    """
    if base_dir is None:
        # Lab2 root: src/.. (i.e., Labs/Github_Labs/Lab2/)
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    mlruns_dir = base_dir / "mlruns"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_synthetic_dataset_from_timestamp(timestamp)

    # Persist the dataset so the run is reproducible and inspectable.
    with open(data_dir / "data.pickle", "wb") as f:
        pickle.dump(X, f)
    with open(data_dir / "target.pickle", "wb") as f:
        pickle.dump(y, f)

    forest = RandomForestClassifier(random_state=rf_random_state)
    forest.fit(X, y)

    y_predict = forest.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_predict),
        "F1 Score": f1_score(y, y_predict),
    }

    model_version = f"model_{timestamp}"
    model_path = models_dir / f"{model_version}_dt_model.joblib"
    dump(forest, model_path)

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(str(mlruns_dir))
        dataset_name = "Reuters Corpus Volume"
        current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        experiment_name = f"{dataset_name}_{current_time}"
        experiment_id = mlflow.create_experiment(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
            mlflow.log_params(
                {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1],
                    "timestamp": timestamp,
                }
            )
            mlflow.log_metrics(metrics)

    # Return metrics for potential unit tests / debugging.
    return metrics, str(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    print(f"Timestamp received from GitHub Actions: {args.timestamp}")
    train_model_and_save(args.timestamp, use_mlflow=True)


if __name__ == "__main__":
    main()
