import os
import base64
import pickle
from collections import Counter

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def validate_data(data_b64: str):
    """
    Validates loaded data: checks shape and null counts, logs a short summary.
    Returns the same data_b64 unchanged so the next task receives the same payload.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum()
    n_nulls = null_counts.sum()
    print(f"Validation: shape=({n_rows}, {n_cols}), total nulls={n_nulls}")
    if n_nulls > 0:
        print("Columns with nulls:", null_counts[null_counts > 0].to_dict())
    return data_b64


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    standard_scaler = StandardScaler()
    clustering_data_scaled = standard_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Uses Silhouette score to select optimal k (instead of saving a fixed k=49).
    Returns the SSE list (JSON-serializable) for elbow-method reporting.
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    silhouette_scores = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        if k >= 2:
            silhouette_scores.append(silhouette_score(df, kmeans.labels_))

    # Optimal k = argmax of silhouette score (best cluster separation)
    optimal_k = 2 + silhouette_scores.index(max(silhouette_scores))
    final_kmeans = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
    final_kmeans.fit(df)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(final_kmeans, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model (trained with Silhouette-chosen k) and reports elbow k for comparison.
    Returns the first prediction (as a plain int) for test.csv.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # Model was saved with k chosen by Silhouette score
    print(f"Model clusters (chosen by Silhouette score): {loaded_model.n_clusters}")
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Elbow method suggests k: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred


def report_metrics(model_filename: str, out_filename: str = "cluster_report.txt"):
    """
    Loads the saved model and training data, computes cluster sizes and
    silhouette score, and writes a short text report to working_data/.
    Returns the output file path (JSON-safe string).
    """
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    working_data_dir = os.path.join(project_root, "working_data")
    os.makedirs(working_data_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, "file.csv"))
    df = df.dropna()
    feature_cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(os.path.join(model_dir, model_filename), "rb") as f:
        model = pickle.load(f)
    labels = model.predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sizes = Counter(labels)
    lines = [
        "Cluster metrics report",
        "======================",
        f"n_clusters: {model.n_clusters}",
        f"silhouette_score: {sil:.4f}",
        "cluster sizes:",
    ]
    for c in sorted(sizes.keys()):
        lines.append(f"  cluster {c}: {sizes[c]} samples")
    lines.append(f"total samples: {len(labels)}")
    report = "\n".join(lines)
    out_path = os.path.join(working_data_dir, out_filename)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote report to {out_path}")
    return out_path
