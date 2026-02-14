# Airflow lab

- In order to install Airflow using docker you can watch our [Airflow Lab1 Tutorial Video](https://youtu.be/exFSeGUbn4Q?feature=shared)
- For latest step-by-step instructions, check out this blog - [AirFlow Lab-1](https://www.mlwithramin.com/blog/airflow-lab1)

### ML Model

This script is designed for data clustering using K-Means clustering and determining the optimal number of clusters using the elbow method. It provides functionality to load data from a CSV file, perform data preprocessing, build and save a K-Means clustering model, and determine the number of clusters based on the elbow method.

#### Prerequisites

Before using this script, make sure you have the following libraries installed:

- pandas
- scikit-learn (sklearn)
- kneed
- pickle

#### Usage

You can use this script to perform K-Means clustering on your dataset as follows:

```python
# Load the data
data = load_data()

# Preprocess the data
preprocessed_data = data_preprocessing(data)

# Build and save the clustering model
sse_values = build_save_model(preprocessed_data, 'clustering_model.pkl')

# Load the saved model and determine the number of clusters
result = load_model_elbow('clustering_model.pkl', sse_values)
print(result)
```

#### Functions

1. **load_data():**

   - _Description:_ Loads data from a CSV file, serializes it, and returns the serialized data.
   - _Usage:_
     ```python
     data = load_data()
     ```

2. **data_preprocessing(data)**

   - _Description:_ Deserializes data, performs data preprocessing, and returns serialized clustered data.
   - _Usage:_
     ```python
     preprocessed_data = data_preprocessing(data)
     ```

3. **build_save_model(data, filename)**

   - _Description:_ Builds a K-Means clustering model, saves it to a file, and returns SSE values.
   - _Usage:_
     ```python
     sse_values = build_save_model(preprocessed_data, 'clustering_model.pkl')
     ```

4. **load_model_elbow(filename, sse)**
   - _Description:_ Loads a saved K-Means clustering model and determines the number of clusters using the elbow method.
   - _Usage:_
     ```python
     result = load_model_elbow('clustering_model.pkl', sse_values)
     ```

### Airflow Setup

Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

References

- Product - https://airflow.apache.org/
- Documentation - https://airflow.apache.org/docs/
- Github - https://github.com/apache/airflow

#### Installation

Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).

Local

- Docker Desktop Running

Cloud

- Linux VM
- SSH Connection
- Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

#### Tutorial

1. Create a new directory

   ```bash
   mkdir -p ~/app
   cd ~/app
   ```

2. Running Airflow in Docker - [Refer](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-airflow-in-docker)

   a. You can check if you have enough memory by running this command

   ```bash
   docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
   ```

   b. Fetch [docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml)

   ```bash
   curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml'
   ```

   c. Setting the right Airflow user

   ```bash
   mkdir -p ./dags ./logs ./plugins ./working_data
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

   d. Update the following in docker-compose.yml

   ```bash
   # Donot load examples
   AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

   # Additional python package
   _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas }

   # Output dir
   - ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

   # Change default admin credentials
   _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
   _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
   ```

   e. Initialize the database

   ```bash
   docker compose up airflow-init
   ```

   f. Running Airflow

   ```bash
   docker compose up
   ```

   Wait until terminal outputs

   `app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

   g. Enable port forwarding

   h. Visit `localhost:8080` login with credentials set on step `2.d`

3. Explore UI and add user `Security > List Users`

4. Create a python script [`dags/sandbox.py`](dags/sandbox.py)

   - BashOperator
   - PythonOperator
   - Task Dependencies
   - Params
   - Crontab schedules

   You can have n number of scripts inside dags dir

5. Stop docker containers

   ```bash
   docker compose down
   ```

### Airflow DAG Script

This Markdown file provides a detailed explanation of the Python script that defines an Airflow Directed Acyclic Graph (DAG) for a data processing and modeling workflow.

#### Script Overview

The script defines an Airflow DAG named `your_python_dag` that consists of several tasks. Each task represents a specific operation in a data processing and modeling workflow. The script imports necessary libraries, sets default arguments for the DAG, creates PythonOperators for each task, defines task dependencies, and provides command-line interaction with the DAG.

#### Importing Libraries

```python
# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf
```

The script starts by importing the required libraries and modules. Notable imports include the `DAG` and `PythonOperator` classes from the `airflow` package, datetime manipulation functions, and custom functions from the `src.lab` module.

#### Enable pickle support for XCom, allowing data to be passed between tasks

```python
conf.set('core', 'enable_xcom_pickling', 'True')
```

#### Define default arguments for your DAG

```python
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}
```

Default arguments for the DAG are specified in a dictionary named default_args. These arguments include the DAG owner's name, the start date, the number of retries, and the retry delay in case of task failure.

#### Create a DAG instance named 'your_python_dag' with the defined default arguments

```python
dag = DAG(
    'your_python_dag',
    default_args=default_args,
    description='Your Python DAG Description',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)
```

Here, the DAG object dag is created with the name 'your_python_dag' and the specified default arguments. The description provides a brief description of the DAG, and schedule_interval defines the execution schedule (in this case, it's set to None for manual triggering). catchup is set to False to prevent backfilling of missed runs.

#### Task to load data, calls the 'load_data' Python function

```python
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)
```

#### Task to perform data preprocessing, depends on 'load_data_task'

```python
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)
```

The 'data_preprocessing_task' depends on the 'load_data_task' and calls the data_preprocessing function, which is provided with the output of the 'load_data_task'.

#### Task to build and save a model, depends on 'data_preprocessing_task'

```python
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)
```

The 'build_save_model_task' depends on the 'data_preprocessing_task' and calls the build_save_model function. It also provides additional context information and arguments.

#### Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'

```python
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)
```

The 'load_model_task' depends on the 'build_save_model_task' and calls the load_model_elbow function with specific arguments.

#### Set task dependencies

```python
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
```

Task dependencies are defined using the >> operator. In this case, the tasks are executed in sequence: 'load_data_task' -> 'data_preprocessing_task' -> 'build_save_model_task' -> 'load_model_task'.

#### If this script is run directly, allow command-line interaction with the DAG

```python
if __name__ == "__main__":
    dag.cli()
```

- Lastly, the script allows for command-line interaction with the DAG. When the script is run directly, the dag.cli() function is called, providing the ability to trigger and manage the DAG from the command line.
- This script defines a comprehensive Airflow DAG for a data processing and modeling workflow, with clear task dependencies and default arguments.

### Running an Apache Airflow DAG Pipeline in Docker

This section describes how to set up and run the **Airflow_Lab1** DAG in Docker and what the run looks like (including the two submission screenshots).

#### Prerequisites

- **Docker Desktop** installed and running (allocate at least 4GB RAM, ideally 8GB).
- Terminal opened in the **Lab_1** directory.

#### Step 1: Directory structure

The project layout:

```plaintext
Lab_1/
├── dags/
│   ├── airflow.py          # DAG definition
│   ├── src/
│   │   └── lab.py          # ML pipeline (load, validate, preprocess, model, report)
│   └── data/
│       ├── file.csv        # training data
│       └── test.csv        # test data
├── working_data/           # DAG output (e.g. cluster_report.txt)
├── docker-compose.yaml     # Airflow + Postgres
├── setup.sh                # Professor’s one-time setup script
├── ss_1.png                # Screenshot: successful DAG run (all tasks green)
└── ss_2.png                # Screenshot: DAG Graph view (6 tasks and dependencies)
```

#### Step 2: One-time setup (professor’s script)

From the **Lab_1** directory run:

```bash
bash setup.sh
```

This removes old `.env`, `logs/`, `plugins/`, `config/`, runs `docker compose down -v`, recreates `logs`, `plugins`, `config`, `working_data`, writes `AIRFLOW_UID` to `.env`, and runs `docker compose run --rm airflow-cli airflow config list` (first time may take a while while the DB initializes). The included `docker-compose.yaml` mounts `./dags` and `./working_data`, sets `LOAD_EXAMPLES: false`, and adds `pandas`, `scikit-learn`, and `kneed`.

#### Step 3: Start Airflow

```bash
docker compose up
```

Wait until the webserver is ready (e.g. log line with `GET /health HTTP/1.1" 200`). Leave this terminal running.

#### Step 4: Open the Airflow UI

1. In your browser go to **http://localhost:8080**.
2. Log in with **Username:** `airflow`, **Password:** `airflow`.

#### Step 5: Run the DAG and capture outputs

1. On the DAGs page, find **Airflow_Lab1** and turn it **ON** if needed.
2. Open the **Graph** tab to see the six tasks and their order:  
   `load_data_task` → `validate_data_task` → `data_preprocessing_task` → `build_save_model_task` → `load_model_task` → `report_metrics_task`.
3. Click **Trigger DAG** (play button) and wait for the run to finish (all tasks green). The pipeline may take 1–2 minutes because of model training.
4. After a successful run, the saved model appears under `dags/model/` and the metrics report under `working_data/cluster_report.txt`.

#### Step 6: Stop Airflow

In the terminal where `docker compose up` is running, press **Ctrl+C**, then:

```bash
docker compose down
```

To also remove the database volume: `docker compose down -v`.

---

### Changes made (Lab submission)

This lab is not identical to the original repo. Below is a detailed list of all modifications.

---

#### 1. Preprocessing: MinMaxScaler → StandardScaler

**File:** `dags/src/lab.py`  
**Function:** `data_preprocessing()`

- **Original behaviour:** Features `BALANCE`, `PURCHASES`, and `CREDIT_LIMIT` were scaled with **MinMaxScaler** (values in [0, 1]).
- **Change:** MinMaxScaler was replaced with **StandardScaler**.
- **Reason:** StandardScaler centres the data (zero mean) and scales by standard deviation (unit variance). This is often better when features have different units or when we care about relative spread rather than a fixed [0, 1] range.
- **Code:** `MinMaxScaler()` and `fit_transform(clustering_data)` were replaced with `StandardScaler()` and `fit_transform(clustering_data)`; variable names were updated accordingly (e.g. `clustering_data_scaled`).

---

#### 2. Model selection: fixed k=49 → Silhouette-based optimal k

**File:** `dags/src/lab.py`  
**Function:** `build_save_model()`

- **Original behaviour:** The loop fitted KMeans for k = 1, 2, …, 49; the **last** model (k = 49) was saved. The elbow method was used only for reporting.
- **Change:** The pipeline now uses the **Silhouette score** to choose the number of clusters:
  - For each k from 2 to 49, KMeans is fitted and the Silhouette score is computed (k = 1 is skipped because Silhouette is undefined for a single cluster).
  - The k with the **maximum** Silhouette score is selected.
  - A **final** KMeans model is fitted with this optimal k and **that** model is saved (not k = 49).
- **Reason:** Silhouette score measures how well separated and compact the clusters are; maximising it gives a data-driven choice of k instead of saving an arbitrary large k.
- **Code:** `silhouette_score` from `sklearn.metrics` is used; after the loop, `optimal_k` is set to `2 + silhouette_scores.index(max(silhouette_scores))`, and `final_kmeans = KMeans(n_clusters=optimal_k, ...)` is fitted and saved. The function still returns the SSE list so the elbow method can be reported elsewhere.

**Function:** `load_model_elbow()`

- **Change:** Logging was updated to reflect that the loaded model was trained with Silhouette-chosen k. It now prints both “Model clusters (chosen by Silhouette score): &lt;k&gt;” and “Elbow method suggests k: &lt;k_elbow&gt;” for comparison.

---

#### 3. DAG structure: two new tasks

**File:** `dags/airflow.py`

The DAG was extended with two additional tasks so the workflow is different from the original and from a “single model + predictions CSV” style pipeline.

**New task 1 — `validate_data_task`**

- **Placement:** Runs immediately after `load_data_task` and before `data_preprocessing_task`.
- **Purpose:** Data validation step: decode the base64 payload from load_data, inspect the DataFrame (shape and null counts), and log a short summary (e.g. “Validation: shape=(8950, 18), total nulls=314” and which columns have nulls). The same payload is returned unchanged so the next task receives the same data.
- **Implementation:** New function `validate_data(data_b64)` in `dags/src/lab.py`. It decodes the pickled DataFrame, runs `df.shape` and `df.isnull().sum()`, prints the summary, and returns the original `data_b64` string.
- **Effect:** The DAG now has an explicit data-quality check before preprocessing.

**New task 2 — `report_metrics_task`**

- **Placement:** Runs after `load_model_task` (last task in the pipeline).
- **Purpose:** Produce a small **text report** of clustering metrics (not a full predictions CSV). It loads the saved KMeans model and the training data, applies the same StandardScaler preprocessing, predicts cluster labels, computes the Silhouette score and per-cluster sample counts, and writes a report to `working_data/cluster_report.txt`.
- **Implementation:** New function `report_metrics(model_filename, out_filename)` in `dags/src/lab.py`. It reads `file.csv`, drops nulls, scales the three features with StandardScaler, loads the model from `dags/model/`, runs `model.predict()`, computes `silhouette_score()` and cluster sizes (e.g. with `Counter`), and writes lines such as “n_clusters”, “silhouette_score”, “cluster sizes”, “total samples” to a file under `working_data/` (path derived from project root so it works in Docker and locally).
- **Effect:** The pipeline produces (1) the saved model in `dags/model/` and (2) a metrics report file; the artifact is a short report, not a large predictions CSV.

**Updated task order**

- **Original:** `load_data_task` → `data_preprocessing_task` → `build_save_model_task` → `load_model_task`
- **New:**  
  `load_data_task` → **`validate_data_task`** → `data_preprocessing_task` → `build_save_model_task` → `load_model_task` → **`report_metrics_task`**

**DAG code changes:** In `airflow.py`, `validate_data` and `report_metrics` are imported from `src.lab`; `validate_data_task` is added with `op_args=[load_data_task.output]`; `data_preprocessing_task` now takes `validate_data_task.output` instead of `load_data_task.output`; `report_metrics_task` is added with `op_args=["model.sav", "cluster_report.txt"]`; the dependency chain is updated to the order above.

---

#### 4. Summary of files and functions modified

| Location          | Change                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `dags/src/lab.py` | Use StandardScaler instead of MinMaxScaler in `data_preprocessing()`.                                                     |
| `dags/src/lab.py` | In `build_save_model()`, add Silhouette-based choice of k and save the model with that k; return SSE for elbow reporting. |
| `dags/src/lab.py` | In `load_model_elbow()`, update logs to show Silhouette-chosen k and elbow k.                                             |
| `dags/src/lab.py` | Add `validate_data(data_b64)` for shape and null checks.                                                                  |
| `dags/src/lab.py` | Add `report_metrics(model_filename, out_filename)` to write `working_data/cluster_report.txt`.                            |
| `dags/airflow.py` | Add `validate_data_task` and `report_metrics_task`; wire new dependencies and update `data_preprocessing_task` input.     |

---

#### 5. Screenshots (submission)

Two screenshots in the Lab_1 root document a successful run of the modified DAG:

- **ss_1** — Successful run: Grid or Graph view with **Airflow_Lab1** and all tasks in **success** (green) for one run.

  ![Successful DAG run (ss_1)](https://github.com/RohitKulkarni02/MLOps/blob/main/Labs/Airflow_Labs/Lab_1/ss_1.jpeg)

- **ss_2** — Graph view: The **Airflow_Lab1** DAG in **Graph** view, showing all six tasks and their dependencies.

  ![DAG Graph view (ss_2)](https://github.com/RohitKulkarni02/MLOps/blob/main/Labs/Airflow_Labs/Lab_1/ss_2.jpeg)
