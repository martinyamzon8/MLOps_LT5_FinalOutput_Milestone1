# MLOps LT5 - PAL Passenger Data Pipeline (Milestone 1 + 2)

Philippine Airlines currently has no empirical way to classify passengers as
leisure, business, or other travel segments. This project operationalizes a
clustering model that assigns each booking to a market segment based on
historical booking data (flight date, route, cabin, farebrand, POS region,
ticketing channel, purchase lead time, group status, and related fields).
This model has no current MLOps practice around it -- training happens ad
hoc in a data analyst's Jupyter notebook, with manual Excel-based cleaning
and no version control or drift monitoring. Milestone 1 built an
orchestrated, validated, and versioned data pipeline. Milestone 2 adds
MLflow experiment tracking, an automated pytest suite, and a GitHub
Actions CI workflow on top of it.

## Pipeline

A 3-task Airflow DAG (`dags/training_pipeline.py`):

1. **extract** - reads raw PAL booking exports (`.xlsx`) from `data/raw/`,
   concatenates them, and derives the two calculated fields from the
   problem framing doc: `PurchaseLeadTime` and `Group Status`.
2. **validate** - enforces a Pandera schema (`dags/pipeline_logic.py`) on
   the extracted data. If any row violates a rule (e.g. an invalid Cabin
   code, a negative fare, a PAX Count below 1), the task raises
   `SchemaError` and the DAG run fails -- this is an enforced check, not a
   warning.
3. **load** - writes the validated data as a timestamped, versioned
   parquet file.

## Continuous Integration

![CI](https://github.com/martinyamzon8/MLOps_LT5_FinalOutput_Milestone1/actions/workflows/ci.yml/badge.svg)

Every push and pull request against `main` runs lint (Ruff), the full
pytest suite, and a coverage report via GitHub Actions
(`.github/workflows/ci.yml`, GitHub-hosted `ubuntu-latest` runner). See the
[Actions tab](https://github.com/martinyamzon8/MLOps_LT5_FinalOutput_Milestone1/actions/workflows/ci.yml)
for run history.

## Running it locally

Requires [uv](https://docs.astral.sh/uv/) and Docker.

### 1. Install dependencies

```bash
uv sync
```

### 2. Run the full test suite

```bash
uv run ruff check .
uv run pytest
uv run --with pytest-cov pytest --cov=. --cov-report=term-missing
```

This runs the data validation, model quality, and pipeline integration
tests (`tests/`) and prints a line-coverage report for the pipeline and
training code.

### 3. Start Airflow + the MLflow tracking server

```bash
docker compose up --build
```

This starts three containers: Postgres, Airflow (`standalone` mode), and
MLflow (`mlflow` service, tracking server on port 5001). On first boot,
Airflow auto-generates an admin user and prints the password to the
terminal (also saved to `logs/standalone_admin_password.txt`).

- Airflow UI: [http://localhost:8080](http://localhost:8080)
  (`admin` / password from terminal output)
- MLflow UI: [http://localhost:5001](http://localhost:5001)

The MLflow tracking URI is read from the `MLFLOW_TRACKING_URI` environment
variable (defaults to `http://127.0.0.1:5001` if unset), not hardcoded.
The tracking server uses a SQLite backend
(`BACKEND_STORE_URI=sqlite:////opt/mlflow/mlflow.db`) with artifacts
written to `/opt/mlflow/mlruns`.

### 4. Run the pipeline

In the Airflow UI, find the `pal_passenger_data_pipeline` DAG, un-pause it,
and trigger a run (the play button). Watch the three tasks
(`extract -> validate -> load`) turn green in the Graph view. Each
training/evaluation run also appears as a logged run in the MLflow UI.

To stop everything:

```bash
docker compose down
```

## Output artifact

The `load` task writes to `data/processed/`:

```
data/processed/pal_passengers_clean_<UTC timestamp>.parquet
```

e.g. `pal_passengers_clean_20260717T055021Z.parquet`. Each run produces a
new file, so historical runs are never overwritten -- this is the
"versioned artifact" required by the spec (each run is uniquely
identifiable by its UTC run ID). Intermediate staging files used to pass
data between tasks live in `data/staging/` and are not meant to be
committed (see `.gitignore`).

## Repository structure

```
├── .github/
│   └── workflows/
│       └── ci.yml             # lint + test + coverage on push/PR to main
├── dags/
│   ├── pipeline_logic.py     # extract/validate/load functions + Pandera schema
│   └── training_pipeline.py  # Airflow DAG wiring the three tasks together
├── data/
│   ├── raw/                  # sample/aggregated PAL exports (no PII)
│   ├── staging/              # intermediate files between tasks (gitignored)
│   └── processed/            # versioned clean output artifacts (gitignored)
├── models/
│   └── train.py               # training script with MLflow tracking
├── tests/
│   ├── test_pipeline.py             # data validation tests (Pandera schema)
│   ├── test_model_quality.py        # model quality / threshold tests
│   └── test_pipeline_integration.py # end-to-end pipeline integration test
├── Dockerfile                 # Airflow + MLflow image, our pipeline dependencies
├── docker-compose.yaml        # Postgres + Airflow (standalone) + MLflow
├── requirements-airflow.txt   # deps installed inside the Airflow container
├── pyproject.toml
├── .pre-commit-config.yaml
└── .gitignore
```

## Data note

The files in `data/raw/` are aggregated exports (PAX counts and average
fares grouped by route/cabin/farebrand/etc.), not passenger-level records,
so no individual PII is present.
