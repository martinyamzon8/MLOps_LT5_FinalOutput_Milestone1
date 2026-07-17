# MLOps LT5 - Milestone 1: PAL Passenger Data Pipeline

Philippine Airlines currently has no empirical way to classify passengers as
leisure, business, or other travel segments. This project operationalizes a
clustering model that assigns each booking to a market segment based on
historical booking data (flight date, route, cabin, farebrand, POS region,
ticketing channel, purchase lead time, group status, and related fields).
This model has no current MLOps practice around it -- training happens ad
hoc in a data analyst's Jupyter notebook, with manual Excel-based cleaning
and no version control or drift monitoring. Milestone 1 builds the first
piece of that MLOps foundation: an orchestrated, validated, and versioned
data pipeline that will feed the clustering model in later milestones.

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

## Running it locally

Requires [uv](https://docs.astral.sh/uv/) and Docker.

### 1. Install dependencies and run tests

```bash
uv sync
uv run pytest
uv run ruff check .
```

### 2. Start Airflow

```bash
docker compose up --build
```

On first boot, Airflow's `standalone` mode auto-generates an admin user and
prints the password to the terminal (also saved to
`logs/standalone_admin_password.txt`). Once you see `Airflow is ready`,
open [http://localhost:8080](http://localhost:8080) and log in with:

- username: `admin`
- password: (from the terminal output / `logs/standalone_admin_password.txt`)

### 3. Run the pipeline

In the Airflow UI, find the `pal_passenger_data_pipeline` DAG, un-pause it,
and trigger a run (the play button). Watch the three tasks
(`extract -> validate -> load`) turn green in the Graph view.

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
├── dags/
│   ├── pipeline_logic.py     # extract/validate/load functions + Pandera schema
│   └── training_pipeline.py  # Airflow DAG wiring the three tasks together
├── data/
│   ├── raw/                  # sample/aggregated PAL exports (no PII)
│   ├── staging/              # intermediate files between tasks (gitignored)
│   └── processed/            # versioned clean output artifacts (gitignored)
├── tests/
│   └── test_pipeline.py
├── Dockerfile                 # Airflow image + our pipeline dependencies
├── docker-compose.yaml        # Postgres + Airflow (standalone/LocalExecutor)
├── requirements-airflow.txt   # deps installed inside the Airflow container
├── pyproject.toml
├── .pre-commit-config.yaml
└── .gitignore
```

## Data note

The files in `data/raw/` are aggregated exports (PAX counts and average
fares grouped by route/cabin/farebrand/etc.), not passenger-level records,
so no individual PII is present.
