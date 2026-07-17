"""
Airflow DAG: PAL passenger data pipeline (Milestone 1)

Reads the weekly PAL domestic (Manila Hub) booking exports, validates them
against a Pandera schema, and writes a clean, versioned parquet artifact
that downstream clustering/training work (Milestone 2+) will consume.

Tasks:
    extract  -> pull raw xlsx exports from data/raw, stage as parquet
    validate -> enforce Pandera schema; task fails if data is invalid
    load     -> write timestamped, versioned parquet artifact

Design note: tasks pass *file paths* through XCom rather than full
dataframes. XCom is backed by the Airflow metadata database and is meant
for small values -- pushing ~140k rows of JSON through it would work at
this scale but is the wrong pattern and won't survive larger data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from airflow.decorators import dag, task
from pipeline_logic import extract_data, load_data, validate_data

STAGING_DIR = Path(__file__).resolve().parent.parent / "data" / "staging"


@dag(
    dag_id="pal_passenger_data_pipeline",
    description="Extract, validate, and load PAL passenger booking data for market segmentation",
    schedule="@weekly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "milestone1", "pal"],
)
def pal_passenger_data_pipeline():
    @task()
    def extract() -> str:
        df = extract_data()
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staged_path = STAGING_DIR / "extracted.parquet"
        df.to_parquet(staged_path, index=False)
        return str(staged_path)

    @task()
    def validate(staged_path: str) -> str:
        import pandas as pd

        df = pd.read_parquet(staged_path)
        validated = validate_data(df)  # raises SchemaError on bad data
        validated_path = STAGING_DIR / "validated.parquet"
        validated.to_parquet(validated_path, index=False)
        return str(validated_path)

    @task()
    def load(validated_path: str) -> str:
        import pandas as pd

        df = pd.read_parquet(validated_path)
        out_path = load_data(df)
        print(f"Wrote clean artifact to {out_path}")
        return str(out_path)

    load(validate(extract()))


pal_passenger_data_pipeline()
