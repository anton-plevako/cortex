from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "cortex.parquet"
MODEL_NAME = "gpt-4o"
SQL_MODEL_NAME = "gpt-5.2"


def _load_portfolio_metadata() -> tuple[list[str], set[str], set[str]]:
    df = pd.read_parquet(DATA_PATH)
    property_names: list[str] = sorted(df["property_name"].dropna().unique().tolist())
    valid_years: set[str] = set(df["year"].dropna().unique().tolist())
    valid_quarters: set[str] = set(df["quarter"].dropna().unique().tolist())
    return property_names, valid_years, valid_quarters


PROPERTY_NAMES, VALID_YEARS, VALID_QUARTERS = _load_portfolio_metadata()
