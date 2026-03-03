import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "cortex.parquet"
PROPERTIES_PATH = Path(__file__).parent.parent.parent / "data" / "properties.json"
MODEL_NAME = "gpt-4o"


class PropertyMeta(TypedDict):
    address: str
    current_value: int
    purchase_price: int
    last_appraisal_date: str


def _load_portfolio_metadata() -> tuple[
    list[str], set[str], set[str], dict[str, PropertyMeta]
]:
    df = pd.read_parquet(DATA_PATH)

    property_names: list[str] = sorted(
        df["property_name"].dropna().unique().tolist()
    )
    valid_years: set[str] = set(df["year"].dropna().unique().tolist())
    valid_quarters: set[str] = set(df["quarter"].dropna().unique().tolist())

    with open(PROPERTIES_PATH) as f:
        raw: dict[str, dict] = json.load(f)

    property_metadata: dict[str, PropertyMeta] = {
        name: PropertyMeta(
            address=meta["address"],
            current_value=int(meta["current_value"]),
            purchase_price=int(meta["purchase_price"]),
            last_appraisal_date=meta["last_appraisal_date"],
        )
        for name, meta in raw.items()
        if name in property_names
    }

    return property_names, valid_years, valid_quarters, property_metadata


PROPERTY_NAMES, VALID_YEARS, VALID_QUARTERS, PROPERTY_METADATA = (
    _load_portfolio_metadata()
)
