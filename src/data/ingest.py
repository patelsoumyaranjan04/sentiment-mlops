"""
src/data/ingest.py
------------------
Reads the raw Amazon_review.csv, validates schema,
logs basic statistics, and saves a clean copy to data/raw/.

Run:
    python -m src.data.ingest
"""

import sys
from pathlib import Path
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("ingest")


REQUIRED_COLUMNS = {"review", "sentiment"}
VALID_SENTIMENTS = {0, 1}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    invalid = set(df["sentiment"].dropna().unique()) - VALID_SENTIMENTS
    if invalid:
        raise ValueError(f"Unexpected sentiment values: {invalid}")
    logger.info("Schema validation passed.")


def log_statistics(df: pd.DataFrame) -> dict:
    stats = {
        "total_rows": len(df),
        "null_review": int(df["review"].isna().sum()),
        "null_sentiment": int(df["sentiment"].isna().sum()),
        "duplicates": int(df.duplicated().sum()),
        "positive": int((df["sentiment"] == 1).sum()),
        "negative": int((df["sentiment"] == 0).sum()),
    }
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    return stats


def ingest(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    raw_path = Path(cfg["data"]["raw_file"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}\n"
            "Place Amazon_review.csv in data/raw/ before running the pipeline."
        )

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    # Keep only required columns
    df = df[["review", "sentiment"]]

    validate_schema(df)
    stats = log_statistics(df)

    # Drop nulls and duplicates
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} null/duplicate rows. Remaining: {len(df)}")

    # Ensure correct dtypes
    df["sentiment"] = df["sentiment"].astype(int)

    # Save validated raw snapshot
    out_path = raw_path.parent / "Amazon_review_validated.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Validated raw data saved to {out_path}")

    return df


if __name__ == "__main__":
    ingest()
