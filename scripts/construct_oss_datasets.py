"""Construct monthly cumulative OSS fault-count datasets.

This script reconstructs the monthly cumulative fault-count sequences used in the
paper from the cleaned raw GitHub issue records.

Input
-----
By default, the script reads:

    data/IST_OSS_datasets_clean_release.xlsx

The workbook is expected to contain a sheet named ``Raw_Issue_Records`` with at
least the following columns:

    Dataset ID, Project, Repository, Issue Number, Created At

Output
------
The script writes:

    data/processed_monthly/OSS1_redis_monthly.csv
    data/processed_monthly/OSS2_wox_monthly.csv
    ...
    data/processed_monthly/OSS6_chartjs_monthly.csv
    data/processed_monthly/monthly_cumulative_data_reconstructed.csv

Each output CSV contains monthly fault counts and cumulative fault counts.
Months with no newly reported fault-related issues are retained with zero
increments.

Usage
-----
From the repository root:

    python scripts/construct_oss_datasets.py

Optional arguments:

    python scripts/construct_oss_datasets.py \
        --input data/IST_OSS_datasets_clean_release.xlsx \
        --output-dir data/processed_monthly
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


RAW_SHEET_NAME = "Raw_Issue_Records"
MONTHLY_SHEET_NAME = "Monthly_Cumulative_Data"

# Expected order in the paper.
DATASET_ORDER = ["OSS1", "OSS2", "OSS3", "OSS4", "OSS5", "OSS6"]

# File-name suffixes used for per-dataset CSV files.
DATASET_SLUGS: Dict[str, str] = {
    "OSS1": "redis",
    "OSS2": "wox",
    "OSS3": "backbone",
    "OSS4": "brew",
    "OSS5": "pytorch",
    "OSS6": "chartjs",
}


def normalize_column_name(name: str) -> str:
    """Normalize a column name for robust matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert expected Excel column names into snake_case names."""
    column_map = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=column_map)

    required_columns = {
        "dataset_id",
        "project",
        "repository",
        "issue_number",
        "created_at",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "The Raw_Issue_Records sheet is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return df


def full_month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.PeriodIndex:
    """Return a continuous calendar-month range from start to end, inclusive."""
    start_month = start.to_period("M")
    end_month = end.to_period("M")
    return pd.period_range(start_month, end_month, freq="M")


def construct_monthly_counts(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Construct monthly and cumulative counts for all OSS datasets."""
    raw_df = rename_columns(raw_df)
    raw_df = raw_df.copy()

    raw_df["dataset_id"] = raw_df["dataset_id"].astype(str).str.strip()
    raw_df["project"] = raw_df["project"].astype(str).str.strip()
    raw_df["repository"] = raw_df["repository"].astype(str).str.strip()
    raw_df["created_at"] = pd.to_datetime(raw_df["created_at"], errors="coerce")

    invalid_dates = raw_df["created_at"].isna().sum()
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} records with invalid Created At values.")

    # Remove duplicate issue numbers within each dataset.
    # This protects the construction procedure even if the raw file contains
    # repeated issue records.
    raw_df = raw_df.drop_duplicates(subset=["dataset_id", "issue_number"])

    records = []
    dataset_ids: Iterable[str] = [d for d in DATASET_ORDER if d in set(raw_df["dataset_id"])]

    for dataset_id in dataset_ids:
        group = raw_df.loc[raw_df["dataset_id"] == dataset_id].copy()
        group = group.sort_values("created_at")

        project = group["project"].iloc[0]
        repository = group["repository"].iloc[0]

        months = full_month_range(group["created_at"].min(), group["created_at"].max())
        issue_months = group["created_at"].dt.to_period("M")
        monthly_counts = issue_months.value_counts().sort_index()
        monthly_counts = monthly_counts.reindex(months, fill_value=0).astype(int)
        cumulative_counts = monthly_counts.cumsum().astype(int)

        for month, monthly_count in monthly_counts.items():
            records.append(
                {
                    "Dataset ID": dataset_id,
                    "Project": project,
                    "Repository": repository,
                    "Month": str(month),
                    "Monthly Fault Count": int(monthly_count),
                    "Cumulative Fault Count": int(cumulative_counts.loc[month]),
                }
            )

    if not records:
        raise ValueError("No recognized OSS dataset IDs were found in the raw records.")

    return pd.DataFrame(records)


def write_outputs(monthly_df: pd.DataFrame, output_dir: Path) -> None:
    """Write combined and per-dataset monthly CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "monthly_cumulative_data_reconstructed.csv"
    monthly_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    for dataset_id in DATASET_ORDER:
        sub = monthly_df.loc[monthly_df["Dataset ID"] == dataset_id].copy()
        if sub.empty:
            continue
        slug = DATASET_SLUGS.get(dataset_id, dataset_id.lower())
        path = output_dir / f"{dataset_id}_{slug}_monthly.csv"
        sub.to_csv(path, index=False, encoding="utf-8-sig")


def compare_with_workbook(input_path: Path, reconstructed_df: pd.DataFrame) -> None:
    """Compare reconstructed monthly data with the workbook sheet, if available."""
    try:
        expected_df = pd.read_excel(input_path, sheet_name=MONTHLY_SHEET_NAME)
    except Exception:
        print(f"[INFO] Sheet '{MONTHLY_SHEET_NAME}' was not found. Skipping comparison.")
        return

    common_cols = [
        "Dataset ID",
        "Project",
        "Repository",
        "Month",
        "Monthly Fault Count",
        "Cumulative Fault Count",
    ]
    expected_df = expected_df[common_cols].copy()
    reconstructed_df = reconstructed_df[common_cols].copy()

    # Normalize strings to avoid false mismatches caused by spreadsheet types.
    for col in ["Dataset ID", "Project", "Repository", "Month"]:
        expected_df[col] = expected_df[col].astype(str).str.strip()
        reconstructed_df[col] = reconstructed_df[col].astype(str).str.strip()

    expected_df = expected_df.sort_values(["Dataset ID", "Month"]).reset_index(drop=True)
    reconstructed_df = reconstructed_df.sort_values(["Dataset ID", "Month"]).reset_index(drop=True)

    if expected_df.equals(reconstructed_df):
        print("[OK] Reconstructed monthly data matches the Monthly_Cumulative_Data sheet.")
    else:
        print("[WARNING] Reconstructed monthly data does not exactly match the workbook sheet.")
        print(f"          Expected rows:      {len(expected_df)}")
        print(f"          Reconstructed rows: {len(reconstructed_df)}")

        merged = expected_df.merge(
            reconstructed_df,
            on=["Dataset ID", "Project", "Repository", "Month"],
            how="outer",
            suffixes=("_expected", "_reconstructed"),
            indicator=True,
        )
        diff = merged.loc[
            (merged["_merge"] != "both")
            | (
                merged["Monthly Fault Count_expected"]
                != merged["Monthly Fault Count_reconstructed"]
            )
            | (
                merged["Cumulative Fault Count_expected"]
                != merged["Cumulative Fault Count_reconstructed"]
            )
        ]
        if not diff.empty:
            print("[WARNING] First mismatching rows:")
            print(diff.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct monthly cumulative OSS fault-count datasets from cleaned raw issue records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/IST_OSS_datasets_clean_release.xlsx"),
        help="Path to IST_OSS_datasets_clean_release.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_monthly"),
        help="Directory where reconstructed CSV files will be saved",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Do not compare reconstructed data with the Monthly_Cumulative_Data sheet in the workbook",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run this script from the repository root or pass --input explicitly."
        )

    raw_df = pd.read_excel(args.input, sheet_name=RAW_SHEET_NAME)
    monthly_df = construct_monthly_counts(raw_df)
    write_outputs(monthly_df, args.output_dir)

    print(f"[OK] Wrote reconstructed monthly datasets to: {args.output_dir}")
    print("[OK] Generated files:")
    print(f"     - {args.output_dir / 'monthly_cumulative_data_reconstructed.csv'}")
    for dataset_id in DATASET_ORDER:
        slug = DATASET_SLUGS.get(dataset_id, dataset_id.lower())
        path = args.output_dir / f"{dataset_id}_{slug}_monthly.csv"
        if path.exists():
            print(f"     - {path}")

    summary = (
        monthly_df.groupby(["Dataset ID", "Project"], as_index=False)
        .agg(
            Monthly_Observations=("Month", "count"),
            Final_Cumulative_Fault_Count=("Cumulative Fault Count", "max"),
        )
        .sort_values("Dataset ID")
    )
    print("\nDataset summary:")
    print(summary.to_string(index=False))

    if not args.no_compare:
        compare_with_workbook(args.input, monthly_df)


if __name__ == "__main__":
    main()
