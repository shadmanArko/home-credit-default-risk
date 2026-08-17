from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
DESCRIPTION_FILE = "HomeCredit_columns_description.csv"


def inspect_csv(path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    if path.name == DESCRIPTION_FILE:
        sample = pd.read_csv(
            path,
            nrows=5,
            encoding="latin-1",
            sep=";",
        )
    else:
        sample = pd.read_csv(
            path,
            nrows=5,
        )

    print(f"Columns: {len(sample.columns)}")

    print("\nColumns:")
    print(sample.columns.tolist())

    print("\nDtypes:")
    print(sample.dtypes.to_string())

    print("\nFirst 5 rows:")
    print(sample.to_string(index=False))


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")

    print(f"Found {len(csv_files)} CSV files.")

    for path in csv_files:
        inspect_csv(path)


if __name__ == "__main__":
    main()
