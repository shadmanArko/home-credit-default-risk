from pathlib import Path

import kagglehub

COMPETITION = "home-credit-default-risk"
DATA_DIR = Path("data/raw")
COMPLETE_MARKER = DATA_DIR / ".complete"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if COMPLETE_MARKER.exists():
        print(f"Dataset already exists at: {DATA_DIR}")
        return

    path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(DATA_DIR),
    )

    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
