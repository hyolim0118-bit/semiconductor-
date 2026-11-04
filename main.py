"""Run data validation, EDA and reproducible train/validation/holdout splits."""

from src.data_loader import PROJECT_ROOT, create_data_splits, validate_csv_files
from src.eda import run_eda


if __name__ == "__main__":
    paths = validate_csv_files(PROJECT_ROOT)
    run_eda(paths, PROJECT_ROOT / "outputs")
    create_data_splits(paths, PROJECT_ROOT / "outputs", random_state=42)
