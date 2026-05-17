"""
config.py

Shared lightweight configuration for the LinkedIn Job Analysis project.
Existing scripts do not need to be refactored immediately; new scripts and
future cleanup can import these values to avoid repeated hard-coded paths.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

SALARY_MIN = 10_000
SALARY_MAX = 300_000
CURRENCY = "USD"
RANDOM_STATE = 42

MODEL03_PATH = MODELS_DIR / "best_salary_model_03_advanced_progress.joblib"
MODEL04_PATH = MODELS_DIR / "best_salary_model_04_stacking_intervals.joblib"

MODEL04_METRICS_PATH = OUTPUT_DIR / "model_04_metrics.json"
MODEL04_INTERVAL_SCORES_PATH = OUTPUT_DIR / "model_04_interval_method_scores.csv"
MODEL04_COMPARISON_PATH = OUTPUT_DIR / "model_04_comparison_metrics.csv"
