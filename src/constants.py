# ==============================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# ==============================================================

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "dataset_cell_growth_4factor.csv"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"

TARGET = "Specific Growth Rate [d^-1]"
FEATURES = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

POLY_DEGREES = [1, 2, 3, 4, 5]
KNN_NEIGHBORS = [1, 3, 5, 10, 15, 20, 30, 50]
