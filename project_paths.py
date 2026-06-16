from pathlib import Path

# Root folder of the whole project
DIR = Path(__file__).resolve().parent

DATA_DIR = DIR / "1. Data"
MOMENTS_DIR = DATA_DIR / "momenter"

RESULTS_DIR = DIR / "4. Results"
FIRST_STAGE_RESULTS_DIR = RESULTS_DIR / "first_stage estimation"
STRUCTURAL_RESULTS_DIR = RESULTS_DIR / "Structural estimation"
SIM_RESULTS_DIR = RESULTS_DIR / "Simulation"
COMPARISON_DIR = RESULTS_DIR / "Comparison"

FIRST_STAGE_DIR = DIR / "first_stage"

PLOTS_DIR = DIR / RESULTS_DIR / "plots"
SIM_PLOTS_DIR = PLOTS_DIR / "simulation"

MODEL_FUNCTIONS_DIR = DIR / "0. Functions"