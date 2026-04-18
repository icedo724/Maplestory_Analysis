import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# exp_analysis/dashboard/config.py → BASE_DIR = maple/
BASE_DIR    = os.path.dirname(os.path.dirname(current_dir))

AGG_DIR       = os.path.join(BASE_DIR, "data", "processed", "showcase", "aggregated")
SHOWCASE_DATE = "2025-12-13"
