import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import meta_training
from prop_analyzer.utils import common

def main():
    """
    Execution script for the Error-Prediction Meta-Model.
    Runs the XGBoost calibrator training over graded history.
    """
    common.setup_logging(name="run_meta_training_script")
    meta_training.train_meta_classifier()

if __name__ == "__main__":
    main()