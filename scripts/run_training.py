import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import training
from prop_analyzer.utils import common

def main():
    """
    Execution script for the Advanced Prop Training Pipeline.
    Delegates process execution to the updated training module.
    """
    common.setup_logging(name="run_training_script")
    training.main()

if __name__ == "__main__":
    main()