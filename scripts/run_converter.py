import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import parser
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="converter")
    
    try:
        logging.info(">>> STARTING PROP CONVERTER <<<")
        
        # Execute parsing (Logic moved to parser.py for better encapsulation)
        parser.parse_text_to_csv()
        
        logging.info("<<< CONVERTER COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Converter: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()