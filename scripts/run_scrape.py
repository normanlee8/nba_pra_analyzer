import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import scraper
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="scraper")
    
    try:
        logging.info(">>> STARTING DATA SCRAPER <<<")
        
        # Execute scraping logic
        scraper.main()
        
        logging.info("<<< SCRAPER COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Scraper: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()