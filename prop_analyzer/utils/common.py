import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(name=None, level=logging.INFO):
    """
    Configures standard logging for the application.
    Logs to console and a file in the local directory.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Determine log filename
    if name:
        filename = log_dir / f"{name}.log"
    else:
        filename = log_dir / "app.log"
        
    # Formatting
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Handlers
    handlers = [
        logging.StreamHandler(sys.stdout), # Console
        logging.FileHandler(filename, encoding='utf-8') # File
    ]
    
    # Configure
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True # Overwrite any existing config
    )
    
    logging.info(f"Logging initialized. Writing to {filename}")

def get_nba_season_id(date_obj):
    """
    Returns the NBA API Season ID string for a given date.
    Format: '2' + Year (e.g., '22025' for the 2025-26 Regular Season).
    
    Logic:
    - If month is Oct (10) or later, the season started this year.
    - If month is before Oct, the season started the previous year.
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        except:
            # Fallback to current date if parsing fails
            date_obj = datetime.now()
            
    # NBA Season cutover usually happens in October
    if date_obj.month >= 10:
        season_start_year = date_obj.year
    else:
        season_start_year = date_obj.year - 1
        
    # '2' prefix denotes Regular Season in NBA API
    return f"2{season_start_year}"

def get_season_year_str(date_obj):
    """
    Returns the season string like '2025-26' for filenames.
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        except:
            date_obj = datetime.now()

    if date_obj.month >= 10:
        start = date_obj.year
        end = date_obj.year + 1
    else:
        start = date_obj.year - 1
        end = date_obj.year
        
    return f"{start}-{str(end)[-2:]}"