"""Logging setup for BriefX"""

import logging
import sys
from rich.logging import RichHandler

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup rich handler for beautiful console output
    handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler]
    )
    
    # Reduce noise from third party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)