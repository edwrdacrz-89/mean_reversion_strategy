"""
One-shot signal runner for Windows Task Scheduler.
Runs once, generates signals, saves to track_record.json, then exits.
"""

import os
import sys

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import and run
from signal_service import run_signal_generation, logger

if __name__ == "__main__":
    logger.info("Task Scheduler triggered signal generation")
    run_signal_generation()
    logger.info("Done")
