#!/usr/bin/env python
"""
Automated Trading Scheduler

This script sets up a scheduled task to run the trading system
at specified intervals (e.g., daily at market open).
"""

import argparse
import os
import time
from datetime import datetime, timedelta
import schedule
import subprocess
import logging
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trading_scheduler")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Schedule automated trading tasks.")
    
    parser.add_argument("--time", type=str, default="09:30",
                      help="Time to run trading (HH:MM in 24-hour format, default: 09:30)")
    parser.add_argument("--tickers", type=str,
                      help="Comma-separated list of ticker symbols (overrides config)")
    parser.add_argument("--capital", type=float,
                      help="Initial capital")
    parser.add_argument("--model", type=str,
                      help="LLM model to use")
    parser.add_argument("--state-file", type=str, default="trading_system_state.json",
                      help="State file to load/save (default: trading_system_state.json)")
    parser.add_argument("--log-dir", type=str, default="logs",
                      help="Directory to store logs (default: logs)")
    parser.add_argument("--test-run", action="store_true",
                      help="Run once immediately instead of scheduling")
    
    return parser.parse_args()

def run_trading_cycle(args):
    """Run a single trading cycle."""
    logger.info("Starting trading cycle")
    
    # Create command
    cmd = ["python", "trading_cli.py", "live", "--days", "1"]
    
    # Add optional arguments
    if args.tickers:
        cmd.extend(["--tickers", args.tickers])
    
    if args.capital:
        cmd.extend(["--capital", str(args.capital)])
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"trading_{timestamp}.log"
        
        with open(log_file, "w") as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if process.returncode == 0:
            logger.info(f"Trading cycle completed successfully. Log saved to {log_file}")
        else:
            logger.error(f"Trading cycle failed with code {process.returncode}. See {log_file} for details")
    
    except Exception as e:
        logger.error(f"Error running trading cycle: {e}")

def main():
    """Main function to schedule and run trading tasks."""
    args = parse_arguments()
    
    # Parse the time
    try:
        hour, minute = map(int, args.time.split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError("Invalid time")
    except:
        logger.error(f"Invalid time format: {args.time}. Please use HH:MM in 24-hour format")
        return
    
    if args.test_run:
        logger.info("Running test cycle")
        run_trading_cycle(args)
    else:
        # Schedule daily task
        schedule.every().day.at(args.time).do(run_trading_cycle, args)
        
        logger.info(f"Trading scheduled to run daily at {args.time}")
        logger.info("Press Ctrl+C to exit")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    main()