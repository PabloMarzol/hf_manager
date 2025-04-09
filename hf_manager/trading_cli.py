#!/usr/bin/env python
"""
Trading System Runner

This script provides a command-line interface to run the trading system
in different modes: backtest, live, or analyze.
"""

import argparse
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import sys

from trading_system import TradingSystem
from config import UNIVERSE, INITIAL_CAPITAL, MARGIN_REQUIREMENT, ANALYSTS, LLM_MODEL

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the LLM-powered trading system.")
    
    # Main command
    parser.add_argument("command", choices=["backtest", "live", "analyze"], 
                      help="Command to run")
    
    # Common options
    parser.add_argument("--tickers", type=str, 
                      help="Comma-separated list of ticker symbols (overrides config)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                      help=f"Initial capital (default: {INITIAL_CAPITAL})")
    parser.add_argument("--margin", type=float, default=MARGIN_REQUIREMENT,
                      help=f"Margin requirement (default: {MARGIN_REQUIREMENT})")
    parser.add_argument("--model", type=str, default=LLM_MODEL,
                      help=f"LLM model to use (default: {LLM_MODEL})")
    parser.add_argument("--analysts", type=str, 
                      help="Comma-separated list of analysts to use (overrides config)")
    
    # Backtest options
    parser.add_argument("--start-date", type=str,
                      help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                      help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--num-days", type=int, default=0,
                      help="Number of trading days to backtest (alternative to end-date)")
    
    # Live trading options
    parser.add_argument("--days", type=int, default=1,
                      help="Number of days to run live trading (default: 1)")
    
    # Analysis options
    parser.add_argument("--results-file", type=str,
                      help="Results file to analyze")
    
    return parser.parse_args()

def main():
    """Main function to run the trading system."""
    args = parse_arguments()
    
     # Check for required environment variables
    required_vars = ["FINANCIAL_DATASETS_API_KEY"]
    if args.model.startswith("gpt") or args.model.startswith("o1") or args.model.startswith("o3"):
        required_vars.append("OPENAI_API_KEY")
    elif args.model.startswith("claude"):
        required_vars.append("ANTHROPIC_API_KEY")
    elif args.model.startswith("deepseek"):
        required_vars.append("DEEPSEEK_API_KEY")
    elif args.model.startswith("gemini"):
        required_vars.append("GOOGLE_API_KEY")
    elif args.model.startswith("llama") or args.model.startswith("meta-llama"):
        required_vars.append("GROQ_API_KEY")
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: The following required environment variables are missing: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment.")
        sys.exit(1)
    
    # Override config with command line arguments
    config_override = {}
    
    if args.tickers:
        config_override["UNIVERSE"] = [ticker.strip() for ticker in args.tickers.split(",")]
    
    if args.capital != INITIAL_CAPITAL:
        config_override["INITIAL_CAPITAL"] = args.capital
    
    if args.margin != MARGIN_REQUIREMENT:
        config_override["MARGIN_REQUIREMENT"] = args.margin
    
    if args.model != LLM_MODEL:
        config_override["LLM_MODEL"] = args.model
        # Set provider based on model prefix
        if args.model.startswith("gpt") or args.model.startswith("o1") or args.model.startswith("o3"):
            config_override["LLM_PROVIDER"] = "OpenAI"
        elif args.model.startswith("claude"):
            config_override["LLM_PROVIDER"] = "Anthropic"
        elif args.model.startswith("deepseek"):
            config_override["LLM_PROVIDER"] = "DeepSeek"
        elif args.model.startswith("gemini"):
            config_override["LLM_PROVIDER"] = "Gemini"
        elif args.model.startswith("llama") or args.model.startswith("meta-llama"):
            config_override["LLM_PROVIDER"] = "Groq"
    
    if args.analysts:
        config_override["ANALYSTS"] = [analyst.strip() for analyst in args.analysts.split(",")]
    
    # Create trading system
    trading_system = TradingSystem(config_override)
    
    # Execute command
    if args.command == "backtest":
        # Run backtest
        start_date = args.start_date or (datetime.now() - relativedelta(years=1)).strftime("%Y-%m-%d")
        
        # If num_days is specified, calculate end_date from start_date + num_days
        if args.num_days > 0:
            end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=args.num_days)).strftime("%Y-%m-%d")
        else:
            end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Tickers: {trading_system.config['UNIVERSE']}")
        print(f"Initial capital: ${trading_system.config['INITIAL_CAPITAL']:,.2f}")
        print(f"Using model: {trading_system.config['LLM_MODEL']}")
        print(f"Using analysts: {trading_system.config['ANALYSTS']}")
        
        # Run backtest
        backtest_results = trading_system.backtest(start_date, end_date)
        
        # Analyze results
        if backtest_results:
            trading_system.analyze_results(backtest_results)
        
    elif args.command == "live":
        # Run live trading
        print(f"Running live trading for {args.days} days")
        print(f"Tickers: {trading_system.config['UNIVERSE']}")
        print(f"Initial capital: ${trading_system.config['INITIAL_CAPITAL']:,.2f}")
        print(f"Using model: {trading_system.config['LLM_MODEL']}")
        print(f"Using analysts: {trading_system.config['ANALYSTS']}")
        
        # Run live trading
        live_results = trading_system.run_live_trading(args.days)
        
        # Save state
        trading_system.save_state()
        
    elif args.command == "analyze":
        # Analyze results
        if args.results_file:
            print(f"Analyzing results from {args.results_file}")
            # Load state file
            if trading_system.load_state(args.results_file):
                # Analyze results
                trading_system.analyze_results()
        else:
            print("Please specify a results file to analyze with --results-file")
            sys.exit(1)

if __name__ == "__main__":
    main()