# main.py
import asyncio
import argparse
import yaml
from datetime import datetime
from dotenv import load_dotenv
import os

from trading_system.mixgo import MixGoAgent
from trading_system.stanley_drucken import StanleyDruckenmillerAgent
from trading_system.charlie_munger import CharlieMungerAgent
from trading_system.technical_analyst import TechnicalAnalystAgent
from signals.data.fetcher import DataFetcher
from signals.brokers.ig_index import IGIndexBroker
from signals.brokers.mock import MockBroker
from signals.llm.client import LLMClient
from signals.utils.config import Config
from signals.utils.display import print_trading_output
from signals.utils.progress import progress

# Load environment variables
load_dotenv()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MixGo Trading System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to trade")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--provider", type=str, help="LLM provider")
    parser.add_argument("--mock", action="store_true", help="Use mock broker for testing")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command-line arguments if provided
    if args.tickers:
        config.tickers = args.tickers.split(",")
    if args.model:
        config.llm.model = args.model
    if args.provider:
        config.llm.provider = args.provider
    
    # Initialize progress tracking
    progress.start()
    
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        
        # Initialize LLM client
        llm_client = LLMClient(
            model_name=config.llm.model,
            model_provider=config.llm.provider
        )
        
        # Initialize broker
        if args.mock or args.backtest:
            broker = MockBroker()
        else:
            broker = IGIndexBroker()
            await broker.connect({
                "api_key": os.getenv("IG_API_KEY"),
                "account_id": os.getenv("IG_ACCOUNT_ID"),
                "password": os.getenv("IG_PASSWORD")
            })
        
        # Initialize agents
        druckenmiller_agent = StanleyDruckenmillerAgent()
        munger_agent = CharlieMungerAgent()
        technical_agent = TechnicalAnalystAgent()
        
        # Initialize MixGo agent
        mixgo_agent = MixGoAgent(
            llm_client=llm_client,
            agents=[druckenmiller_agent, munger_agent, technical_agent]
        )
        
        # Set dates for analysis
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - \
                     datetime.timedelta(days=config.analysis.lookback_days)).strftime("%Y-%m-%d")
        
        # Get current portfolio state
        account_info = await broker.get_account_info()
        positions = await broker.get_positions()
        
        # Create portfolio object
        portfolio = {
            "cash": account_info.get("available"),
            "positions": {
                p.ticker: {
                    "direction": p.direction,
                    "quantity": p.quantity,
                    "average_price": p.average_price
                } for p in positions
            }
        }
        
        # Run the MixGo agent
        results = await mixgo_agent.analyze(
            tickers=config.tickers,
            data_fetcher=data_fetcher,
            portfolio=portfolio,
            end_date=end_date,
            start_date=start_date
        )
        
        # Display results
        print_trading_output(results)
        
        # If not in backtest mode, execute trades
        if not args.backtest:
            for ticker, decision in results.items():
                if decision.action in ["buy", "sell", "short", "cover"] and decision.quantity > 0:
                    print(f"Executing {decision.action} order for {decision.quantity} shares of {ticker}...")
                    order_status = await broker.place_order(
                        ticker=ticker,
                        direction=decision.action,
                        quantity=decision.quantity,
                        order_type="market"
                    )
                    print(f"Order status: {order_status.model_dump()}")
    
    finally:
        # Stop progress tracking
        progress.stop()

if __name__ == "__main__":
    asyncio.run(main())