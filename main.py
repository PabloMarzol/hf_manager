# mixgo/main.py
import asyncio
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

from trading_system.bill_ackman import BillAckmanAgent
from trading_system.michael_burry import MichaelBurryAgent
from trading_system.technical_analyst import TechnicalAnalystAgent
from trading_system.mixgo import MixGoAgent
from signals.brokers.alpaca import AlpacaBroker
from signals.brokers.mock import MockBroker
from signals.data.fetcher import DataFetcher
from signals.llm.client import LLMClient
from signals.utils.progress import progress

# Load environment variables
load_dotenv()

async def run_mixgo(args):
    """Run the MixGo trading system."""
    # Initialize progress tracking
    progress.update_status("mixgo", None, "Initializing")
    
    # Parse tickers
    tickers = args.tickers.split(",") if args.tickers else ["AAPL", "MSFT", "GOOGL"]
    
    # Set dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
    
    # Initialize components
    data_fetcher = DataFetcher(use_cache=True)
    
    # Initialize broker
    if args.mock:
        broker = MockBroker()
    else:
        broker = AlpacaBroker()
    
    # Connect to broker
    if args.mock:
        await broker.connect({})
    else:
        await broker.connect({
            "api_key": os.getenv("ALPACA_API_KEY"),
            "api_secret": os.getenv("ALPACA_API_SECRET"),
            "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        })
    
    # Get account info and positions
    account_info = await broker.get_account_info()
    positions = await broker.get_positions()
    
    # Create portfolio dictionary
    portfolio = {
        "cash": float(account_info["cash"]),
        "margin_requirement": 0.5,  # 50% margin requirement
        "margin_used": 0.0,
        "positions": {}
    }
    
    # Initialize positions
    for position in positions:
        ticker = position.ticker
        direction = position.direction
        quantity = position.quantity
        average_price = position.average_price
        
        if ticker not in portfolio["positions"]:
            portfolio["positions"][ticker] = {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0
            }
        
        if direction == "long":
            portfolio["positions"][ticker]["long"] = quantity
            portfolio["positions"][ticker]["long_cost_basis"] = average_price
        else:  # short
            portfolio["positions"][ticker]["short"] = quantity
            portfolio["positions"][ticker]["short_cost_basis"] = average_price
            # Calculate margin used (50% of position value)
            margin = quantity * average_price * 0.5
            portfolio["positions"][ticker]["short_margin_used"] = margin
            portfolio["margin_used"] += margin
    
    # Create agents
    ackman_agent = BillAckmanAgent()
    burry_agent = MichaelBurryAgent()
    technical_agent = TechnicalAnalystAgent()
    
    # Create LLM client
    llm_client = LLMClient(
        model_name=args.model or "gpt-4o",
        model_provider=args.provider or "OpenAI"
    )
    
    # Create MixGo agent
    mixgo_agent = MixGoAgent(
        llm_client=llm_client,
        agents=[ackman_agent, burry_agent, technical_agent]
    )
    
    # Run analysis
    progress.update_status("mixgo", None, "Running analysis")
    decisions = await mixgo_agent.analyze(
        tickers=tickers,
        data_fetcher=data_fetcher,
        portfolio=portfolio,
        end_date=end_date,
        start_date=start_date
    )
    
    # Display results
    print("\n\n===== TRADING DECISIONS =====")
    for ticker, decision in decisions.items():
        print(f"\n{ticker}:")
        print(f"  Action: {decision.action.upper()}")
        print(f"  Quantity: {decision.quantity}")
        print(f"  Confidence: {decision.confidence:.1f}%")
        print(f"  Reasoning: {decision.reasoning[:200]}..." if len(decision.reasoning) > 200 else f"  Reasoning: {decision.reasoning}")
    
    # Execute trades if not in dry run mode
    if not args.dry_run and not args.mock:
        print("\n\n===== EXECUTING TRADES =====")
        for ticker, decision in decisions.items():
            if decision.action != "hold" and decision.quantity > 0:
                try:
                    print(f"Executing {decision.action} order for {decision.quantity} shares of {ticker}...")
                    order = await broker.place_order(
                        ticker=ticker,
                        direction=decision.action,
                        quantity=decision.quantity,
                        order_type="market"
                    )
                    print(f"Order placed: {order.order_id} - Status: {order.status}")
                except Exception as e:
                    print(f"Error executing trade: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MixGo Trading System")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to trade")
    parser.add_argument("--start-date", type=str, help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("--mock", action="store_true", help="Use mock broker for testing")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades, just analyze")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--provider", type=str, help="LLM provider (OpenAI or Anthropic)")
    
    args = parser.parse_args()
    
    # Run the application
    asyncio.run(run_mixgo(args))