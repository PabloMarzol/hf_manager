import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_run():
    """Run MixGo with test parameters."""
    from main import run_mixgo
    from argparse import Namespace
    
    # Create test arguments
    args = Namespace(
        tickers="AAPL,MSFT,GOOGL",
        start_date=None,
        end_date=None,
        mock=True,
        dry_run=True,
        model="llama-3.1-70b-versatile", 
        provider="Groq"
    )
    
    # Run MixGo
    await run_mixgo(args)

if __name__ == "__main__":
    asyncio.run(test_run())