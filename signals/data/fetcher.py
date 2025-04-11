# data/fetcher.py
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from data.cache import get_cache
from signals.api import (
    get_prices, 
    get_financial_metrics, 
    search_line_items, 
    get_insider_trades,
    get_company_news, 
    get_market_cap
)

class DataFetcher:
    """
    Handles data fetching with caching, normalization,
    and unified error handling.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the data fetcher.
        
        Args:
            use_cache: Whether to use the cache (default: True)
        """
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        try:
            prices = get_prices(ticker, start_date, end_date)
            # Convert to DataFrame using polars
            df = pd.DataFrame([p.model_dump() for p in prices])
            if df.empty:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
            
            # Process DataFrame
            df["date"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])
            df = df.set_index("date").sort_index()
            return df
            
        except Exception as e:
            print(f"Error fetching prices for {ticker}: {e}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get financial metrics for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date (YYYY-MM-DD)
            period: Period type (default: "ttm")
            limit: Maximum number of results to return
            
        Returns:
            List of financial metric dictionaries
        """
        try:
            metrics = get_financial_metrics(ticker, end_date, period, limit)
            return [m.model_dump() for m in metrics]
        except Exception as e:
            print(f"Error fetching financial metrics for {ticker}: {e}")
            return []
    
    def get_line_items(
        self, 
        ticker: str, 
        end_date: str, 
        line_items: Optional[List[str]] = None,
        period: str = "ttm", 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get specific financial line items for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date (YYYY-MM-DD)
            line_items: List of line items to fetch (default: common items)
            period: Period type (default: "ttm")
            limit: Maximum number of results to return
            
        Returns:
            List of line item dictionaries
        """
        # Default to common line items if none provided
        if line_items is None:
            line_items = [
                "revenue",
                "net_income",
                "earnings_per_share",
                "free_cash_flow",
                "operating_margin",
                "gross_margin",
                "debt_to_equity",
                "return_on_equity",
                "cash_and_equivalents",
                "total_debt",
                "total_assets",
                "total_liabilities",
                "research_and_development"
            ]
        
        try:
            items = search_line_items(ticker, line_items, end_date, period, limit)
            return [item.model_dump() for item in items]
        except Exception as e:
            print(f"Error fetching line items for {ticker}: {e}")
            return []
    
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """
        Get market capitalization for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Market cap value or None if not available
        """
        try:
            return get_market_cap(ticker, end_date)
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            return None