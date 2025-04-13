# mixgo/data/fetcher.py
import os
import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

from data.cache import get_cache
from data.models import Price, FinancialMetric, LineItem, InsiderTrade, CompanyNews

class DataFetcher:
    """
    Handles fetching financial and market data with caching and error handling.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the data fetcher.
        
        Args:
            use_cache: Whether to use data caching
        """
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
        self.base_url = "https://api.financialdatasets.ai"
    
    def get_prices(self, ticker: str, start_date: Optional[str], end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker and return as DataFrame.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (optional)
            end_date: End date
            
        Returns:
            DataFrame with price data (date, open, high, low, close, volume)
        """
        try:
            # Build request URL
            url = f"{self.base_url}/prices/?ticker={ticker}&interval=day&interval_multiplier=1"
            if start_date:
                url += f"&start_date={start_date}"
            url += f"&end_date={end_date}"
            
            # Make API request
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching prices for {ticker}: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
            # Parse response
            data = response.json()
            prices = [Price(**p) for p in data.get("prices", [])]
            
            if not prices:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame([p.model_dump() for p in prices])
            df["date"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"]).set_index("date").sort_index()
            
            return df
            
        except Exception as e:
            print(f"Error in get_prices for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 5) -> List[FinancialMetric]:
        """
        Fetch financial metrics for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date
            period: Period type (default: "ttm")
            limit: Max number of records to return
            
        Returns:
            List of FinancialMetric objects
        """
        try:
            url = f"{self.base_url}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching financial metrics for {ticker}: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            return [FinancialMetric(**m) for m in data.get("financial_metrics", [])]
            
        except Exception as e:
            print(f"Error in get_financial_metrics for {ticker}: {e}")
            return []
    
    def get_line_items(
        self, 
        ticker: str, 
        end_date: str, 
        line_items: Optional[List[str]] = None,
        period: str = "ttm", 
        limit: int = 5
    ) -> List[LineItem]:
        """
        Fetch specific financial line items for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date
            line_items: List of line items to fetch (default: common items)
            period: Period type (default: "ttm")
            limit: Max number of records to return
            
        Returns:
            List of LineItem objects
        """
        try:
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
                    "outstanding_shares"
                ]
            
            url = f"{self.base_url}/financials/search/line-items"
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                headers["Content-Type"] = "application/json"
                
            body = {
                "tickers": [ticker],
                "line_items": line_items,
                "end_date": end_date,
                "period": period,
                "limit": limit,
            }
            
            response = requests.post(url, headers=headers, json=body)
            if response.status_code != 200:
                print(f"Error fetching line items for {ticker}: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            return [LineItem(**item) for item in data.get("search_results", [])]
            
        except Exception as e:
            print(f"Error in get_line_items for {ticker}: {e}")
            return []
    
    def get_insider_trades(
        self, 
        ticker: str, 
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 50
    ) -> List[InsiderTrade]:
        """
        Fetch insider trades for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date
            start_date: Start date (optional)
            limit: Max number of records to return
            
        Returns:
            List of InsiderTrade objects
        """
        try:
            url = f"{self.base_url}/insider-trades/?ticker={ticker}&filing_date_lte={end_date}"
            if start_date:
                url += f"&filing_date_gte={start_date}"
            url += f"&limit={limit}"
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching insider trades for {ticker}: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            return [InsiderTrade(**trade) for trade in data.get("insider_trades", [])]
            
        except Exception as e:
            print(f"Error in get_insider_trades for {ticker}: {e}")
            return []
    
    def get_company_news(
        self, 
        ticker: str, 
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 50
    ) -> List[CompanyNews]:
        """
        Fetch company news for a ticker.
        
        Args:
            ticker: Ticker symbol
            end_date: End date
            start_date: Start date (optional)
            limit: Max number of records to return
            
        Returns:
            List of CompanyNews objects
        """
        try:
            url = f"{self.base_url}/news/?ticker={ticker}&end_date={end_date}"
            if start_date:
                url += f"&start_date={start_date}"
            url += f"&limit={limit}"
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching company news for {ticker}: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            return [CompanyNews(**news) for news in data.get("news", [])]
            
        except Exception as e:
            print(f"Error in get_company_news for {ticker}: {e}")
            return []
    
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """
        Get market cap for a ticker as of end_date.
        
        Args:
            ticker: Ticker symbol
            end_date: Date to get market cap for
            
        Returns:
            Market cap value or None if not available
        """
        try:
            # Get from financial metrics
            metrics = self.get_financial_metrics(ticker, end_date, limit=1)
            if metrics and metrics[0].market_cap is not None:
                return metrics[0].market_cap
                
            return None
            
        except Exception as e:
            print(f"Error in get_market_cap for {ticker}: {e}")
            return None