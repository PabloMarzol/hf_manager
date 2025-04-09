"""
Cache module for financial data.
Uses a simple JSON file store to cache data and reduce API calls.
"""

import json
import os
from pathlib import Path
import time
from datetime import datetime

class DataCache:
    """
    Simple file-based cache for financial data.
    Stores data in JSON files within a cache directory.
    """
    
    def __init__(self, cache_dir="data/cache"):
        """
        Initialize the cache with a directory.
        
        Args:
            cache_dir (str): Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        self.prices_dir = self.cache_dir / "prices"
        self.metrics_dir = self.cache_dir / "metrics"
        self.news_dir = self.cache_dir / "news"
        self.insider_dir = self.cache_dir / "insider_trades"
        
        self.prices_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.news_dir.mkdir(exist_ok=True)
        self.insider_dir.mkdir(exist_ok=True)
        
        # Cache stats
        self.hits = 0
        self.misses = 0
    
    def _get_cache_file(self, ticker, cache_type):
        """
        Get the path to a cache file for a ticker and type.
        
        Args:
            ticker (str): Ticker symbol
            cache_type (str): Type of data ("prices", "metrics", "news", "insider_trades")
            
        Returns:
            Path: Path to the cache file
        """
        if cache_type == "prices":
            return self.prices_dir / f"{ticker.lower()}.json"
        elif cache_type == "metrics":
            return self.metrics_dir / f"{ticker.lower()}.json"
        elif cache_type == "news":
            return self.news_dir / f"{ticker.lower()}.json"
        elif cache_type == "insider_trades":
            return self.insider_dir / f"{ticker.lower()}.json"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _load_cache(self, ticker, cache_type):
        """
        Load data from the cache for a ticker and type.
        
        Args:
            ticker (str): Ticker symbol
            cache_type (str): Type of data
            
        Returns:
            dict: Cached data or None if not found
        """
        cache_file = self._get_cache_file(ticker, cache_type)
        
        if not cache_file.exists():
            self.misses += 1
            return None
        
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            
            # Check if cache is too old (1 day for most data, 7 days for historical data)
            max_age = 86400  # 1 day in seconds
            if "timestamp" in data:
                age = time.time() - data["timestamp"]
                if age > max_age:
                    self.misses += 1
                    return None
            
            self.hits += 1
            return data.get("data")
        except:
            self.misses += 1
            return None
    
    def _save_cache(self, ticker, cache_type, data):
        """
        Save data to the cache.
        
        Args:
            ticker (str): Ticker symbol
            cache_type (str): Type of data
            data: Data to cache
        """
        cache_file = self._get_cache_file(ticker, cache_type)
        
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "data": data,
                    "timestamp": time.time(),
                    "date": datetime.now().isoformat()
                }, f)
            return True
        except:
            return False
    
    # Prices cache
    def get_prices(self, ticker):
        """Get cached prices for a ticker."""
        return self._load_cache(ticker, "prices")
    
    def set_prices(self, ticker, prices):
        """Set cached prices for a ticker."""
        return self._save_cache(ticker, "prices", prices)
    
    # Financial metrics cache
    def get_financial_metrics(self, ticker):
        """Get cached financial metrics for a ticker."""
        return self._load_cache(ticker, "metrics")
    
    def set_financial_metrics(self, ticker, metrics):
        """Set cached financial metrics for a ticker."""
        return self._save_cache(ticker, "metrics", metrics)
    
    # News cache
    def get_company_news(self, ticker):
        """Get cached news for a ticker."""
        return self._load_cache(ticker, "news")
    
    def set_company_news(self, ticker, news):
        """Set cached news for a ticker."""
        return self._save_cache(ticker, "news", news)
    
    # Insider trades cache
    def get_insider_trades(self, ticker):
        """Get cached insider trades for a ticker."""
        return self._load_cache(ticker, "insider_trades")
    
    def set_insider_trades(self, ticker, trades):
        """Set cached insider trades for a ticker."""
        return self._save_cache(ticker, "insider_trades", trades)
    
    def clear_cache(self, tickers=None):
        """
        Clear the cache for specified tickers or all tickers.
        
        Args:
            tickers (list, optional): List of tickers to clear. If None, clear all.
        """
        if tickers:
            for ticker in tickers:
                for cache_type in ["prices", "metrics", "news", "insider_trades"]:
                    cache_file = self._get_cache_file(ticker, cache_type)
                    if cache_file.exists():
                        cache_file.unlink()
        else:
            # Clear all cache files
            for cache_dir in [self.prices_dir, self.metrics_dir, self.news_dir, self.insider_dir]:
                for file in cache_dir.glob("*.json"):
                    file.unlink()
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate
        }


# Create a singleton cache instance
_cache_instance = None

def get_cache():
    """
    Get the singleton cache instance.
    
    Returns:
        DataCache: The cache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance