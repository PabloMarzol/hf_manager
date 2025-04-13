# agents/risk_manager.py
from typing import Dict, List
import pandas as pd
import numpy as np
from signals.data.models import AnalystSignal

class RiskManager:
    """
    Handles risk management by setting position limits and 
    suggesting stop-loss levels based on volatility and portfolio allocation.
    """
    
    def __init__(self, max_position_pct=0.05, max_risk_per_trade=0.02):
        """
        Initialize the risk manager.
        
        Args:
            max_position_pct: Maximum position size as percentage of portfolio (default: 5%)
            max_risk_per_trade: Maximum risk per trade as percentage of portfolio (default: 2%)
        """
        self.max_position_pct = max_position_pct
        self.max_risk_per_trade = max_risk_per_trade
    
    def analyze(self, tickers, prices_dict, portfolio, end_date):
        """
        Generate risk management signals for multiple tickers.
        
        Args:
            tickers: List of tickers to analyze
            prices_dict: Dictionary of price DataFrames by ticker
            portfolio: Current portfolio state
            end_date: Analysis end date
            
        Returns:
            dict: Ticker-to-signal mapping with position limits and stop levels
        """
        signals = {}
        portfolio_value = self._calculate_portfolio_value(portfolio)
        
        for ticker in tickers:
            # Get price data for volatility calculations
            price_df = prices_dict.get(ticker, pd.DataFrame())
            if price_df.empty:
                signals[ticker] = self._create_default_signal(ticker, portfolio_value)
                continue
            
            # Calculate position size limits
            max_position_value = portfolio_value * self.max_position_pct
            max_risk_value = portfolio_value * self.max_risk_per_trade
            
            # Calculate current price and volatility
            current_price = price_df['close'].iloc[-1] if not price_df.empty else 0
            volatility = self._calculate_volatility(price_df)
            
            # Calculate stop loss based on volatility and risk tolerance
            atr = self._calculate_atr(price_df)
            stop_loss_pct = min(0.05, max(0.02, volatility * 2))  # 2-5% range based on volatility
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            # Calculate maximum shares based on position value limit
            max_shares = int(max_position_value / current_price) if current_price > 0 else 0
            
            # Calculate risk-adjusted position size
            if atr > 0:
                risk_per_share = atr * 2  # 2x ATR for stop placement
                risk_adjusted_shares = int(max_risk_value / risk_per_share) if risk_per_share > 0 else 0
                max_shares = min(max_shares, risk_adjusted_shares)
            
            # Create signal
            signals[ticker] = AnalystSignal(
                signal="neutral",  # Risk manager doesn't provide directional signals
                confidence=None,
                reasoning={
                    "current_price": current_price,
                    "volatility": volatility,
                    "atr": atr,
                    "max_position_value": max_position_value,
                    "max_position_shares": max_shares,
                    "stop_loss_level": stop_loss_price,
                    "stop_loss_pct": stop_loss_pct
                },
                max_position_size=max_shares
            )
        
        return signals
    
    def _calculate_portfolio_value(self, portfolio):
        """Calculate total portfolio value including cash and positions."""
        cash = portfolio.get("cash", 0)
        positions_value = sum(
            pos.get("quantity", 0) * pos.get("average_price", 0)
            for pos in portfolio.get("positions", {}).values()
        )
        return cash + positions_value
    
    def _calculate_volatility(self, price_df):
        """Calculate historical volatility (20-day)."""
        if len(price_df) < 20:
            return 0.02  # Default to 2% if not enough data
            
        returns = price_df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualize daily volatility
    
    def _calculate_atr(self, price_df, period=14):
        """Calculate Average True Range."""
        if len(price_df) < period:
            return price_df['close'].iloc[-1] * 0.02  # Default to 2% of price
            
        high = price_df['high']
        low = price_df['low']
        close = price_df['close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _create_default_signal(self, ticker, portfolio_value):
        """Create a default risk signal when price data is unavailable."""
        max_position_value = portfolio_value * self.max_position_pct
        
        return AnalystSignal(
            signal="neutral",
            confidence=None,
            reasoning={
                "note": "Insufficient price data for detailed risk analysis",
                "max_position_value": max_position_value,
                "max_position_shares": 0  # Can't calculate without price
            },
            max_position_size=0
        )