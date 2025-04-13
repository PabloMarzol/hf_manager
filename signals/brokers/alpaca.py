# brokers/alpaca.py
import alpaca_trade_api as tradeapi
from datetime import datetime
from typing import Dict, List, Any, Optional

from signals.brokers.base import Broker
from signals.data.models import Position, OrderStatus

class AlpacaBroker(Broker):
    """Alpaca broker implementation."""
    
    def __init__(self):
        """Initialize the Alpaca broker."""
        self.api = None
        self.account = None
        
    async def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to Alpaca API."""
        api_key = credentials.get("api_key")
        api_secret = credentials.get("api_secret")
        base_url = credentials.get("base_url", "https://paper-api.alpaca.markets")  # Default to paper trading
        
        if not all([api_key, api_secret]):
            raise ValueError("Missing required Alpaca credentials")
        
        try:
            self.api = tradeapi.REST(api_key, api_secret, base_url)
            self.account = self.api.get_account()
            return True
        except Exception as e:
            print(f"Error connecting to Alpaca: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.api or not self.account:
            raise Exception("Not authenticated. Call connect() first.")
        
        try:
            self.account = self.api.get_account()
            return {
                "id": self.account.id,
                "cash": float(self.account.cash),
                "portfolio_value": float(self.account.portfolio_value),
                "buying_power": float(self.account.buying_power),
                "equity": float(self.account.equity),
                "margin_multiplier": float(self.account.multiplier),
                "status": self.account.status,
                "trading_blocked": self.account.trading_blocked,
                "account_blocked": self.account.account_blocked,
                "pattern_day_trader": self.account.pattern_day_trader,
                "daytrade_count": self.account.daytrade_count,
                "daytrading_buying_power": float(self.account.daytrading_buying_power),
                "last_equity": float(self.account.last_equity),
                "long_market_value": float(self.account.long_market_value),
                "short_market_value": float(self.account.short_market_value),
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self.api:
            raise Exception("Not authenticated. Call connect() first.")
        
        try:
            alpaca_positions = self.api.list_positions()
            positions = []
            
            for pos in alpaca_positions:
                # Determine direction based on quantity
                qty = int(pos.qty)
                direction = "long" if qty > 0 else "short"
                abs_qty = abs(qty)
                
                # Get current price
                current_price = float(pos.current_price)
                avg_entry = float(pos.avg_entry_price)
                
                # Calculate P&L
                market_value = float(pos.market_value)
                cost_basis = float(pos.cost_basis)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)
                
                positions.append(Position(
                    ticker=pos.symbol,
                    direction=direction,
                    quantity=abs_qty,
                    average_price=avg_entry,
                    current_price=current_price,
                    profit_loss=unrealized_pl,
                    profit_loss_pct=unrealized_plpc * 100  # Convert to percentage
                ))
            
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    async def place_order(
        self,
        ticker: str,
        direction: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> OrderStatus:
        """Place a new order."""
        if not self.api:
            raise Exception("Not authenticated. Call connect() first.")
        
        # Map direction to Alpaca side and quantity
        side = "buy"
        if direction == "sell":
            side = "sell"
        elif direction == "short":
            side = "sell"
        elif direction == "cover":
            side = "buy"
            
        # Map order_type to Alpaca order type
        alpaca_order_type = "market"
        if order_type.lower() == "limit":
            alpaca_order_type = "limit"
        elif order_type.lower() == "stop":
            alpaca_order_type = "stop"
        elif order_type.lower() == "stop_limit":
            alpaca_order_type = "stop_limit"
        
        try:
            # Place order based on type
            if alpaca_order_type == "market":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side=side,
                    type=alpaca_order_type,
                    time_in_force="day"
                )
            elif alpaca_order_type == "limit":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side=side,
                    type=alpaca_order_type,
                    time_in_force="day",
                    limit_price=limit_price
                )
            elif alpaca_order_type == "stop":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side=side,
                    type=alpaca_order_type,
                    time_in_force="day",
                    stop_price=stop_price
                )
            elif alpaca_order_type == "stop_limit":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side=side,
                    type=alpaca_order_type,
                    time_in_force="day",
                    limit_price=limit_price,
                    stop_price=stop_price
                )
            
            # Create OrderStatus from response
            return OrderStatus(
                order_id=order.id,
                ticker=ticker,
                direction=direction,
                quantity=quantity,
                price=float(order.limit_price) if order.limit_price else None,
                status=order.status,
                timestamp=order.submitted_at.isoformat() if order.submitted_at else datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if not self.api:
            raise Exception("Not authenticated. Call connect() first.")
        
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    async def close_position(self, ticker: str, direction: str) -> OrderStatus:
        """Close an existing position."""
        if not self.api:
            raise Exception("Not authenticated. Call connect() first.")
        
        try:
            # Get current position
            position = self.api.get_position(ticker)
            quantity = abs(int(position.qty))
            
            # Map direction to correct close action
            close_action = "sell" if direction == "long" else "buy"
            
            # Close the position
            order = self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=close_action,
                type="market",
                time_in_force="day"
            )
            
            return OrderStatus(
                order_id=order.id,
                ticker=ticker,
                direction=close_action,
                quantity=quantity,
                price=None,  # Market order has no preset price
                status=order.status,
                timestamp=order.submitted_at.isoformat() if order.submitted_at else datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error closing position: {e}")
            raise