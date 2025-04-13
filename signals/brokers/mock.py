from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from brokers.base import Broker
from data.models import Position, OrderStatus

class MockBroker(Broker):
    """Mock broker implementation for testing."""
    
    def __init__(self):
        """Initialize the mock broker with an empty portfolio."""
        self.connected = False
        self.account_info = {
            "id": "mock-account",
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "buying_power": 200000.0,
            "equity": 100000.0,
            "margin_multiplier": 2.0,
            "status": "ACTIVE",
            "trading_blocked": False,
            "account_blocked": False,
            "pattern_day_trader": False,
            "daytrade_count": 0,
            "daytrading_buying_power": 200000.0,
            "last_equity": 100000.0,
            "long_market_value": 0.0,
            "short_market_value": 0.0,
        }
        self.positions = {}  # ticker -> Position
        self.orders = {}  # order_id -> OrderStatus
        self.ticker_prices = {}  # ticker -> price
    
    async def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to the mock broker (always succeeds)."""
        self.connected = True
        return True
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get mock account information."""
        if not self.connected:
            raise Exception("Not connected. Call connect() first.")
        
        # Update portfolio value based on positions
        long_value = sum(pos.quantity * pos.current_price 
                        for pos in self.positions.values() if pos.direction == "long")
        short_value = sum(pos.quantity * pos.current_price 
                        for pos in self.positions.values() if pos.direction == "short")
        
        self.account_info["long_market_value"] = long_value
        self.account_info["short_market_value"] = short_value
        self.account_info["portfolio_value"] = self.account_info["cash"] + long_value - short_value
        self.account_info["equity"] = self.account_info["portfolio_value"]
        
        return self.account_info
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self.connected:
            raise Exception("Not connected. Call connect() first.")
        
        return list(self.positions.values())
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker (mock implementation)."""
        # Return stored price or default
        return self.ticker_prices.get(ticker, 100.0)
    
    def set_ticker_price(self, ticker: str, price: float):
        """Set the current price for a ticker (for testing)."""
        self.ticker_prices[ticker] = price
        
        # Update position values if the ticker exists in positions
        if ticker in self.positions:
            pos = self.positions[ticker]
            old_price = pos.current_price
            pos.current_price = price
            
            # Update P&L
            if pos.direction == "long":
                pos.profit_loss = pos.quantity * (price - pos.average_price)
                pos.profit_loss_pct = (price / pos.average_price - 1) * 100 if pos.average_price > 0 else 0
            else:  # short
                pos.profit_loss = pos.quantity * (pos.average_price - price)
                pos.profit_loss_pct = (pos.average_price / price - 1) * 100 if price > 0 else 0
    
    async def place_order(
        self,
        ticker: str,
        direction: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> OrderStatus:
        """Place a mock order."""
        if not self.connected:
            raise Exception("Not connected. Call connect() first.")
        
        # Get current price
        current_price = self._get_current_price(ticker)
        
        # Execute the order immediately (simplified mock)
        order_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create order status
        order = OrderStatus(
            order_id=order_id,
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            price=limit_price if limit_price else current_price,
            status="filled",
            timestamp=timestamp
        )
        
        # Store the order
        self.orders[order_id] = order
        
        # Update positions
        self._update_positions(ticker, direction, quantity, current_price)
        
        return order
    
    def _update_positions(self, ticker: str, direction: str, quantity: int, price: float):
        """Update positions after an order is executed."""
        # Create position if it doesn't exist
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                direction="long",  # Default
                quantity=0,
                average_price=0.0,
                current_price=price,
                profit_loss=0.0,
                profit_loss_pct=0.0
            )
        
        position = self.positions[ticker]
        
        # Update position based on order direction
        if direction == "buy":
            # If short position exists, reduce it first (cover)
            if position.direction == "short" and position.quantity > 0:
                if quantity <= position.quantity:
                    # Partial cover
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                    # Update cash: release margin and pay for shares
                    margin_released = price * quantity * 0.5  # Assuming 50% margin
                    self.account_info["cash"] += margin_released - (price * quantity)
                else:
                    # Cover the entire short position and go long with remainder
                    remaining = quantity - position.quantity
                    # Release all margin
                    margin_released = price * position.quantity * 0.5  # Assuming 50% margin
                    self.account_info["cash"] += margin_released - (price * position.quantity)
                    
                    # Now go long with the remainder
                    position.direction = "long"
                    position.quantity = remaining
                    position.average_price = price
                    self.account_info["cash"] -= price * remaining
            else:
                # Add to long position
                if position.quantity > 0:
                    # Update average price
                    total_cost = position.average_price * position.quantity
                    new_cost = price * quantity
                    position.quantity += quantity
                    position.average_price = (total_cost + new_cost) / position.quantity
                else:
                    # New long position
                    position.direction = "long"
                    position.quantity = quantity
                    position.average_price = price
                
                # Update cash
                self.account_info["cash"] -= price * quantity
                
        elif direction == "sell":
            # Can only sell if long position exists
            if position.direction == "long" and position.quantity > 0:
                if quantity <= position.quantity:
                    # Partial sell
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                    # Update cash
                    self.account_info["cash"] += price * quantity
                else:
                    # This would be an error in a real system, but for mock:
                    # Sell all and go short with remainder
                    remaining = quantity - position.quantity
                    # First sell all long
                    self.account_info["cash"] += price * position.quantity
                    
                    # Then go short with remainder
                    position.direction = "short"
                    position.quantity = remaining
                    position.average_price = price
                    # Add proceeds but reserve margin
                    self.account_info["cash"] += price * remaining - (price * remaining * 0.5)  # 50% margin
            else:
                # Error - can't sell what you don't have
                print(f"Error: Cannot sell {quantity} shares of {ticker}, no long position")
                
        elif direction == "short":
            # If long position exists, reduce it first (sell)
            if position.direction == "long" and position.quantity > 0:
                if quantity <= position.quantity:
                    # Partial sell
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                    # Update cash
                    self.account_info["cash"] += price * quantity
                else:
                    # Sell all and go short with remainder
                    remaining = quantity - position.quantity
                    # First sell all long
                    self.account_info["cash"] += price * position.quantity
                    
                    # Then go short with remainder
                    position.direction = "short"
                    position.quantity = remaining
                    position.average_price = price
                    # Add proceeds but reserve margin
                    self.account_info["cash"] += price * remaining - (price * remaining * 0.5)  # 50% margin
            else:
                # New short position
                position.direction = "short"
                position.quantity += quantity
                # Update average price if adding to existing short
                if position.quantity > quantity:
                    total_cost = position.average_price * (position.quantity - quantity)
                    new_cost = price * quantity
                    position.average_price = (total_cost + new_cost) / position.quantity
                else:
                    position.average_price = price
                
                # Update cash: add proceeds but reserve margin
                self.account_info["cash"] += price * quantity - (price * quantity * 0.5)  # 50% margin
                
        elif direction == "cover":
            # Can only cover if short position exists
            if position.direction == "short" and position.quantity > 0:
                if quantity <= position.quantity:
                    # Partial cover
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                    # Update cash: release margin and pay for shares
                    margin_released = price * quantity * 0.5  # Assuming 50% margin
                    self.account_info["cash"] += margin_released - (price * quantity)
                else:
                    # This would be an error in a real system, but for mock:
                    # Cover all and go long with remainder
                    remaining = quantity - position.quantity
                    # First cover all short
                    margin_released = price * position.quantity * 0.5  # Assuming 50% margin
                    self.account_info["cash"] += margin_released - (price * position.quantity)
                    
                    # Then go long with remainder
                    position.direction = "long"
                    position.quantity = remaining
                    position.average_price = price
                    self.account_info["cash"] -= price * remaining
            else:
                # Error - can't cover what you don't have
                print(f"Error: Cannot cover {quantity} shares of {ticker}, no short position")
        
        # Clean up if position quantity is 0
        if position.quantity == 0:
            self.positions.pop(ticker, None)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order (mock implementation)."""
        if not self.connected:
            raise Exception("Not connected. Call connect() first.")
        
        # In our mock, orders are filled immediately, so cancellation is not possible
        # But we'll return success if the order exists
        return order_id in self.orders
    
    async def close_position(self, ticker: str, direction: str) -> OrderStatus:
        """Close an existing position."""
        if not self.connected:
            raise Exception("Not connected. Call connect() first.")
        
        # Check if position exists
        if ticker not in self.positions:
            raise Exception(f"No position for {ticker}")
        
        position = self.positions[ticker]
        if position.direction != direction:
            raise Exception(f"No {direction} position for {ticker}")
        
        # Determine close action
        close_action = "sell" if direction == "long" else "cover"
        
        # Place an order to close the position
        return await self.place_order(
            ticker=ticker,
            direction=close_action,
            quantity=position.quantity,
            order_type="market"
        )