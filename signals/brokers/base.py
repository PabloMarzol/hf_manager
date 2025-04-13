from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from signals.data.models import Position, OrderStatus

class Broker(ABC):
    """Abstract broker interface defining common operations."""
    
    @abstractmethod
    async def connect(self, credentials: Dict[str, str]) -> bool:
        """
        Establish connection to the broker.
        
        Args:
            credentials: Dictionary of credentials needed to connect
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information (balance, margin available, etc.).
        
        Returns:
            dict: Account information
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            list: List of Position objects
        """
        pass
    
    @abstractmethod
    async def place_order(
        self,
        ticker: str,
        direction: str,  # "buy", "sell", "short", "cover"
        quantity: int,
        order_type: str,  # "market", "limit", "stop", etc.
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> OrderStatus:
        """
        Place a new order.
        
        Args:
            ticker: Ticker symbol
            direction: Order direction (buy, sell, short, cover)
            quantity: Number of shares/contracts
            order_type: Type of order (market, limit, stop, etc.)
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            OrderStatus: Status of the placed order
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def close_position(self, ticker: str, direction: str) -> OrderStatus:
        """
        Close an existing position.
        
        Args:
            ticker: Ticker symbol
            direction: Direction of position to close ("long" or "short")
            
        Returns:
            OrderStatus: Status of the closing order
        """
        pass