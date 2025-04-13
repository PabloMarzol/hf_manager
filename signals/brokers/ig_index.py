# brokers/ig_index.py
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from brokers.base import Broker, OrderStatus, Position

class IGIndexBroker(Broker):
    """IG Index broker implementation."""
    
    def __init__(self):
        """Initialize the IG Index broker."""
        self.demo_url = "https://demo-api.ig.com/gateway/deal"
        self.base_url = "https://api.ig.com/gateway/deal"
        self.session = None
        self.cst = None  # Client Secure Token
        self.x_security_token = None
        self.account_id = None
        
    async def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to IG Index API."""
        self.api_key = credentials.get("api_key")
        self.account_id = credentials.get("account_id")
        password = credentials.get("password")
        
        if not all([self.api_key, self.account_id, password]):
            raise ValueError("Missing required credentials")
        
        self.session = aiohttp.ClientSession()
        
        # Authenticate with IG Index
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "X-IG-API-KEY": self.api_key,
            "Version": "2"
        }
        
        payload = {
            "identifier": self.account_id,
            "password": password
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/session", 
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    self.cst = response.headers.get("CST")
                    self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                    return True
                else:
                    error_data = await response.json()
                    raise Exception(f"Authentication failed: {error_data}")
        except Exception as e:
            print(f"Error connecting to IG Index: {e}")
            return False
    
    async def _make_request(self, method, endpoint, data=None, version="2"):
        """Make an authenticated request to IG Index API."""
        if not self.session or not self.cst or not self.x_security_token:
            raise Exception("Not authenticated. Call connect() first.")
        
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "X-IG-API-KEY": self.api_key,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Version": version
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with getattr(self.session, method.lower())(
                url, headers=headers, json=data
            ) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise Exception(f"API request failed: {error_data}")
        except Exception as e:
            print(f"Error making request to IG Index: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        response = await self._make_request("GET", f"accounts/{self.account_id}")
        return response
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        response = await self._make_request("GET", "positions")
        
        positions = []
        for position in response.get("positions", []):
            position_data = position.get("position", {})
            market_data = position.get("market", {})
            
            direction = "long" if position_data.get("direction") == "BUY" else "short"
            quantity = position_data.get("size", 0)
            average_price = position_data.get("level", 0)
            current_price = market_data.get("bid", 0)  # Use bid for simplicity
            
            profit_loss = position_data.get("profit", {}).get("value", 0)
            profit_loss_pct = (
                (current_price - average_price) / average_price * 100
                if direction == "long" else
                (average_price - current_price) / average_price * 100
            )
            
            positions.append(Position(
                ticker=market_data.get("epic", ""),
                direction=direction,
                quantity=quantity,
                average_price=average_price,
                current_price=current_price,
                profit_loss=profit_loss,
                profit_loss_pct=profit_loss_pct
            ))
        
        return positions
    
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
        # Convert direction to IG Index format
        ig_direction = "BUY"
        if direction in ["sell", "short"]:
            ig_direction = "SELL"
        
        # Create order payload
        payload = {
            "epic": ticker,
            "direction": ig_direction,
            "size": quantity,
            "orderType": order_type.upper(),
            "timeInForce": "FILL_OR_KILL",
            "guaranteedStop": False,
            "forceOpen": direction in ["buy", "short"]  # Force open for new positions
        }
        
        if limit_price and order_type.lower() == "limit":
            payload["level"] = limit_price
        
        if stop_price:
            payload["stopLevel"] = stop_price
        
        # Place the order
        response = await self._make_request(
            "POST", "positions/otc", data=payload, version="2"
        )
        
        # Create order status
        deal_reference = response.get("dealReference", "")
        
        # Get deal confirmation
        confirmation = await self._make_request(
            "GET", f"confirms/{deal_reference}", version="1"
        )
        
        return OrderStatus(
            order_id=confirmation.get("dealId", ""),
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            price=confirmation.get("level", 0),
            status=confirmation.get("status", ""),
            timestamp=confirmation.get("date", "")
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self._make_request("DELETE", f"orders/{order_id}")
            return True
        except Exception:
            return False
    
    async def close_position(self, ticker: str, direction: str) -> OrderStatus:
        """Close an existing position."""
        # Get current position to determine quantity
        positions = await self.get_positions()
        position = next((p for p in positions if p.ticker == ticker and p.direction == direction), None)
        
        if not position:
            raise Exception(f"No open {direction} position found for {ticker}")
        
        # Direction for closing is opposite of position
        close_direction = "sell" if direction == "long" else "buy"
        
        return await self.place_order(
            ticker=ticker,
            direction=close_direction,
            quantity=position.quantity,
            order_type="market"
        )