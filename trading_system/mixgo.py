from typing import Dict, List, Any, TypedDict
from pydantic import BaseModel, Field
from signals.data.models import AnalystSignal
from signals.llm.client import LLMClient
from signals.llm.prompts import MEGA_AGENT_PROMPT
from signals.utils.progress import progress

class MegaAgentDecision(BaseModel):
    """Final trading decision model."""
    action: str = Field(description="Trading action: buy, sell, short, cover, or hold")
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision (0-100)")
    reasoning: str = Field(description="Explanation for the decision")

class TickerContext(TypedDict):
    """Context information for a ticker's decision."""
    ticker: str
    signals: Dict[str, Any]
    position: Dict[str, Any]
    price: float
    cash: float
    portfolio_context: Dict[str, Any]

class MixGoAgent:
    """
    MixGo agent that integrates signals from multiple analysts and uses
    LLM-powered meta-reasoning to make final trading decisions.
    """
    
    def __init__(self, llm_client: LLMClient, agents=None):
        """
        Initialize the MixGo agent.
        
        Args:
            llm_client: LLM client for meta-reasoning
            agents: List of analyst agents to use (defaults to the three core agents)
        """
        self.llm_client = llm_client
        
        # Default to the three core agents if none provided
        if agents is None:
            from trading_system.bill_ackman import BillAckmanAgent
            from trading_system.michael_burry import MichaelBurryAgent
            from trading_system.technical_analyst import TechnicalAnalystAgent
            
            self.agents = [
                BillAckmanAgent(),
                MichaelBurryAgent(),
                TechnicalAnalystAgent()
            ]
        else:
            self.agents = agents
    
    async def analyze(self, tickers: List[str], data_fetcher, portfolio, end_date, start_date=None) -> Dict[str, MegaAgentDecision]:
        """
        Generate trading decisions by combining signals from all agents
        and applying LLM meta-reasoning.
        
        Args:
            tickers: List of tickers to analyze
            data_fetcher: Data fetching service
            portfolio: Current portfolio state
            end_date: Analysis end date
            start_date: Analysis start date
            
        Returns:
            dict: Ticker-to-decision mapping with quantities and reasoning
        """
        # Step 1: Collect signals from all agents
        progress.update_status("mixgo_agent", None, "Collecting signals from all agents")
        all_signals = {}
        
        for agent in self.agents:
            agent_signals = agent.analyze(tickers, data_fetcher, end_date, start_date)
            all_signals[agent.name] = agent_signals
        
        # Step 2: Organize signals by ticker
        ticker_signals = {ticker: {} for ticker in tickers}
        for agent_name, agent_signals in all_signals.items():
            for ticker, signal in agent_signals.items():
                ticker_signals[ticker][agent_name] = signal.model_dump()
        
        # Step 3: Generate decisions with LLM meta-reasoning
        decisions = {}
        progress.update_status("mixgo_agent", None, "Generating trading decisions")
        
        for ticker in tickers:
            progress.update_status("mixgo_agent", ticker, "Applying LLM meta-reasoning")
            
            # Get current position and cash information
            position = self._get_position_info(portfolio, ticker)
            cash = portfolio.get("cash", 0)
            current_price = self._get_current_price(data_fetcher, ticker, end_date)
            
            # Prepare the context for the LLM
            context = TickerContext(
                ticker=ticker,
                signals=ticker_signals[ticker],
                position=position,
                price=current_price,
                cash=cash,
                portfolio_context={
                    "total_value": self._calculate_portfolio_value(portfolio),
                    "exposure": self._calculate_portfolio_exposure(portfolio),
                    "margin_used": portfolio.get("margin_used", 0),
                    "margin_requirement": portfolio.get("margin_requirement", 0)
                }
            )
            
            # Call the LLM for meta-reasoning and decision
            decision = await self.llm_client.generate_decision(
                context=context,
                system_prompt=MEGA_AGENT_PROMPT,
                output_model=MegaAgentDecision
            )
            
            decisions[ticker] = decision
            progress.update_status("mixgo_agent", ticker, "Decision generated")
        
        progress.update_status("mixgo_agent", None, "All decisions generated")
        return decisions
    
    def _get_position_info(self, portfolio, ticker):
        """Extract position information for a specific ticker."""
        positions = portfolio.get("positions", {})
        ticker_position = positions.get(ticker, {})
        
        return {
            "long": ticker_position.get("long", 0),
            "short": ticker_position.get("short", 0),
            "long_cost_basis": ticker_position.get("long_cost_basis", 0),
            "short_cost_basis": ticker_position.get("short_cost_basis", 0),
            "short_margin_used": ticker_position.get("short_margin_used", 0)
        }
    
    def _get_current_price(self, data_fetcher, ticker, end_date):
        """Get the current price for a ticker."""
        try:
            # Try to get the most recent price
            prices_df = data_fetcher.get_prices(ticker, None, end_date)
            if not prices_df.empty:
                return float(prices_df["close"].iloc[-1])
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
        
        # Default fallback value if price can't be fetched
        return 0.0
    
    def _calculate_portfolio_value(self, portfolio):
        """Calculate total portfolio value including positions and cash."""
        cash = portfolio.get("cash", 0)
        positions = portfolio.get("positions", {})
        
        position_value = 0
        for ticker_pos in positions.values():
            long_value = ticker_pos.get("long", 0) * ticker_pos.get("long_cost_basis", 0)
            # Short positions represent a liability, so we subtract them
            short_value = ticker_pos.get("short", 0) * ticker_pos.get("short_cost_basis", 0)
            position_value += long_value - short_value
        
        return cash + position_value
    
    def _calculate_portfolio_exposure(self, portfolio):
        """Calculate portfolio exposure metrics."""
        positions = portfolio.get("positions", {})
        
        long_exposure = 0
        short_exposure = 0
        for ticker_pos in positions.values():
            long_value = ticker_pos.get("long", 0) * ticker_pos.get("long_cost_basis", 0)
            short_value = ticker_pos.get("short", 0) * ticker_pos.get("short_cost_basis", 0)
            long_exposure += long_value
            short_exposure += short_value
        
        return {
            "long": long_exposure,
            "short": short_exposure,
            "gross": long_exposure + short_exposure,
            "net": long_exposure - short_exposure
        }