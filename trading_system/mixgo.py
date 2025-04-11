# agents/mega_agent.py
from typing import Dict, List
from pydantic import BaseModel, Field
from signals.data.models import AnalystSignal
from signals.llm.client import LLMClient
from signals.llm.prompts import MEGA_AGENT_PROMPT

class MixgoDecision(BaseModel):
    """Final trading decision model."""
    action: str = Field(description="Trading action: buy, sell, short, cover, or hold")
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision (0-100)")
    reasoning: str = Field(description="Explanation for the decision")

class MixGoAgent:
    """
    MixGo agent that integrates signals from multiple analysts and uses
    LLM-powered meta-reasoning to make final trading decisions.
    """
    
    def __init__(self, llm_client, agents=None):
        """
        Initialize the MixGo agent.
        
        Args:
            llm_client: LLM client for meta-reasoning
            agents: List of analyst agents to use (defaults to the three core agents)
        """
        self.llm_client = llm_client
        
        # Default to the three core agents if none provided
        if agents is None:
            from trading_system.stanley_drucken import StanleyDruckenmillerAgent
            from trading_system.charlie_munger import CharlieMungerAgent
            from trading_system.technical_analyst import TechnicalAnalystAgent
            
            self.agents = [
                StanleyDruckenmillerAgent(),
                CharlieMungerAgent(),
                TechnicalAnalystAgent()
            ]
        else:
            self.agents = agents
    
    async def analyze(self, tickers, data_fetcher, portfolio, end_date, start_date=None):
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
        
        for ticker in tickers:
            # Get current position and cash information
            position = portfolio.get_position(ticker)
            cash = portfolio.get_cash()
            
            # Prepare the context for the LLM
            context = {
                "ticker": ticker,
                "signals": ticker_signals[ticker],
                "position": position.model_dump(),
                "cash": cash,
                "portfolio_context": {
                    "total_value": portfolio.get_total_value(),
                    "exposure": portfolio.get_exposure(),
                }
            }
            
            # Call the LLM for meta-reasoning and decision
            decision = await self.llm_client.generate_decision(
                context=context,
                system_prompt=MEGA_AGENT_PROMPT,
                output_model=MixgoDecision
            )
            
            decisions[ticker] = decision
        
        return decisions