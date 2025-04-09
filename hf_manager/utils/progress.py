"""
Progress tracking utility for agents.
"""

import threading
import time
from enum import Enum
from colorama import Fore, Style, init
from datetime import datetime

# Initialize colorama
init(autoreset=True)


class ProgressStateType(Enum):
    """Types of progress states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressTracker:
    """
    Track and display progress of agent operations.
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.agents = {}
        self.running = False
        self.thread = None
        self.display_lock = threading.Lock()
    
    def start(self):
        """Start tracking progress."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._display_progress)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop tracking progress."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def update_status(self, agent_name, ticker, status):
        """
        Update the status of an agent.
        
        Args:
            agent_name (str): Name of the agent
            ticker (str): Ticker symbol or None for general status
            status (str): Status message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with self.display_lock:
            if agent_name not in self.agents:
                self.agents[agent_name] = {
                    "state": ProgressStateType.RUNNING,
                    "tickers": {},
                    "last_update": timestamp,
                    "current_ticker": ticker,
                    "status": status
                }
            else:
                self.agents[agent_name]["state"] = ProgressStateType.RUNNING
                self.agents[agent_name]["last_update"] = timestamp
                self.agents[agent_name]["current_ticker"] = ticker
                self.agents[agent_name]["status"] = status
            
            if ticker:
                if ticker not in self.agents[agent_name]["tickers"]:
                    self.agents[agent_name]["tickers"][ticker] = {
                        "state": ProgressStateType.RUNNING,
                        "status": status,
                        "last_update": timestamp
                    }
                else:
                    self.agents[agent_name]["tickers"][ticker]["state"] = ProgressStateType.RUNNING
                    self.agents[agent_name]["tickers"][ticker]["status"] = status
                    self.agents[agent_name]["tickers"][ticker]["last_update"] = timestamp
            
            # Mark agent as completed if status is "Done"
            if status == "Done":
                if ticker:
                    self.agents[agent_name]["tickers"][ticker]["state"] = ProgressStateType.COMPLETED
                else:
                    self.agents[agent_name]["state"] = ProgressStateType.COMPLETED
    
    def _display_progress(self):
        """Display progress in the console."""
        last_display = {}
        
        while self.running:
            # Skip display if nothing changed
            with self.display_lock:
                current_state = {agent: {"status": data["status"], "ticker": data["current_ticker"]} 
                               for agent, data in self.agents.items()}
                
                if current_state != last_display:
                    # Only display if something changed
                    last_display = current_state
                    self._render_progress()
            
            time.sleep(1.0)  
    
    def _render_progress(self):
        """Render the progress display."""
        with self.display_lock:
            # Build the display string
            display = []
            
            display.append(f"{Fore.CYAN}{Style.BRIGHT}Agent Progress:{Style.RESET_ALL}")
            
            for agent_name, agent_data in self.agents.items():
                # Format agent name
                formatted_agent = agent_name.replace("_agent", "").replace("_", " ").title()
                
                # Get agent status
                status = agent_data["status"]
                ticker = agent_data["current_ticker"]
                
                # Format status line
                if agent_data["state"] == ProgressStateType.COMPLETED:
                    agent_line = f"  {Fore.GREEN}✓ {formatted_agent}: {status}{Style.RESET_ALL}"
                elif agent_data["state"] == ProgressStateType.FAILED:
                    agent_line = f"  {Fore.RED}✗ {formatted_agent}: {status}{Style.RESET_ALL}"
                else:
                    agent_line = f"  {Fore.YELLOW}⟳ {formatted_agent}: {status}{Style.RESET_ALL}"
                
                display.append(agent_line)
                
                # Show ticker-specific status if applicable
                if ticker and ticker in agent_data["tickers"]:
                    ticker_data = agent_data["tickers"][ticker]
                    ticker_status = ticker_data["status"]
                    
                    if ticker_data["state"] == ProgressStateType.COMPLETED:
                        ticker_line = f"    {Fore.GREEN}✓ {ticker}: {ticker_status}{Style.RESET_ALL}"
                    elif ticker_data["state"] == ProgressStateType.FAILED:
                        ticker_line = f"    {Fore.RED}✗ {ticker}: {ticker_status}{Style.RESET_ALL}"
                    else:
                        ticker_line = f"    {Fore.YELLOW}⟳ {ticker}: {ticker_status}{Style.RESET_ALL}"
                    
                    display.append(ticker_line)
            
            # Print the display
            print("\n".join(display))
            print("\n" + "-" * 80)


# Create a singleton progress tracker
progress = ProgressTracker()