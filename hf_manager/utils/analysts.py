"""
Analyst utilities for the trading system.
"""

from typing import Dict, List, Tuple, Callable, Any

# Import agent functions
try:
    from investors.warren_buffett import warren_buffett_agent
    from investors.ben_graham import ben_graham_agent
    from investors.charlie_munger import charlie_munger_agent
    from investors.peter_lynch import peter_lynch_agent
    from investors.bill_ackman import bill_ackman_agent
    from investors.michael_burry import michael_burry_agent
    from investors.stanley_druckenmiller import stanley_druckenmiller_agent
    from investors.cathie_wood import cathie_wood_agent
    from investors.phil_fisher import phil_fisher_agent
    from investors.fundamentals import fundamentals_agent
    from investors.technicals import technical_analyst_agent
    from investors.sentiment import sentiment_agent
    from investors.valuation import valuation_agent
except ImportError:
    # Create placeholder functions for missing agents
    def placeholder_agent(state: Any) -> Any:
        return {"messages": state["messages"], "data": state["data"]}
    
    warren_buffett_agent = placeholder_agent
    ben_graham_agent = placeholder_agent
    charlie_munger_agent = placeholder_agent
    peter_lynch_agent = placeholder_agent
    bill_ackman_agent = placeholder_agent
    michael_burry_agent = placeholder_agent
    stanley_druckenmiller_agent = placeholder_agent
    cathie_wood_agent = placeholder_agent
    phil_fisher_agent = placeholder_agent
    fundamentals_agent = placeholder_agent
    technical_analyst_agent = placeholder_agent
    sentiment_agent = placeholder_agent
    valuation_agent = placeholder_agent

# Order of analysts to display in UI
ANALYST_ORDER = [
    ("Warren Buffett", "warren_buffett_agent"),
    ("Charlie Munger", "charlie_munger_agent"),
    ("Benjamin Graham", "ben_graham_agent"),
    ("Peter Lynch", "peter_lynch_agent"),
    ("Phil Fisher", "phil_fisher_agent"),
    ("Bill Ackman", "bill_ackman_agent"),
    ("Michael Burry", "michael_burry_agent"),
    ("Stanley Druckenmiller", "stanley_druckenmiller_agent"),
    ("Cathie Wood", "cathie_wood_agent"),
    ("Fundamentals Analysis", "fundamentals_agent"),
    ("Technical Analysis", "technicals_agent"),
    ("Sentiment Analysis", "sentiment_agent"),
    ("Valuation Analysis", "valuation_agent"),
]

# Map of analyst keys to their node names and functions
def get_analyst_nodes() -> Dict[str, Tuple[str, Callable]]:
    """
    Get a dictionary mapping analyst keys to tuples of (node_name, node_function).
    
    Returns:
        Dict[str, Tuple[str, Callable]]: Dictionary of analyst nodes
    """
    return {
        "warren_buffett_agent": ("warren_buffett_agent", warren_buffett_agent),
        "ben_graham_agent": ("ben_graham_agent", ben_graham_agent),
        "charlie_munger_agent": ("charlie_munger_agent", charlie_munger_agent),
        "peter_lynch_agent": ("peter_lynch_agent", peter_lynch_agent),
        "bill_ackman_agent": ("bill_ackman_agent", bill_ackman_agent),
        "michael_burry_agent": ("michael_burry_agent", michael_burry_agent),
        "stanley_druckenmiller_agent": ("stanley_druckenmiller_agent", stanley_druckenmiller_agent),
        "cathie_wood_agent": ("cathie_wood_agent", cathie_wood_agent),
        "phil_fisher_agent": ("phil_fisher_agent", phil_fisher_agent),
        "fundamentals_agent": ("fundamentals_agent", fundamentals_agent),
        "technicals_agent": ("technical_analyst_agent", technical_analyst_agent),
        "sentiment_agent": ("sentiment_agent", sentiment_agent),
        "valuation_agent": ("valuation_agent", valuation_agent),
    }