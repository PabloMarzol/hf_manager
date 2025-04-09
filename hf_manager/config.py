import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# API Keys (should be in .env file)
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Portfolio Configuration
INITIAL_CAPITAL = 100_000.0  # Starting capital
MARGIN_REQUIREMENT = 0.5    # Margin requirement for short positions (e.g., 0.5 = 50%)
MAX_POSITION_SIZE = 0.2     # Maximum position size as a fraction of portfolio (20%)

# Strategy Configuration
REBALANCE_FREQUENCY = "DAILY"  # Options: DAILY, WEEKLY, MONTHLY
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "NVDA", "TSLA", "JPM", "V", "WMT"
]  # Universe of stocks to consider

# Analysis Configuration
ANALYSTS = [
    "michael_burry_agent",
    "technicals_agent"
]
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Model to use
LLM_PROVIDER = "Groq"  # Provider to use

# Backtesting Configuration
BACKTEST_START_DATE = (datetime.now() - relativedelta(years=1)).strftime("%Y-%m-%d")
BACKTEST_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
CONSOLE_OUTPUT = True  # Whether to output to console
FILE_OUTPUT = True  # Whether to output to file
LOG_FILE = "trading_system.log"  # Log file name