import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Import from the project

from utils.progress import progress

# Initialize colorama
init(autoreset=True)

class TradingSystem:
    """
    Automated trading system that leverages the LLM-powered hedge fund system
    for stock analysis and portfolio management.
    """
    
    def __init__(self, config_override=None):
        """
        Initialize the trading system.
        
        Args:
            config_override (dict, optional): Override configuration parameters.
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self._load_config(config_override)
        
        # Initialize portfolio
        self._initialize_portfolio()
        
        # Don't initialize workflow here to avoid circular imports
        # We'll create it when needed
        self.workflow = None
        self.app = None
        
        # Initialize data storage
        self.trades_history = []
        self.portfolio_history = []
        
        self.logger.info(f"Trading system initialized with {len(self.config['UNIVERSE'])} tickers")
        self.logger.info(f"Using analysts: {', '.join(self.config['ANALYSTS'])}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler("trading_system.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def _load_config(self, config_override):
        """Load configuration from config.py and apply overrides."""
        # Default configuration from config.py module
        from config import (
                INITIAL_CAPITAL,
                MARGIN_REQUIREMENT,
                MAX_POSITION_SIZE,
                REBALANCE_FREQUENCY,
                UNIVERSE,
                ANALYSTS,
                LLM_MODEL,
                LLM_PROVIDER,
                BACKTEST_START_DATE,
                BACKTEST_END_DATE,
                LOG_LEVEL,
            )
        
        self.config = {
            "INITIAL_CAPITAL": INITIAL_CAPITAL,
            "MARGIN_REQUIREMENT": MARGIN_REQUIREMENT,
            "MAX_POSITION_SIZE": MAX_POSITION_SIZE,
            "REBALANCE_FREQUENCY": REBALANCE_FREQUENCY,
            "UNIVERSE": UNIVERSE,
            "ANALYSTS": ANALYSTS,
            "LLM_MODEL": LLM_MODEL,
            "LLM_PROVIDER": LLM_PROVIDER,
            "BACKTEST_START_DATE": BACKTEST_START_DATE,
            "BACKTEST_END_DATE": BACKTEST_END_DATE,
            "LOG_LEVEL": LOG_LEVEL,
        }
        
        # Apply overrides if any
        if config_override:
            self.config.update(config_override)
    
    def _initialize_portfolio(self):
        """Initialize the portfolio structure."""
        self.portfolio = {
            "cash": self.config["INITIAL_CAPITAL"],
            "margin_requirement": self.config["MARGIN_REQUIREMENT"],
            "margin_used": 0.0,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                } for ticker in self.config["UNIVERSE"]
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                } for ticker in self.config["UNIVERSE"]
            }
        }
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate the current portfolio value.
        
        Args:
            current_prices (dict): Current prices for all tickers.
            
        Returns:
            float: Total portfolio value.
        """
        total_value = self.portfolio["cash"]
        
        for ticker in self.config["UNIVERSE"]:
            position = self.portfolio["positions"][ticker]
            price = current_prices.get(ticker, 0)
            
            # Long position value
            long_value = position["long"] * price
            total_value += long_value
            
            # Short position unrealized PnL
            if position["short"] > 0:
                total_value += position["short"] * (position["short_cost_basis"] - price)
        
        return total_value
    
    def execute_trade(self, ticker, action, quantity, current_price):
        """
        Execute a trade.
        
        Args:
            ticker (str): The ticker symbol.
            action (str): The action to take (buy, sell, short, cover).
            quantity (int): The quantity to trade.
            current_price (float): The current price.
            
        Returns:
            int: The actual quantity traded.
        """
        if quantity <= 0:
            return 0
        
        quantity = int(quantity)  # Force integer shares
        position = self.portfolio["positions"][ticker]
        
        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares
                
                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
                
                position["long"] += quantity
                self.portfolio["cash"] -= cost
                
                self.logger.info(f"BUY {ticker}: {quantity} shares @ ${current_price:.2f} = ${cost:.2f}")
                self.trades_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "buy",
                    "quantity": quantity,
                    "price": current_price,
                    "cost": cost
                })
                
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] / current_price)
                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity
                    
                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
                    
                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    
                    self.logger.info(f"BUY {ticker} (partial): {max_quantity} shares @ ${current_price:.2f} = ${cost:.2f}")
                    self.trades_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker,
                        "action": "buy",
                        "quantity": max_quantity,
                        "price": current_price,
                        "cost": cost
                    })
                    
                    return max_quantity
                return 0
        
        elif action == "sell":
            # You can only sell as many as you own
            quantity = min(quantity, position["long"])
            if quantity > 0:
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                proceeds = quantity * current_price
                
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain
                position["long"] -= quantity
                self.portfolio["cash"] += proceeds
                
                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0
                
                self.logger.info(f"SELL {ticker}: {quantity} shares @ ${current_price:.2f} = ${proceeds:.2f} (P&L: ${realized_gain:.2f})")
                self.trades_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "sell",
                    "quantity": quantity,
                    "price": current_price,
                    "proceeds": proceeds,
                    "realized_gain": realized_gain
                })
                
                return quantity
        
        elif action == "short":
            proceeds = current_price * quantity
            margin_required = proceeds * self.portfolio["margin_requirement"]
            if margin_required <= self.portfolio["cash"]:
                # Weighted average short cost basis
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares
                
                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
                
                position["short"] += quantity
                
                # Update margin usage
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required
                
                # Increase cash by proceeds, then subtract the required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                
                self.logger.info(f"SHORT {ticker}: {quantity} shares @ ${current_price:.2f} (Margin used: ${margin_required:.2f})")
                self.trades_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "short",
                    "quantity": quantity,
                    "price": current_price,
                    "proceeds": proceeds,
                    "margin_required": margin_required
                })
                
                return quantity
            else:
                # Calculate maximum shortable quantity
                margin_ratio = self.portfolio["margin_requirement"]
                if margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (current_price * margin_ratio))
                else:
                    max_quantity = 0
                
                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * margin_ratio
                    
                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity
                    
                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
                    
                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required
                    
                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    
                    self.logger.info(f"SHORT {ticker} (partial): {max_quantity} shares @ ${current_price:.2f} (Margin used: ${margin_required:.2f})")
                    self.trades_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker,
                        "action": "short",
                        "quantity": max_quantity,
                        "price": current_price,
                        "proceeds": proceeds,
                        "margin_required": margin_required
                    })
                    
                    return max_quantity
                return 0
        
        elif action == "cover":
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - current_price) * quantity
                
                if position["short"] > 0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0
                
                margin_to_release = portion * position["short_margin_used"]
                
                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release
                
                # Pay the cost to cover, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost
                
                self.portfolio["realized_gains"][ticker]["short"] += realized_gain
                
                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0
                
                self.logger.info(f"COVER {ticker}: {quantity} shares @ ${current_price:.2f} (P&L: ${realized_gain:.2f}, Margin released: ${margin_to_release:.2f})")
                self.trades_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "cover",
                    "quantity": quantity,
                    "price": current_price,
                    "cost": cover_cost,
                    "realized_gain": realized_gain,
                    "margin_released": margin_to_release
                })
                
                return quantity
        
        return 0
    
    def run_trading_cycle(self, date=None):
        """
        Run a single trading cycle, generating analysis and executing trades.
        
        Args:
            date (str, optional): The date to run the cycle for. Defaults to today.
            
        Returns:
            dict: Results of the trading cycle.
        """
        # Use today's date if not specified
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate start date (30 days before)
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        
        self.logger.info(f"Running trading cycle for {date}")
        
        # Get current prices
        try:
            from api.api import get_price_data
            current_prices = {}
            for ticker in self.config["UNIVERSE"]:
                price_data = get_price_data(ticker, date, date)
                if not price_data.empty:
                    current_prices[ticker] = price_data.iloc[-1]["close"]
                else:
                    # Use previous day if no data for the current day
                    prev_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                    price_data = get_price_data(ticker, prev_date, prev_date)
                    if not price_data.empty:
                        current_prices[ticker] = price_data.iloc[-1]["close"]
                    else:
                        self.logger.warning(f"No price data for {ticker} on {date} or previous day")
                        continue
        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}")
            return None
        
        # Run the investment analysis
        try:
            # Start progress tracking
            progress.start()
            
            # Instead of directly calling run_hedge_fund, we'll import it here to avoid circular imports
            from main import run_hedge_fund
            
            result = run_hedge_fund(
                tickers=self.config["UNIVERSE"],
                start_date=start_date,
                end_date=date,
                portfolio=self.portfolio,
                show_reasoning=False,
                selected_analysts=self.config["ANALYSTS"],
                model_name=self.config["LLM_MODEL"],
                model_provider=self.config["LLM_PROVIDER"],
            )
            
            # Stop progress tracking
            progress.stop()
            
            decisions = result["decisions"]
            analyst_signals = result["analyst_signals"]
            
            # Execute trades
            for ticker in self.config["UNIVERSE"]:
                if ticker not in current_prices:
                    continue
                
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action = decision.get("action", "hold")
                quantity = decision.get("quantity", 0)
                
                if action in ["buy", "sell", "short", "cover"]:
                    self.execute_trade(ticker, action, quantity, current_prices[ticker])
            
            # Calculate portfolio value after trades
            portfolio_value = self.get_portfolio_value(current_prices)
            
            # Record portfolio state
            portfolio_record = {
                "date": date,
                "cash": self.portfolio["cash"],
                "portfolio_value": portfolio_value,
                "margin_used": self.portfolio["margin_used"],
                "positions": {ticker: {
                    "long": self.portfolio["positions"][ticker]["long"],
                    "short": self.portfolio["positions"][ticker]["short"],
                    "current_price": current_prices.get(ticker, 0),
                    "long_value": self.portfolio["positions"][ticker]["long"] * current_prices.get(ticker, 0),
                    "short_value": self.portfolio["positions"][ticker]["short"] * current_prices.get(ticker, 0),
                } for ticker in self.config["UNIVERSE"] if ticker in current_prices}
            }
            self.portfolio_history.append(portfolio_record)
            
            self.logger.info(f"Trading cycle completed. Portfolio value: ${portfolio_value:.2f}")
            
            return {
                "date": date,
                "decisions": decisions,
                "analyst_signals": analyst_signals,
                "portfolio_value": portfolio_value,
                "portfolio_state": portfolio_record
            }
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            # Make sure progress tracking is stopped even if there's an error
            try:
                progress.stop()
            except:
                pass
            return None
    
    def backtest(self, start_date=None, end_date=None):
        """
        Run a backtest over a specified date range.
        
        Args:
            start_date (str, optional): Start date for backtest. Defaults to config value.
            end_date (str, optional): End date for backtest. Defaults to config value.
            
        Returns:
            dict: Results of the backtest.
        """
        # Use config dates if not specified
        start_date = start_date or self.config["BACKTEST_START_DATE"]
        end_date = end_date or self.config["BACKTEST_END_DATE"]
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Reset portfolio to initial state
        self._initialize_portfolio()
        self.trades_history = []
        self.portfolio_history = []
        
        # Initialize workflow if not already done
        if self.workflow is None:
            from main import create_workflow
            self.workflow = create_workflow(self.config["ANALYSTS"])
            self.app = self.workflow.compile()
        
        # Generate list of trading days
        dates = pd.date_range(start_date, end_date, freq="B")
        
        # Run trading cycle for each date
        results = []
        for i, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            result = self.run_trading_cycle(date_str)
            if result:
                results.append(result)
                
            # Add a pause between cycles to prevent rate limiting (every 5 days)
            if i % 5 == 4:
                self.logger.info("Pausing for 10 seconds to prevent rate limiting...")
                import time
                time.sleep(10)
        
        # Convert portfolio history to DataFrame
        if self.portfolio_history:
            portfolio_df = pl.DataFrame([
                {
                    "date": record["date"],
                    "portfolio_value": record["portfolio_value"],
                    "cash": record["cash"],
                    "margin_used": record["margin_used"]
                }
                for record in self.portfolio_history
            ])
            
            # Calculate performance metrics
            initial_value = self.config["INITIAL_CAPITAL"]
            final_value = self.portfolio_history[-1]["portfolio_value"]
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate daily returns
            portfolio_df = portfolio_df.with_columns(
                pl.col("portfolio_value").pct_change().alias("daily_return")
            )
            
            # Annualized return
            days = len(portfolio_df)
            annualized_return = (1 + total_return) ** (252 / days) - 1
            
            # Sharpe ratio (assuming risk-free rate of 4%)
            risk_free_rate = 0.04
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = portfolio_df["daily_return"] - daily_risk_free
            sharpe_ratio = None
            if len(excess_returns.drop_nulls()) > 0:
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * (252 ** 0.5)
            
            # Maximum drawdown
            portfolio_df = portfolio_df.with_columns(
                pl.col("portfolio_value").cummax().alias("peak_value")
            )
            portfolio_df = portfolio_df.with_columns(
                ((pl.col("portfolio_value") - pl.col("peak_value")) / pl.col("peak_value")).alias("drawdown")
            )
            max_drawdown = portfolio_df["drawdown"].min()
            
            backtest_results = {
                "start_date": start_date,
                "end_date": end_date,
                "initial_value": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "portfolio_history": self.portfolio_history,
                "trades_history": self.trades_history,
            }
            
            self.logger.info(f"Backtest completed. Total return: {total_return*100:.2f}%, Sharpe ratio: {sharpe_ratio:.2f}")
            
            return backtest_results
        else:
            self.logger.warning("No portfolio history found. Backtest may have failed.")
            return None
    
    def analyze_results(self, backtest_results=None):
        """
        Analyze and plot backtest results.
        
        Args:
            backtest_results (dict, optional): Results from backtest. If None, uses the last backtest.
            
        Returns:
            dict: Analysis metrics.
        """
        if backtest_results is None:
            if not self.portfolio_history:
                self.logger.error("No backtest results to analyze. Run backtest first.")
                return None
            
            # Create results dict from portfolio history
            initial_value = self.config["INITIAL_CAPITAL"]
            final_value = self.portfolio_history[-1]["portfolio_value"]
            total_return = (final_value - initial_value) / initial_value
            
            backtest_results = {
                "start_date": self.portfolio_history[0]["date"],
                "end_date": self.portfolio_history[-1]["date"],
                "initial_value": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "portfolio_history": self.portfolio_history,
                "trades_history": self.trades_history,
            }
        
        # Create DataFrame from portfolio history
        portfolio_df = pl.DataFrame([
            {
                "date": record["date"],
                "portfolio_value": record["portfolio_value"],
                "cash": record["cash"],
                "margin_used": record["margin_used"]
            }
            for record in backtest_results["portfolio_history"]
        ])
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 8))
        
        # Convert to pandas for plotting
        portfolio_pd = portfolio_df.to_pandas()
        portfolio_pd['date'] = pd.to_datetime(portfolio_pd['date'])
        portfolio_pd.set_index('date', inplace=True)
        
        plt.subplot(2, 1, 1)
        portfolio_pd['portfolio_value'].plot(color='blue', label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Plot cash and margin usage
        plt.subplot(2, 1, 2)
        portfolio_pd['cash'].plot(color='green', label='Cash')
        portfolio_pd['margin_used'].plot(color='red', label='Margin Used')
        plt.title('Cash and Margin Usage')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
        
        # Create trade analysis
        trades_df = pl.DataFrame([
            {
                "timestamp": trade["timestamp"],
                "ticker": trade["ticker"],
                "action": trade["action"],
                "quantity": trade["quantity"],
                "price": trade["price"],
                "value": trade["quantity"] * trade["price"]
            }
            for trade in backtest_results["trades_history"]
        ])
        
        # Print trade analysis
        print(f"\n{Fore.WHITE}{Style.BRIGHT}BACKTEST RESULTS:{Style.RESET_ALL}")
        print(f"Period: {backtest_results['start_date']} to {backtest_results['end_date']}")
        print(f"Initial value: ${backtest_results['initial_value']:,.2f}")
        print(f"Final value: ${backtest_results['final_value']:,.2f}")
        print(f"Total return: {Fore.GREEN if backtest_results['total_return'] >= 0 else Fore.RED}{backtest_results['total_return']*100:.2f}%{Style.RESET_ALL}")
        
        if "annualized_return" in backtest_results:
            print(f"Annualized return: {Fore.GREEN if backtest_results['annualized_return'] >= 0 else Fore.RED}{backtest_results['annualized_return']*100:.2f}%{Style.RESET_ALL}")
        
        if "sharpe_ratio" in backtest_results and backtest_results["sharpe_ratio"] is not None:
            print(f"Sharpe ratio: {Fore.GREEN if backtest_results['sharpe_ratio'] >= 1 else Fore.YELLOW}{backtest_results['sharpe_ratio']:.2f}{Style.RESET_ALL}")
        
        if "max_drawdown" in backtest_results:
            print(f"Maximum drawdown: {Fore.RED}{backtest_results['max_drawdown']*100:.2f}%{Style.RESET_ALL}")
        
        if len(trades_df) > 0:
            buy_trades = trades_df.filter(pl.col("action") == "buy")
            sell_trades = trades_df.filter(pl.col("action") == "sell")
            short_trades = trades_df.filter(pl.col("action") == "short")
            cover_trades = trades_df.filter(pl.col("action") == "cover")
            
            total_trades = len(trades_df)
            print(f"Total trades: {total_trades}")
            print(f"Buy trades: {len(buy_trades)}")
            print(f"Sell trades: {len(sell_trades)}")
            print(f"Short trades: {len(short_trades)}")
            print(f"Cover trades: {len(cover_trades)}")
            
            # Calculate win/loss ratio for closed positions
            if "realized_gain" in trades_df.columns:
                winning_trades = trades_df.filter(pl.col("realized_gain") > 0)
                losing_trades = trades_df.filter(pl.col("realized_gain") < 0)
                win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if len(winning_trades) + len(losing_trades) > 0 else 0
                print(f"Win rate: {win_rate*100:.2f}%")
                
                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    avg_win = winning_trades["realized_gain"].mean()
                    avg_loss = losing_trades["realized_gain"].mean()
                    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                    print(f"Average win: ${avg_win:.2f}, Average loss: ${avg_loss:.2f}, Win/Loss ratio: {win_loss_ratio:.2f}")
        
        # Save results to file
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save portfolio history
        pl.DataFrame([
            {
                "date": record["date"],
                "portfolio_value": record["portfolio_value"],
                "cash": record["cash"],
                "margin_used": record["margin_used"]
            }
            for record in backtest_results["portfolio_history"]
        ]).write_csv(output_dir / "portfolio_history.csv")
        
        # Save trades history
        if len(trades_df) > 0:
            trades_df.write_csv(output_dir / "trades_history.csv")
        
        return backtest_results
    
    def run_live_trading(self, days=1):
        """
        Run live trading for a specified number of days.
        
        Args:
            days (int): Number of days to run for. Default is 1.
            
        Returns:
            dict: Results of the live trading run.
        """
        self.logger.info(f"Starting live trading for {days} days")
        
        for i in range(days):
            # Get current date
            date = datetime.now().strftime("%Y-%m-%d")
            
            # Run trading cycle
            result = self.run_trading_cycle(date)
            
            if result:
                self.logger.info(f"Day {i+1}/{days} completed. Portfolio value: ${result['portfolio_value']:.2f}")
            else:
                self.logger.error(f"Day {i+1}/{days} failed.")
            
            # Sleep until next day if not the last day
            if i < days - 1:
                # In a real system, you would wait until the next trading day
                self.logger.info("Waiting for next trading day...")
                # time.sleep(86400)  # Sleep for 24 hours in seconds
        
        # Save final state
        output_dir = Path("live_trading_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save portfolio state
        with open(output_dir / "portfolio_state.json", "w") as f:
            json.dump(self.portfolio, f, indent=4)
        
        # Save portfolio history
        pl.DataFrame([
            {
                "date": record["date"],
                "portfolio_value": record["portfolio_value"],
                "cash": record["cash"],
                "margin_used": record["margin_used"]
            }
            for record in self.portfolio_history
        ]).write_csv(output_dir / "portfolio_history.csv")
        
        # Save trades history
        if self.trades_history:
            pl.DataFrame(self.trades_history).write_csv(output_dir / "trades_history.csv")
        
        return {
            "days_run": days,
            "final_portfolio_value": self.portfolio_history[-1]["portfolio_value"] if self.portfolio_history else None,
            "total_trades": len(self.trades_history),
            "portfolio_history": self.portfolio_history,
            "trades_history": self.trades_history
        }
    
    def save_state(self, filename="trading_system_state.json"):
        """Save the current state of the trading system."""
        state = {
            "portfolio": self.portfolio,
            "portfolio_history": self.portfolio_history,
            "trades_history": self.trades_history,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(state, f, indent=4)
        
        self.logger.info(f"Trading system state saved to {filename}")
    
    def load_state(self, filename="trading_system_state.json"):
        """Load the state of the trading system from a file."""
        if not os.path.exists(filename):
            self.logger.error(f"State file {filename} not found")
            return False
        
        try:
            with open(filename, "r") as f:
                state = json.load(f)
            
            self.portfolio = state["portfolio"]
            self.portfolio_history = state["portfolio_history"]
            self.trades_history = state["trades_history"]
            
            # Update config if present, but keep current config as fallback
            if "config" in state:
                for key, value in state["config"].items():
                    if key in self.config:
                        self.config[key] = value
            
            self.logger.info(f"Trading system state loaded from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False