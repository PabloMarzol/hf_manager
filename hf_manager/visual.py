#!/usr/bin/env python
"""
Trading System Performance Visualizer

This script provides visualizations for analyzing trading system performance,
including portfolio value, drawdowns, returns, and trading patterns.
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize trading system performance.")
    
    parser.add_argument("--portfolio-file", type=str, default="backtest_results/portfolio_history.csv",
                       help="Portfolio history CSV file (default: backtest_results/portfolio_history.csv)")
    parser.add_argument("--trades-file", type=str, default="backtest_results/trades_history.csv",
                       help="Trades history CSV file (default: backtest_results/trades_history.csv)")
    parser.add_argument("--state-file", type=str,
                       help="Trading system state file (alternative to CSV files)")
    parser.add_argument("--output-dir", type=str, default="analysis",
                       help="Directory to save output files (default: analysis)")
    parser.add_argument("--benchmark", type=str,
                       help="Benchmark ticker symbol (e.g., SPY)")
    
    return parser.parse_args()

def load_data(args):
    """Load data from files."""
    portfolio_df = None
    trades_df = None
    
    if args.state_file and os.path.exists(args.state_file):
        # Load from state file
        with open(args.state_file, "r") as f:
            state = json.load(f)
        
        portfolio_history = state.get("portfolio_history", [])
        trades_history = state.get("trades_history", [])
        
        if portfolio_history:
            portfolio_df = pl.DataFrame([
                {
                    "date": record["date"],
                    "portfolio_value": record["portfolio_value"],
                    "cash": record["cash"],
                    "margin_used": record.get("margin_used", 0)
                }
                for record in portfolio_history
            ])
        
        if trades_history:
            trades_df = pl.DataFrame(trades_history)
    else:
        # Load from CSV files
        if args.portfolio_file and os.path.exists(args.portfolio_file):
            portfolio_df = pl.read_csv(args.portfolio_file)
        
        if args.trades_file and os.path.exists(args.trades_file):
            trades_df = pl.read_csv(args.trades_file)
    
    return portfolio_df, trades_df

def calculate_metrics(portfolio_df, trades_df):
    """Calculate performance metrics."""
    metrics = {}
    
    if portfolio_df is not None and not portfolio_df.is_empty():
        # Make sure date is in datetime format
        if isinstance(portfolio_df["date"][0], str):
            portfolio_df = portfolio_df.with_columns(
                pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d").alias("date")
            )
        
        # Sort by date
        portfolio_df = portfolio_df.sort("date")
        
        # Calculate returns
        portfolio_df = portfolio_df.with_columns(
            pl.col("portfolio_value").pct_change().alias("daily_return")
        )
        
        # Calculate cumulative returns
        portfolio_df = portfolio_df.with_columns(
            (pl.col("portfolio_value") / portfolio_df["portfolio_value"].iloc[0] - 1).alias("cumulative_return")
        )
        
        # Calculate drawdowns
        portfolio_df = portfolio_df.with_columns(
            pl.col("portfolio_value").cummax().alias("peak_value")
        )
        portfolio_df = portfolio_df.with_columns(
            ((pl.col("portfolio_value") - pl.col("peak_value")) / pl.col("peak_value")).alias("drawdown")
        )
        
        # Calculate key metrics
        initial_value = portfolio_df["portfolio_value"].iloc[0]
        final_value = portfolio_df["portfolio_value"].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = len(portfolio_df)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        daily_returns = portfolio_df["daily_return"].drop_nulls()
        volatility = daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate of 4%)
        risk_free_rate = 0.04
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = daily_returns - daily_risk_free
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * (252 ** 0.5) if len(excess_returns) > 1 and excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        max_drawdown = portfolio_df["drawdown"].min()
        max_drawdown_date = portfolio_df.filter(pl.col("drawdown") == max_drawdown)["date"].iloc[0]
        
        # Calculate win/loss metrics if there are trades
        if trades_df is not None and not trades_df.is_empty():
            # Make sure timestamp is in datetime format
            if "timestamp" in trades_df.columns and isinstance(trades_df["timestamp"][0], str):
                trades_df = trades_df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime).alias("timestamp")
                )
            
            # Count trades by type
            buy_count = len(trades_df.filter(pl.col("action") == "buy"))
            sell_count = len(trades_df.filter(pl.col("action") == "sell"))
            short_count = len(trades_df.filter(pl.col("action") == "short"))
            cover_count = len(trades_df.filter(pl.col("action") == "cover"))
            
            # Calculate win/loss metrics if realized_gain column exists
            if "realized_gain" in trades_df.columns:
                winning_trades = trades_df.filter(pl.col("realized_gain") > 0)
                losing_trades = trades_df.filter(pl.col("realized_gain") < 0)
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                
                win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
                
                avg_win = winning_trades["realized_gain"].mean() if not winning_trades.is_empty() else 0
                avg_loss = losing_trades["realized_gain"].mean() if not losing_trades.is_empty() else 0
                
                win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                profit_factor = abs(winning_trades["realized_gain"].sum() / losing_trades["realized_gain"].sum()) if losing_trades["realized_gain"].sum() != 0 else float('inf')
                
                metrics.update({
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "win_loss_ratio": win_loss_ratio,
                    "profit_factor": profit_factor
                })
            
            metrics.update({
                "buy_count": buy_count,
                "sell_count": sell_count,
                "short_count": short_count,
                "cover_count": cover_count,
                "total_trades": len(trades_df)
            })
        
        metrics.update({
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_date": max_drawdown_date,
            "days": days
        })
    
    return metrics, portfolio_df, trades_df

def millions_formatter(x, pos):
    """Format numbers in millions."""
    return f'${x/1e6:.1f}M'

def thousands_formatter(x, pos):
    """Format numbers in thousands."""
    return f'${x/1e3:.1f}K'

def create_visualizations(portfolio_df, trades_df, metrics, args):
    """Create visualizations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if portfolio_df is None or portfolio_df.is_empty():
        print("No portfolio data to visualize")
        return
    
    # Convert to pandas for easier plotting
    portfolio_pd = portfolio_df.to_pandas()
    if isinstance(portfolio_pd['date'].iloc[0], str):
        portfolio_pd['date'] = pd.to_datetime(portfolio_pd['date'])
    portfolio_pd.set_index('date', inplace=True)
    
    # Set up the style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(font_scale=1.2)
    
    # 1. Portfolio Value Over Time
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    
    # Format y-axis based on portfolio value scale
    if metrics["final_value"] > 1e6:
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    # Plot portfolio value
    portfolio_pd['portfolio_value'].plot(ax=ax, linewidth=2, color='royalblue')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(portfolio_pd) > 30:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)
    
    # Add title and labels
    plt.title(f'Portfolio Drawdown\nMax Drawdown: {metrics["max_drawdown"]*100:.2f}% on {metrics["max_drawdown_date"].strftime("%Y-%m-%d")}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png")
    
    # 3. Returns Distribution
    if len(portfolio_pd) > 5:  # Need at least a few data points
        plt.figure(figsize=(12, 6))
        
        # Filter out NaN returns
        returns = portfolio_pd['daily_return'].dropna()
        
        # Plot histogram of returns
        sns.histplot(returns, bins=30, kde=True, color='royalblue')
        
        # Plot normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        y = np.exp(-(x - returns.mean())**2 / (2 * returns.std()**2)) / (returns.std() * np.sqrt(2 * np.pi))
        y = y / y.max() * returns.value_counts(bins=30, normalize=True).max()
        plt.plot(x, y, 'r--', linewidth=2)
        
        # Add vertical line at 0
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Add title and labels
        plt.title(f'Distribution of Daily Returns\nMean: {returns.mean()*100:.2f}%, Std Dev: {returns.std()*100:.2f}%', fontsize=16)
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / "returns_distribution.png")
    
    # 4. Equity Curve with Cash and Margin
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    
    # Format y-axis based on portfolio value scale
    if metrics["final_value"] > 1e6:
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    # Plot portfolio value, cash, and margin
    portfolio_pd['portfolio_value'].plot(ax=ax, linewidth=2, color='royalblue', label='Portfolio Value')
    portfolio_pd['cash'].plot(ax=ax, linewidth=1.5, color='green', label='Cash')
    
    if 'margin_used' in portfolio_pd.columns:
        portfolio_pd['margin_used'].plot(ax=ax, linewidth=1.5, color='red', label='Margin Used')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(portfolio_pd) > 30:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)
    
    # Add title and labels
    plt.title('Portfolio Composition Over Time', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_composition.png")
    
    # 5. Trade Analysis (if trade data available)
    if trades_df is not None and not trades_df.is_empty():
        # Convert to pandas
        trades_pd = trades_df.to_pandas()
        
        # A. Trade Actions Pie Chart
        plt.figure(figsize=(10, 10))
        
        action_counts = trades_pd['action'].value_counts()
        colors = ['royalblue', 'green', 'red', 'orange', 'purple']
        
        plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors, explode=[0.05] * len(action_counts))
        
        plt.axis('equal')
        plt.title('Trade Actions Distribution', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "trade_actions.png")
        
        # B. Trade P&L Analysis (if available)
        if 'realized_gain' in trades_pd.columns:
            plt.figure(figsize=(12, 8))
            
            # Filter trades with realized gains/losses
            trades_with_pnl = trades_pd[trades_pd['realized_gain'].notna() & 
                                        (trades_pd['action'].isin(['sell', 'cover']))]
            
            if not trades_with_pnl.empty:
                # Convert timestamp to datetime if needed
                if isinstance(trades_with_pnl['timestamp'].iloc[0], str):
                    trades_with_pnl['timestamp'] = pd.to_datetime(trades_with_pnl['timestamp'])
                
                # Sort by timestamp
                trades_with_pnl = trades_with_pnl.sort_values('timestamp')
                
                # Calculate cumulative P&L
                trades_with_pnl['cumulative_pnl'] = trades_with_pnl['realized_gain'].cumsum()
                
                # Plot cumulative P&L
                plt.plot(trades_with_pnl['timestamp'], trades_with_pnl['cumulative_pnl'], 
                        marker='o', linestyle='-', color='royalblue')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                
                # Format y-axis
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.2f}'))
                
                # Format x-axis dates
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                # Add title and labels
                plt.title('Cumulative Realized P&L', fontsize=16)
                plt.xlabel('Date')
                plt.ylabel('Profit/Loss ($)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / "cumulative_pnl.png")
                
                # C. Win/Loss Analysis
                plt.figure(figsize=(14, 7))
                
                # Create scatter plot of P&L
                winning_trades = trades_with_pnl[trades_with_pnl['realized_gain'] > 0]
                losing_trades = trades_with_pnl[trades_with_pnl['realized_gain'] < 0]
                
                # Calculate sizes based on amount (scaled for visibility)
                win_sizes = winning_trades['realized_gain'] / winning_trades['realized_gain'].max() * 200
                loss_sizes = abs(losing_trades['realized_gain']) / abs(losing_trades['realized_gain'].min()) * 200
                
                # Plot winning trades
                plt.scatter(winning_trades['timestamp'], winning_trades['realized_gain'], 
                        color='green', alpha=0.7, s=win_sizes, label='Winning Trades')
                
                # Plot losing trades
                plt.scatter(losing_trades['timestamp'], losing_trades['realized_gain'], 
                        color='red', alpha=0.7, s=loss_sizes, label='Losing Trades')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                
                # Format y-axis
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.2f}'))
                
                # Format x-axis dates
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                # Add title and labels
                win_rate = metrics.get('win_rate', 0) * 100
                plt.title(f'Individual Trade P&L\nWin Rate: {win_rate:.1f}%', fontsize=16)
                plt.xlabel('Date')
                plt.ylabel('Profit/Loss ($)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / "trade_pnl.png")
    
    # 6. Performance Metrics Summary
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    
    # Create a text summary of metrics
    text = f"""
    Performance Metrics Summary:
    
    Time Period: {portfolio_pd.index[0].strftime('%Y-%m-%d')} to {portfolio_pd.index[-1].strftime('%Y-%m-%d')} ({metrics['days']} days)
    
    Portfolio Value:
    • Initial: ${metrics['initial_value']:,.2f}
    • Final: ${metrics['final_value']:,.2f}
    • Total Return: {metrics['total_return']*100:.2f}%
    • Annualized Return: {metrics['annualized_return']*100:.2f}%
    
    Risk Metrics:
    • Volatility (Annualized): {metrics['volatility']*100:.2f}%
    • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    • Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
    """
    
    # Add trading metrics if available
    if 'total_trades' in metrics:
        text += f"""
    Trading Activity:
    • Total Trades: {metrics['total_trades']}
    • Buys: {metrics.get('buy_count', 0)}
    • Sells: {metrics.get('sell_count', 0)}
    • Shorts: {metrics.get('short_count', 0)}
    • Covers: {metrics.get('cover_count', 0)}
    """
    
    # Add win/loss metrics if available
    if 'win_rate' in metrics:
        text += f"""
    Win/Loss Statistics:
    • Win Rate: {metrics['win_rate']*100:.1f}%
    • Profit Factor: {metrics['profit_factor']:.2f}
    • Average Win: ${metrics['avg_win']:,.2f}
    • Average Loss: ${metrics['avg_loss']:,.2f}
    • Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}
    """
    
    plt.text(0.05, 0.95, text, fontsize=14, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png")
    
    # 7. Create metrics JSON file
    metrics_output = {k: v for k, v in metrics.items() if not isinstance(v, (datetime, np.datetime64))}
    # Convert datetime objects to strings
    for key, value in metrics.items():
        if isinstance(value, (datetime, np.datetime64)):
            metrics_output[key] = value.strftime('%Y-%m-%d')
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=4)
    
    print(f"Visualizations created in {output_dir}/")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load data
    portfolio_df, trades_df = load_data(args)
    
    if portfolio_df is None:
        print("Error: No portfolio data found.")
        return
    
    # Calculate metrics
    metrics, portfolio_df, trades_df = calculate_metrics(portfolio_df, trades_df)
    
    # Create visualizations
    create_visualizations(portfolio_df, trades_df, metrics, args)
    
    # Print metrics summary
    print("\nPerformance Metrics Summary:")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    if 'win_rate' in metrics:
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\nDetailed analysis saved in {args.output_dir}/")

if __name__ == "__main__":
    main()

    plt.title(f'Portfolio Value Over Time\nTotal Return: {metrics["total_return"]*100:.2f}%', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_value.png")
    
    # 2. Drawdown Analysis
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    
    portfolio_pd['drawdown'].plot(ax=ax, color='crimson', linewidth=2)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(portfolio_pd) > 30:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
