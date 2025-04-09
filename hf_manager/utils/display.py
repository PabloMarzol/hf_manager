"""
Display utilities for formatting trading output.
"""

from colorama import Fore, Style, init
from tabulate import tabulate

# Initialize colorama
init(autoreset=True)

def print_trading_output(result):
    """
    Print trading decisions in a formatted table.
    
    Args:
        result (dict): Trading result with decisions and analyst signals
    """
    if not result or "decisions" not in result:
        print("No trading decisions available.")
        return
    
    decisions = result["decisions"]
    analyst_signals = result.get("analyst_signals", {})
    
    # Create a table of decisions
    headers = ["Ticker", "Action", "Quantity", "Confidence", "Reasoning"]
    rows = []
    
    for ticker, decision in decisions.items():
        # Format action with color
        action = decision.get("action", "hold")
        if action == "buy":
            action_str = f"{Fore.GREEN}{action.upper()}{Style.RESET_ALL}"
        elif action == "sell":
            action_str = f"{Fore.RED}{action.upper()}{Style.RESET_ALL}"
        elif action == "short":
            action_str = f"{Fore.MAGENTA}{action.upper()}{Style.RESET_ALL}"
        elif action == "cover":
            action_str = f"{Fore.YELLOW}{action.upper()}{Style.RESET_ALL}"
        else:
            action_str = f"{Fore.BLUE}{action.upper()}{Style.RESET_ALL}"
        
        # Format confidence
        confidence = decision.get("confidence", 0)
        if confidence > 80:
            confidence_str = f"{Fore.GREEN}{confidence:.1f}%{Style.RESET_ALL}"
        elif confidence > 50:
            confidence_str = f"{Fore.YELLOW}{confidence:.1f}%{Style.RESET_ALL}"
        else:
            confidence_str = f"{Fore.RED}{confidence:.1f}%{Style.RESET_ALL}"
        
        # Get reasoning (truncate if too long)
        reasoning = decision.get("reasoning", "")
        if len(reasoning) > 100:
            reasoning = reasoning[:97] + "..."
        
        # Add row to table
        rows.append([ticker, action_str, decision.get("quantity", 0), confidence_str, reasoning])
    
    # Print the decisions table
    print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISIONS:{Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Print analyst signals summary if available
    if analyst_signals:
        print(f"\n{Fore.WHITE}{Style.BRIGHT}ANALYST SIGNALS SUMMARY:{Style.RESET_ALL}")
        
        # Get all tickers
        all_tickers = list(decisions.keys())
        
        # Get all analysts
        all_analysts = list(analyst_signals.keys())
        
        # Create a signals summary table
        signals_headers = ["Ticker"] + [a.replace("_agent", "").title() for a in all_analysts]
        signals_rows = []
        
        for ticker in all_tickers:
            row = [ticker]
            
            for analyst in all_analysts:
                if ticker in analyst_signals.get(analyst, {}):
                    signal = analyst_signals[analyst][ticker].get("signal", "")
                    confidence = analyst_signals[analyst][ticker].get("confidence", 0)
                    
                    if signal == "bullish":
                        signal_str = f"{Fore.GREEN}↑ {confidence:.0f}%{Style.RESET_ALL}"
                    elif signal == "bearish":
                        signal_str = f"{Fore.RED}↓ {confidence:.0f}%{Style.RESET_ALL}"
                    else:
                        signal_str = f"{Fore.BLUE}→ {confidence:.0f}%{Style.RESET_ALL}"
                    
                    row.append(signal_str)
                else:
                    row.append("")
            
            signals_rows.append(row)
        
        print(tabulate(signals_rows, headers=signals_headers, tablefmt="grid"))

def format_backtest_row(
    date, ticker, action, quantity, price, shares_owned, position_value,
    bullish_count, bearish_count, neutral_count, 
    is_summary=False, total_value=None, return_pct=None, cash_balance=None,
    total_position_value=None, sharpe_ratio=None, sortino_ratio=None, max_drawdown=None
):
    """
    Format a row for the backtest results table.
    
    Args:
        date (str): Date of the trade
        ticker (str): Ticker symbol
        action (str): Action taken (buy, sell, short, cover, hold)
        quantity (int): Number of shares traded
        price (float): Price per share
        shares_owned (int): Current shares owned
        position_value (float): Current position value
        bullish_count (int): Number of bullish signals
        bearish_count (int): Number of bearish signals
        neutral_count (int): Number of neutral signals
        is_summary (bool): Whether this is a summary row
        total_value (float): Total portfolio value
        return_pct (float): Return percentage
        cash_balance (float): Current cash balance
        total_position_value (float): Total position value
        sharpe_ratio (float): Sharpe ratio
        sortino_ratio (float): Sortino ratio
        max_drawdown (float): Maximum drawdown
        
    Returns:
        dict: Formatted row
    """
    if is_summary:
        return {
            "date": date,
            "is_summary": True,
            "total_value": total_value,
            "return_pct": return_pct,
            "cash_balance": cash_balance,
            "total_position_value": total_position_value,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown
        }
    else:
        return {
            "date": date,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "shares_owned": shares_owned,
            "position_value": position_value,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "is_summary": False
        }

def print_backtest_results(table_rows):
    """
    Print backtest results in a formatted table.
    
    Args:
        table_rows (list): List of formatted rows
    """
    if not table_rows:
        print("No backtest results available.")
        return
    
    # Clear screen
    print("\033c", end="")
    
    # Group rows by date
    dates = {}
    for row in table_rows:
        date = row["date"]
        if date not in dates:
            dates[date] = []
        dates[date].append(row)
    
    # Print each date's results
    for date, rows in dates.items():
        # Print date header
        print(f"\n{Fore.CYAN}{Style.BRIGHT}DATE: {date}{Style.RESET_ALL}")
        
        # Get regular rows and summary row
        regular_rows = [r for r in rows if not r.get("is_summary", False)]
        summary_rows = [r for r in rows if r.get("is_summary", False)]
        
        # Print trade details if any
        if regular_rows:
            headers = ["Ticker", "Action", "Qty", "Price", "Owned", "Value", "Signals (↑/↓/→)"]
            table_data = []
            
            for row in regular_rows:
                # Format action with color
                action = row.get("action", "")
                if action == "buy":
                    action_str = f"{Fore.GREEN}{action.upper()}{Style.RESET_ALL}"
                elif action == "sell":
                    action_str = f"{Fore.RED}{action.upper()}{Style.RESET_ALL}"
                elif action == "short":
                    action_str = f"{Fore.MAGENTA}{action.upper()}{Style.RESET_ALL}"
                elif action == "cover":
                    action_str = f"{Fore.YELLOW}{action.upper()}{Style.RESET_ALL}"
                elif action == "hold":
                    action_str = f"{Fore.BLUE}{action.upper()}{Style.RESET_ALL}"
                else:
                    action_str = ""
                
                # Format signals
                signals = f"{Fore.GREEN}{row.get('bullish_count', 0)}{Style.RESET_ALL}/{Fore.RED}{row.get('bearish_count', 0)}{Style.RESET_ALL}/{Fore.BLUE}{row.get('neutral_count', 0)}{Style.RESET_ALL}"
                
                # Format position value
                position_value = row.get("position_value", 0)
                if position_value > 0:
                    value_str = f"${position_value:,.2f}"
                else:
                    value_str = f"${abs(position_value):,.2f}" if position_value < 0 else "$0.00"
                
                table_data.append([
                    row.get("ticker", ""),
                    action_str,
                    row.get("quantity", 0),
                    f"${row.get('price', 0):,.2f}",
                    row.get("shares_owned", 0),
                    value_str,
                    signals
                ])
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print summary if available
        if summary_rows:
            summary = summary_rows[0]
            print(f"\n{Fore.WHITE}{Style.BRIGHT}SUMMARY:{Style.RESET_ALL}")
            print(f"Portfolio Value: ${summary.get('total_value', 0):,.2f}")
            print(f"Cash Balance: ${summary.get('cash_balance', 0):,.2f}")
            print(f"Position Value: ${summary.get('total_position_value', 0):,.2f}")
            
            return_pct = summary.get("return_pct")
            if return_pct is not None:
                if return_pct >= 0:
                    print(f"Return: {Fore.GREEN}+{return_pct:.2f}%{Style.RESET_ALL}")
                else:
                    print(f"Return: {Fore.RED}{return_pct:.2f}%{Style.RESET_ALL}")
            
            sharpe = summary.get("sharpe_ratio")
            if sharpe is not None:
                print(f"Sharpe Ratio: {sharpe:.2f}")
            
            max_dd = summary.get("max_drawdown")
            if max_dd is not None:
                print(f"Max Drawdown: {Fore.RED}{max_dd*100:.2f}%{Style.RESET_ALL}")
    
    print("\n" + "=" * 80)