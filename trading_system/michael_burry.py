from typing import Dict, List, Any
from datetime import datetime, timedelta
from signals.data.models import AnalystSignal
from signals.utils.progress import progress

class MichaelBurryAgent:
    """
    Analyzes stocks using Michael Burry's investing principles:
    - Deep value approach focusing on severely undervalued companies
    - Contrarian perspective that goes against market consensus
    - Detailed fundamental analysis with focus on balance sheet
    - Prefers tangible assets and companies trading below liquidation value
    - Looks for catalysts that will unlock value
    """
    
    def __init__(self):
        self.name = "michael_burry"
    
    def analyze(self, tickers: List[str], data_fetcher, end_date, start_date=None) -> Dict[str, AnalystSignal]:
        """Generate signals for multiple tickers based on Burry's principles."""
        signals = {}
        
        # We look one year back for insider trades / news flow if not provided
        if not start_date:
            start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()
        
        for ticker in tickers:
            # Fetch data
            progress.update_status(f"{self.name}_agent", ticker, "Fetching financial metrics")
            metrics = data_fetcher.get_financial_metrics(ticker, end_date)
            
            progress.update_status(f"{self.name}_agent", ticker, "Fetching line items")
            line_items = data_fetcher.get_line_items(
                ticker, 
                end_date,
                line_items=[
                    "free_cash_flow",
                    "net_income",
                    "total_debt",
                    "cash_and_equivalents",
                    "total_assets",
                    "total_liabilities",
                    "outstanding_shares",
                    "issuance_or_purchase_of_equity_shares",
                ]
            )
            
            progress.update_status(f"{self.name}_agent", ticker, "Fetching insider trades")
            insider_trades = data_fetcher.get_insider_trades(ticker, end_date, start_date)
            
            progress.update_status(f"{self.name}_agent", ticker, "Fetching company news")
            news = data_fetcher.get_company_news(ticker, end_date, start_date)
            
            progress.update_status(f"{self.name}_agent", ticker, "Fetching market cap")
            market_cap = data_fetcher.get_market_cap(ticker, end_date)
            
            # Run sub-analyses
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing value")
            value_analysis = self._analyze_value(metrics, line_items, market_cap)
            
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing balance sheet")
            balance_sheet_analysis = self._analyze_balance_sheet(metrics, line_items)
            
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing insider activity")
            insider_analysis = self._analyze_insider_activity(insider_trades)
            
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing contrarian sentiment")
            contrarian_analysis = self._analyze_contrarian_sentiment(news)
            
            # Combine sub-analyses with weights
            total_score = (
                value_analysis["score"] * 0.40 +
                balance_sheet_analysis["score"] * 0.30 +
                insider_analysis["score"] * 0.15 +
                contrarian_analysis["score"] * 0.15
            )
            
            max_score = (
                value_analysis["max_score"] * 0.40 +
                balance_sheet_analysis["max_score"] * 0.30 +
                insider_analysis["max_score"] * 0.15 +
                contrarian_analysis["max_score"] * 0.15
            )
            
            # Generate signal based on normalized score
            normalized_score = (total_score / max_score) * 10 if max_score > 0 else 0
            
            if normalized_score >= 7.5:
                signal = "bullish"
                confidence = min(85, 50 + (normalized_score - 7.5) * 10)
            elif normalized_score <= 4.5:
                signal = "bearish"
                confidence = min(85, 50 + (4.5 - normalized_score) * 10)
            else:
                signal = "neutral" 
                confidence = 50
                
            # Create reasoning structure
            reasoning = {
                "deep_value": value_analysis,
                "balance_sheet": balance_sheet_analysis,
                "insider_activity": insider_analysis,
                "contrarian_sentiment": contrarian_analysis,
                "total_score": normalized_score,
                "max_score": 10.0
            }
            
            signals[ticker] = AnalystSignal(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning
            )
            
            progress.update_status(f"{self.name}_agent", ticker, "Done")
            
        return signals
    
    def _latest_line_item(self, line_items: list):
        """Return the most recent line‑item object or *None*."""
        return line_items[0] if line_items else None
    
    def _analyze_value(self, metrics, line_items, market_cap):
        """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""
        max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
        score = 0
        details = []
        
        # Free‑cash‑flow yield
        latest_item = self._latest_line_item(line_items)
        fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
        if fcf is not None and market_cap:
            fcf_yield = fcf / market_cap
            if fcf_yield >= 0.15:
                score += 4
                details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
            elif fcf_yield >= 0.12:
                score += 3
                details.append(f"Very high FCF yield {fcf_yield:.1%}")
            elif fcf_yield >= 0.08:
                score += 2
                details.append(f"Respectable FCF yield {fcf_yield:.1%}")
            else:
                details.append(f"Low FCF yield {fcf_yield:.1%}")
        else:
            details.append("FCF data unavailable")
        
        # EV/EBIT (from financial metrics)
        if metrics:
            ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
            if ev_ebit is not None:
                if ev_ebit < 6:
                    score += 2
                    details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
                elif ev_ebit < 10:
                    score += 1
                    details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
                else:
                    details.append(f"High EV/EBIT {ev_ebit:.1f}")
            else:
                details.append("EV/EBIT data unavailable")
        else:
            details.append("Financial metrics unavailable")
        
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}
    
    def _analyze_balance_sheet(self, metrics, line_items):
        """Leverage and liquidity checks."""
        max_score = 3
        score = 0
        details = []
        
        latest_metrics = metrics[0] if metrics else None
        latest_item = self._latest_line_item(line_items)
        
        debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 2
                details.append(f"Low D/E {debt_to_equity:.2f}")
            elif debt_to_equity < 1:
                score += 1
                details.append(f"Moderate D/E {debt_to_equity:.2f}")
            else:
                details.append(f"High leverage D/E {debt_to_equity:.2f}")
        else:
            details.append("Debt‑to‑equity data unavailable")
        
        # Quick liquidity sanity check (cash vs total debt)
        if latest_item is not None:
            cash = getattr(latest_item, "cash_and_equivalents", None)
            total_debt = getattr(latest_item, "total_debt", None)
            if cash is not None and total_debt is not None:
                if cash > total_debt:
                    score += 1
                    details.append("Net cash position")
                else:
                    details.append("Net debt position")
            else:
                details.append("Cash/debt data unavailable")
        
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}
    
    def _analyze_insider_activity(self, insider_trades):
        """Net insider buying over the last 12 months acts as a hard catalyst."""
        max_score = 2
        score = 0
        details = []
        
        if not insider_trades:
            details.append("No insider trade data")
            return {"score": score, "max_score": max_score, "details": "; ".join(details)}
        
        shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
        shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
        net = shares_bought - shares_sold
        if net > 0:
            score += 2 if net / max(shares_sold, 1) > 1 else 1
            details.append(f"Net insider buying of {net:,} shares")
        else:
            details.append("Net insider selling")
        
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}
    
    def _analyze_contrarian_sentiment(self, news):
        """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""
        max_score = 1
        score = 0
        details = []
        
        if not news:
            details.append("No recent news")
            return {"score": score, "max_score": max_score, "details": "; ".join(details)}
        
        # Count negative sentiment articles
        sentiment_negative_count = sum(
            1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
        )
        
        if sentiment_negative_count >= 5:
            score += 1  # The more hated, the better (assuming fundamentals hold up)
            details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
        else:
            details.append("Limited negative press")
        
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}