from typing import Dict, List, Any
from signals.data.models import AnalystSignal
from signals.utils.progress import progress

class BillAckmanAgent:
    """
    Analyzes stocks using Bill Ackman's investing principles:
    - Focuses on high-quality, simple, dominant businesses
    - Looks for catalysts and potential for activist engagement
    - Concentration on high-conviction ideas
    - Focus on free cash flow generation
    """
    
    def __init__(self):
        self.name = "bill_ackman"
    
    def analyze(self, tickers: List[str], data_fetcher, end_date, start_date=None) -> Dict[str, AnalystSignal]:
        """Generate signals for multiple tickers based on Ackman's principles."""
        signals = {}
        
        for ticker in tickers:
            progress.update_status(f"{self.name}_agent", ticker, "Fetching financial metrics")
            
            # Fetch required data
            metrics = data_fetcher.get_financial_metrics(ticker, end_date)
            financial_line_items = data_fetcher.get_line_items(
                ticker, 
                end_date, 
                line_items=[
                    "revenue",
                    "operating_margin",
                    "debt_to_equity",
                    "free_cash_flow",
                    "total_assets",
                    "total_liabilities",
                    "dividends_and_other_cash_distributions",
                    "outstanding_shares",
                ]
            )
            market_cap = data_fetcher.get_market_cap(ticker, end_date)
            
            # Analyze business quality
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing business quality")
            quality_analysis = self._analyze_business_quality(metrics, financial_line_items)
            
            # Analyze financial discipline and balance sheet
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing balance sheet and capital structure")
            balance_sheet_analysis = self._analyze_financial_discipline(metrics, financial_line_items)
            
            # Analyze activism potential
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing activism potential")
            activism_analysis = self._analyze_activism_potential(financial_line_items)
            
            # Analyze valuation
            progress.update_status(f"{self.name}_agent", ticker, "Calculating intrinsic value & margin of safety")
            valuation_analysis = self._analyze_valuation(financial_line_items, market_cap)
            
            # Combine sub-analyses with weights
            total_score = (
                quality_analysis["score"] * 0.30 +
                balance_sheet_analysis["score"] * 0.25 +
                activism_analysis["score"] * 0.25 +
                valuation_analysis["score"] * 0.20
            )
            
            # Generate signal based on total score
            if total_score >= 7.5:
                signal = "bullish"
                confidence = min(90, 50 + (total_score - 7.5) * 10)
            elif total_score <= 4.5:
                signal = "bearish"
                confidence = min(90, 50 + (4.5 - total_score) * 10)
            else:
                signal = "neutral" 
                confidence = 50
                
            # Create reasoning structure
            reasoning = {
                "business_quality": quality_analysis,
                "financial_discipline": balance_sheet_analysis,
                "activism_potential": activism_analysis,
                "valuation": valuation_analysis,
                "total_score": total_score
            }
            
            signals[ticker] = AnalystSignal(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning
            )
            
            progress.update_status(f"{self.name}_agent", ticker, "Done")
            
        return signals
    
    def _analyze_business_quality(self, metrics, financial_line_items):
        """
        Analyze whether the company has a high-quality business with stable or growing cash flows,
        durable competitive advantages (moats), and potential for long-term growth.
        """
        score = 0
        details = []
        
        if not metrics or not financial_line_items:
            return {
                "score": 0,
                "details": "Insufficient data to analyze business quality"
            }
        
        # 1. Multi-period revenue growth analysis
        revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
        if len(revenues) >= 2:
            initial, final = revenues[0], revenues[-1]
            if initial and final and final > initial:
                growth_rate = (final - initial) / abs(initial)
                if growth_rate > 0.5:  # e.g., 50% cumulative growth
                    score += 2
                    details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
                else:
                    score += 1
                    details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
            else:
                details.append("Revenue did not grow significantly or data insufficient.")
        else:
            details.append("Not enough revenue data for multi-period trend.")
        
        # 2. Operating margin and free cash flow consistency
        fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
        op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
        
        if op_margin_vals:
            above_15 = sum(1 for m in op_margin_vals if m > 0.15)
            if above_15 >= (len(op_margin_vals) // 2 + 1):
                score += 2
                details.append("Operating margins have often exceeded 15% (indicates good profitability).")
            else:
                details.append("Operating margin not consistently above 15%.")
        else:
            details.append("No operating margin data across periods.")
        
        if fcf_vals:
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
                score += 1
                details.append("Majority of periods show positive free cash flow.")
            else:
                details.append("Free cash flow not consistently positive.")
        else:
            details.append("No free cash flow data across periods.")
        
        # 3. Return on Equity (ROE) check from the latest metrics
        if metrics:
            latest_metrics = metrics[0]
            if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
                score += 2
                details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating a competitive advantage.")
            elif latest_metrics.return_on_equity:
                details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
            else:
                details.append("ROE data not available.")
        
        return {
            "score": score,
            "details": "; ".join(details)
        }
    
    def _analyze_financial_discipline(self, metrics, financial_line_items):
        """
        Evaluate the company's balance sheet over multiple periods:
        - Debt ratio trends
        - Capital returns to shareholders over time (dividends, buybacks)
        """
        score = 0
        details = []
        
        if not metrics or not financial_line_items:
            return {
                "score": 0,
                "details": "Insufficient data to analyze financial discipline"
            }
        
        # 1. Multi-period debt ratio or debt_to_equity
        debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
        if debt_to_equity_vals:
            below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
            if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append("Debt-to-equity < 1.0 for the majority of periods (reasonable leverage).")
            else:
                details.append("Debt-to-equity >= 1.0 in many periods (could be high leverage).")
        else:
            # Fallback to total_liabilities / total_assets
            liab_to_assets = []
            for item in financial_line_items:
                if item.total_liabilities and item.total_assets and item.total_assets > 0:
                    liab_to_assets.append(item.total_liabilities / item.total_assets)
            
            if liab_to_assets:
                below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
                if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                    score += 2
                    details.append("Liabilities-to-assets < 50% for majority of periods.")
                else:
                    details.append("Liabilities-to-assets >= 50% in many periods.")
            else:
                details.append("No consistent leverage ratio data available.")
        
        # 2. Capital allocation approach (dividends + share counts)
        dividends_list = [
            item.dividends_and_other_cash_distributions
            for item in financial_line_items
            if item.dividends_and_other_cash_distributions is not None
        ]
        if dividends_list:
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append("Company has a history of returning capital to shareholders (dividends).")
            else:
                details.append("Dividends not consistently paid or no data on distributions.")
        else:
            details.append("No dividend data found across periods.")
        
        # Check for decreasing share count (simple approach)
        shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
        if len(shares) >= 2:
            if shares[-1] < shares[0]:
                score += 1
                details.append("Outstanding shares have decreased over time (possible buybacks).")
            else:
                details.append("Outstanding shares have not decreased over the available periods.")
        else:
            details.append("No multi-period share count data to assess buybacks.")
        
        return {
            "score": score,
            "details": "; ".join(details)
        }
    
    def _analyze_activism_potential(self, financial_line_items):
        """
        Bill Ackman often engages in activism if a company has a decent brand or moat
        but is underperforming operationally.
        """
        if not financial_line_items:
            return {
                "score": 0,
                "details": "Insufficient data for activism potential"
            }
        
        # Check revenue growth vs. operating margin
        revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
        op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
        
        if len(revenues) < 2 or not op_margins:
            return {
                "score": 0,
                "details": "Not enough data to assess activism potential (need multi-year revenue + margins)."
            }
        
        initial, final = revenues[0], revenues[-1]
        revenue_growth = (final - initial) / abs(initial) if initial else 0
        avg_margin = sum(op_margins) / len(op_margins)
        
        score = 0
        details = []
        
        # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
        if revenue_growth > 0.15 and avg_margin < 0.10:
            score += 2
            details.append(
                f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin*100:.1f}%). "
                "Activism could unlock margin improvements."
            )
        else:
            details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")
        
        return {"score": score, "details": "; ".join(details)}
    
    def _analyze_valuation(self, financial_line_items, market_cap):
        """
        Ackman invests in companies trading at a discount to intrinsic value.
        Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
        """
        if not financial_line_items or market_cap is None:
            return {
                "score": 0,
                "details": "Insufficient data to perform valuation"
            }
        
        if not financial_line_items:
            return {
                "score": 0,
                "details": "No financial line items available"
            }
            
        latest = financial_line_items[0]
        fcf = latest.free_cash_flow if latest.free_cash_flow else 0
        
        if fcf <= 0:
            return {
                "score": 0,
                "details": f"No positive FCF for valuation; FCF = {fcf}",
                "intrinsic_value": None
            }
        
        # Basic DCF assumptions
        growth_rate = 0.06
        discount_rate = 0.10
        terminal_multiple = 15
        projection_years = 5
        
        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / ((1 + discount_rate) ** year)
            present_value += pv
        
        # Terminal Value
        terminal_value = (
            fcf * (1 + growth_rate) ** projection_years * terminal_multiple
        ) / ((1 + discount_rate) ** projection_years)
        
        intrinsic_value = present_value + terminal_value
        margin_of_safety = (intrinsic_value - market_cap) / market_cap
        
        score = 0
        # Simple scoring
        if margin_of_safety > 0.3:
            score += 3
        elif margin_of_safety > 0.1:
            score += 1
        
        details = [
            f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
            f"Market cap: ~{market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}"
        ]
        
        return {
            "score": score,
            "details": "; ".join(details),
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety
        }