"""
Pydantic models for financial data.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Price(BaseModel):
    """Price data model."""
    time: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None


class PriceResponse(BaseModel):
    """Response model for price data API."""
    prices: List[Price]


class FinancialMetrics(BaseModel):
    """Financial metrics model."""
    ticker: str
    report_period: str
    price_to_earnings_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    net_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    book_value_growth: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None


class FinancialMetricsResponse(BaseModel):
    """Response model for financial metrics API."""
    financial_metrics: List[FinancialMetrics]


class LineItem(BaseModel):
    """Financial line item model."""
    ticker: str
    period: str
    report_period: str
    # Standard line items
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    earnings_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    free_cash_flow: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_debt: Optional[float] = None
    shareholders_equity: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    operating_income: Optional[float] = None
    outstanding_shares: Optional[float] = None
    dividends_and_other_cash_distributions: Optional[float] = None
    issuance_or_purchase_of_equity_shares: Optional[float] = None
    # Additional line items
    return_on_invested_capital: Optional[float] = None
    capital_expenditure: Optional[float] = None
    operating_expense: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    research_and_development: Optional[float] = None
    goodwill_and_intangible_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    working_capital: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    
    def __getattr__(self, name):
        """Allow access to attributes that don't exist, returning None."""
        return None


class LineItemResponse(BaseModel):
    """Response model for line item search API."""
    search_results: List[LineItem]


class InsiderTrade(BaseModel):
    """Insider trade model."""
    ticker: str
    filing_date: str
    transaction_date: Optional[str] = None
    insider_name: Optional[str] = None
    relationship: Optional[str] = None
    transaction_type: Optional[str] = None
    transaction_shares: Optional[float] = None
    transaction_price: Optional[float] = None
    shares_owned: Optional[float] = None


class InsiderTradeResponse(BaseModel):
    """Response model for insider trades API."""
    insider_trades: List[InsiderTrade]


class CompanyNews(BaseModel):
    """Company news model."""
    ticker: str
    date: str
    title: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    sentiment: Optional[str] = None


class CompanyNewsResponse(BaseModel):
    """Response model for company news API."""
    news: List[CompanyNews]