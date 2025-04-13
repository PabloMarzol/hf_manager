import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from signals.data.models import AnalystSignal
from signals.utils.progress import progress

class TechnicalAnalystAgent:
    """
    Sophisticated technical analysis system that combines multiple trading strategies:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    
    def __init__(self):
        self.name = "technical_analyst"
        # Strategy weights for signal combination
        self.strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }
    
    def analyze(self, tickers: List[str], data_fetcher, end_date, start_date=None) -> Dict[str, AnalystSignal]:
        """Generate signals for multiple tickers based on technical analysis."""
        signals = {}
        
        for ticker in tickers:
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing price data")
            
            # Get price data
            prices_df = data_fetcher.get_prices(ticker, start_date, end_date)
            
            if prices_df.empty:
                progress.update_status(f"{self.name}_agent", ticker, "Failed: No price data found")
                continue
            
            # Calculate individual technical signals
            progress.update_status(f"{self.name}_agent", ticker, "Calculating trend signals")
            trend_signals = self._calculate_trend_signals(prices_df)
            
            progress.update_status(f"{self.name}_agent", ticker, "Calculating mean reversion")
            mean_reversion_signals = self._calculate_mean_reversion_signals(prices_df)
            
            progress.update_status(f"{self.name}_agent", ticker, "Calculating momentum")
            momentum_signals = self._calculate_momentum_signals(prices_df)
            
            progress.update_status(f"{self.name}_agent", ticker, "Analyzing volatility")
            volatility_signals = self._calculate_volatility_signals(prices_df)
            
            progress.update_status(f"{self.name}_agent", ticker, "Statistical analysis")
            stat_arb_signals = self._calculate_stat_arb_signals(prices_df)
            
            # Combine all signals using weighted ensemble approach
            progress.update_status(f"{self.name}_agent", ticker, "Combining signals")
            combined_signal = self._weighted_signal_combination(
                {
                    "trend": trend_signals,
                    "mean_reversion": mean_reversion_signals,
                    "momentum": momentum_signals,
                    "volatility": volatility_signals,
                    "stat_arb": stat_arb_signals,
                }
            )
            
            # Create strategy signals for reasoning
            strategy_signals = {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": trend_signals["metrics"],
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": mean_reversion_signals["metrics"],
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": momentum_signals["metrics"],
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": volatility_signals["metrics"],
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": stat_arb_signals["metrics"],
                },
            }
            
            # Create the final signal
            signals[ticker] = AnalystSignal(
                signal=combined_signal["signal"],
                confidence=round(combined_signal["confidence"] * 100),
                reasoning=strategy_signals
            )
            
            progress.update_status(f"{self.name}_agent", ticker, "Done")
        
        return signals

    def _calculate_trend_signals(self, prices_df):
        """Advanced trend following strategy using multiple timeframes and indicators"""
        # Calculate EMAs for multiple timeframes
        ema_8 = self._calculate_ema(prices_df, 8)
        ema_21 = self._calculate_ema(prices_df, 21)
        ema_55 = self._calculate_ema(prices_df, 55)

        # Calculate ADX for trend strength
        adx = self._calculate_adx(prices_df, 14)

        # Determine trend direction and strength
        short_trend = ema_8 > ema_21
        medium_trend = ema_21 > ema_55

        # Combine signals with confidence weighting
        trend_strength = adx["adx"].iloc[-1] / 100.0

        if short_trend.iloc[-1] and medium_trend.iloc[-1]:
            signal = "bullish"
            confidence = trend_strength
        elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
            signal = "bearish"
            confidence = trend_strength
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "adx": float(adx["adx"].iloc[-1]),
                "trend_strength": float(trend_strength),
            },
        }

    def _calculate_mean_reversion_signals(self, prices_df):
        """Mean reversion strategy using statistical measures and Bollinger Bands"""
        # Calculate z-score of price relative to moving average
        ma_50 = prices_df["close"].rolling(window=50).mean()
        std_50 = prices_df["close"].rolling(window=50).std()
        z_score = (prices_df["close"] - ma_50) / std_50

        # Calculate Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices_df)

        # Calculate RSI with multiple timeframes
        rsi_14 = self._calculate_rsi(prices_df, 14)
        rsi_28 = self._calculate_rsi(prices_df, 28)

        # Mean reversion signals
        price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        # Combine signals
        if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
            signal = "bullish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
        elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
            signal = "bearish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "z_score": float(z_score.iloc[-1]),
                "price_vs_bb": float(price_vs_bb),
                "rsi_14": float(rsi_14.iloc[-1]),
                "rsi_28": float(rsi_28.iloc[-1]),
            },
        }

    def _calculate_momentum_signals(self, prices_df):
        """Multi-factor momentum strategy"""
        # Price momentum
        returns = prices_df["close"].pct_change()
        mom_1m = returns.rolling(21).sum()
        mom_3m = returns.rolling(63).sum()
        mom_6m = returns.rolling(126).sum()

        # Volume momentum
        volume_ma = prices_df["volume"].rolling(21).mean()
        volume_momentum = prices_df["volume"] / volume_ma

        # Calculate momentum score
        momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

        # Volume confirmation
        volume_confirmation = volume_momentum.iloc[-1] > 1.0

        if momentum_score > 0.05 and volume_confirmation:
            signal = "bullish"
            confidence = min(abs(momentum_score) * 5, 1.0)
        elif momentum_score < -0.05 and volume_confirmation:
            signal = "bearish"
            confidence = min(abs(momentum_score) * 5, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "momentum_1m": float(mom_1m.iloc[-1]),
                "momentum_3m": float(mom_3m.iloc[-1]),
                "momentum_6m": float(mom_6m.iloc[-1]),
                "volume_momentum": float(volume_momentum.iloc[-1]),
            },
        }

    def _calculate_volatility_signals(self, prices_df):
        """Volatility-based trading strategy"""
        # Calculate various volatility metrics
        returns = prices_df["close"].pct_change()

        # Historical volatility
        hist_vol = returns.rolling(21).std() * math.sqrt(252)

        # Volatility regime detection
        vol_ma = hist_vol.rolling(63).mean()
        vol_regime = hist_vol / vol_ma

        # Volatility mean reversion
        vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

        # ATR ratio
        atr = self._calculate_atr(prices_df)
        atr_ratio = atr / prices_df["close"]

        # Generate signal based on volatility regime
        current_vol_regime = vol_regime.iloc[-1]
        vol_z = vol_z_score.iloc[-1]

        if current_vol_regime < 0.8 and vol_z < -1:
            signal = "bullish"  # Low vol regime, potential for expansion
            confidence = min(abs(vol_z) / 3, 1.0)
        elif current_vol_regime > 1.2 and vol_z > 1:
            signal = "bearish"  # High vol regime, potential for contraction
            confidence = min(abs(vol_z) / 3, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "historical_volatility": float(hist_vol.iloc[-1]),
                "volatility_regime": float(current_vol_regime),
                "volatility_z_score": float(vol_z),
                "atr_ratio": float(atr_ratio.iloc[-1]),
            },
        }

    def _calculate_stat_arb_signals(self, prices_df):
        """Statistical arbitrage signals based on price action analysis"""
        # Calculate price distribution statistics
        returns = prices_df["close"].pct_change()

        # Skewness and kurtosis
        skew = returns.rolling(63).skew()
        kurt = returns.rolling(63).kurt()

        # Test for mean reversion using Hurst exponent
        hurst = self._calculate_hurst_exponent(prices_df["close"])

        # Generate signal based on statistical properties
        if hurst < 0.4 and skew.iloc[-1] > 1:
            signal = "bullish"
            confidence = (0.5 - hurst) * 2
        elif hurst < 0.4 and skew.iloc[-1] < -1:
            signal = "bearish"
            confidence = (0.5 - hurst) * 2
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "hurst_exponent": float(hurst),
                "skewness": float(skew.iloc[-1]),
                "kurtosis": float(kurt.iloc[-1]),
            },
        }

    def _weighted_signal_combination(self, signals):
        """Combines multiple trading signals using a weighted approach"""
        # Convert signals to numeric values
        signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

        weighted_sum = 0
        total_confidence = 0

        for strategy, signal in signals.items():
            numeric_signal = signal_values[signal["signal"]]
            weight = self.strategy_weights[strategy]
            confidence = signal["confidence"]

            weighted_sum += numeric_signal * weight * confidence
            total_confidence += weight * confidence

        # Normalize the weighted sum
        if total_confidence > 0:
            final_score = weighted_sum / total_confidence
        else:
            final_score = 0

        # Convert back to signal
        if final_score > 0.2:
            signal = "bullish"
        elif final_score < -0.2:
            signal = "bearish"
        else:
            signal = "neutral"

        return {"signal": signal, "confidence": abs(final_score)}

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = prices_df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
        sma = prices_df["close"].rolling(window).mean()
        std_dev = prices_df["close"].rolling(window).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        return upper_band, lower_band

    def _calculate_ema(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df["close"].ewm(span=window, adjust=False).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        # Calculate True Range
        df = df.copy()
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = abs(df["high"] - df["close"].shift())
        df["low_close"] = abs(df["low"] - df["close"].shift())
        df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

        # Calculate Directional Movement
        df["up_move"] = df["high"] - df["high"].shift()
        df["down_move"] = df["low"].shift() - df["low"]

        df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
        df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

        # Calculate ADX
        df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
        df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
        df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
        df["adx"] = df["dx"].ewm(span=period).mean()

        return df[["adx", "+di", "-di"]]

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(period).mean()

    def _calculate_hurst_exponent(self, price_series: pd.Series, max_lag: int = 20) -> float:
        lags = range(2, max_lag)
        # Add small epsilon to avoid log(0)
        tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

        # Return the Hurst exponent from linear fit
        try:
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]  # Hurst exponent is the slope
        except:
            # Return 0.5 (random walk) if calculation fails
            return 0.5