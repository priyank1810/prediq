import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class PredictionExplainer:
    """Analyzes model outputs + market context to produce human-readable explanations."""

    def explain(
        self,
        symbol: str,
        prediction_result: dict,
        df: pd.DataFrame,
        sentiment: Optional[dict] = None,
        global_data: Optional[dict] = None,
    ) -> dict:
        """Generate explanation for why predictions are bullish/bearish.

        Args:
            symbol: Stock symbol
            prediction_result: Full prediction result dict with lstm/prophet/xgboost/ensemble keys
            df: Historical price DataFrame (OHLCV)
            sentiment: Sentiment service result (optional)
            global_data: Global market service result (optional)

        Returns:
            Explanation dict with direction, confidence, summary, key_drivers, risk_factors, support_resistance
        """
        drivers = []
        risk_factors = []
        bullish_score = 0
        total_weight = 0

        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = float(close.iloc[-1])

        # 1. Price direction from ensemble
        ensemble = prediction_result.get("ensemble", {})
        ensemble_preds = ensemble.get("predictions", [])
        if ensemble_preds:
            predicted_price = ensemble_preds[-1]
            price_change_pct = ((predicted_price - current_price) / current_price) * 100

            if price_change_pct > 0.3:
                bullish_score += 25
                impact = "positive"
            elif price_change_pct < -0.3:
                bullish_score -= 25
                impact = "negative"
            else:
                impact = "neutral"
            total_weight += 25

            drivers.append({
                "factor": "Price Forecast",
                "impact": impact,
                "detail": f"Ensemble predicts {'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}% move to ₹{predicted_price:,.2f}",
            })

        # 2. Technical signals
        tech_detail_parts = []
        tech_score = 0

        # RSI
        try:
            rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
            rsi = float(rsi_series.iloc[-1])
            if rsi < 30:
                tech_score += 2
                tech_detail_parts.append(f"RSI oversold at {rsi:.0f}")
                risk_factors.append("RSI in oversold territory — watch for further downside before reversal")
            elif rsi > 70:
                tech_score -= 2
                tech_detail_parts.append(f"RSI overbought at {rsi:.0f}")
                risk_factors.append("RSI approaching overbought territory (>70)")
            elif rsi > 50:
                tech_score += 1
                tech_detail_parts.append(f"RSI at {rsi:.0f}")
            else:
                tech_score -= 1
                tech_detail_parts.append(f"RSI at {rsi:.0f}")
        except Exception:
            rsi = 50

        # MACD crossover
        try:
            macd_ind = ta.trend.MACD(close)
            macd_line = float(macd_ind.macd().iloc[-1])
            macd_signal = float(macd_ind.macd_signal().iloc[-1])
            if macd_line > macd_signal:
                tech_score += 2
                tech_detail_parts.append("MACD bullish crossover")
            else:
                tech_score -= 2
                tech_detail_parts.append("MACD bearish crossover")
        except Exception:
            pass

        # SMA trend
        try:
            sma20 = float(ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1])
            if current_price > sma20:
                tech_score += 1
                tech_detail_parts.append("price above 20-day SMA")
            else:
                tech_score -= 1
                tech_detail_parts.append("price below 20-day SMA")
        except Exception:
            pass

        # Bollinger Band position
        try:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_lower = float(bb.bollinger_lband().iloc[-1])
            bb_upper = float(bb.bollinger_hband().iloc[-1])
            if current_price < bb_lower:
                tech_score += 1
                tech_detail_parts.append("price below lower Bollinger Band")
            elif current_price > bb_upper:
                tech_score -= 1
                tech_detail_parts.append("price above upper Bollinger Band")
        except Exception:
            pass

        tech_impact = "positive" if tech_score > 0 else ("negative" if tech_score < 0 else "neutral")
        bullish_score += tech_score * 5
        total_weight += 30

        drivers.append({
            "factor": "Technical Momentum",
            "impact": tech_impact,
            "detail": ", ".join(tech_detail_parts) if tech_detail_parts else "Mixed signals",
        })

        # 3. Sentiment
        if sentiment:
            sent_score = sentiment.get("score", 0)
            pos_count = sentiment.get("positive_count", 0)
            neg_count = sentiment.get("negative_count", 0)

            if sent_score > 15:
                sent_impact = "positive"
                bullish_score += 15
            elif sent_score < -15:
                sent_impact = "negative"
                bullish_score -= 15
            else:
                sent_impact = "neutral"
            total_weight += 20

            drivers.append({
                "factor": "News Sentiment",
                "impact": sent_impact,
                "detail": f"{pos_count} positive vs {neg_count} negative headlines, composite score {sent_score:.0f}",
            })
        else:
            drivers.append({
                "factor": "News Sentiment",
                "impact": "neutral",
                "detail": "No sentiment data available",
            })

        # 4. Global context
        if global_data and global_data.get("markets"):
            global_score = global_data.get("score", 0)
            markets = global_data["markets"]

            market_parts = []
            for m in markets:
                if m["name"] in ("S&P 500", "India VIX"):
                    direction_str = f"{'+' if m['change_pct'] >= 0 else ''}{m['change_pct']:.1f}%"
                    market_parts.append(f"{m['name']} {direction_str}")

            if global_score > 10:
                global_impact = "positive"
                bullish_score += 10
            elif global_score < -10:
                global_impact = "negative"
                bullish_score -= 10
            else:
                global_impact = "neutral"
            total_weight += 15

            # VIX-specific risk
            vix_market = next((m for m in markets if m["name"] == "India VIX"), None)
            if vix_market and vix_market["change_pct"] > 3:
                risk_factors.append("VIX trending upward — increased volatility expected")

            drivers.append({
                "factor": "Global Markets",
                "impact": global_impact,
                "detail": ", ".join(market_parts) if market_parts else f"Global score: {global_score:.0f}",
            })

        # 5. Model agreement
        model_directions = {}
        for model_name in ("lstm", "prophet", "xgboost"):
            model_data = prediction_result.get(model_name)
            if model_data and model_data.get("predictions"):
                pred = model_data["predictions"][-1]
                change_pct = ((pred - current_price) / current_price) * 100
                model_directions[model_name.upper()] = change_pct

        if model_directions:
            all_bullish = all(v > 0 for v in model_directions.values())
            all_bearish = all(v < 0 for v in model_directions.values())

            parts = [f"{name} {'+' if pct >= 0 else ''}{pct:.1f}%" for name, pct in model_directions.items()]

            if all_bullish:
                agreement_impact = "positive"
                bullish_score += 10
                agreement_detail = f"All {len(model_directions)} models predict upside ({', '.join(parts)})"
            elif all_bearish:
                agreement_impact = "negative"
                bullish_score -= 10
                agreement_detail = f"All {len(model_directions)} models predict downside ({', '.join(parts)})"
            else:
                agreement_impact = "neutral"
                agreement_detail = f"Models disagree ({', '.join(parts)})"
                risk_factors.append("Models disagree on direction — lower conviction")
            total_weight += 10

            drivers.append({
                "factor": "Model Consensus",
                "impact": agreement_impact,
                "detail": agreement_detail,
            })

        # 6. Support/Resistance from rolling window
        try:
            window = min(20, len(df))
            recent_high = float(high.iloc[-window:].max())
            recent_low = float(low.iloc[-window:].min())
        except Exception:
            recent_high = current_price * 1.02
            recent_low = current_price * 0.98

        # Determine overall direction and confidence
        if total_weight > 0:
            normalized = bullish_score / total_weight * 100
        else:
            normalized = 0

        if normalized > 10:
            direction = "BULLISH"
        elif normalized < -10:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = min(95, max(5, int(abs(normalized) + 30)))

        # Build summary
        positive_drivers = [d["factor"] for d in drivers if d["impact"] == "positive"]
        negative_drivers = [d["factor"] for d in drivers if d["impact"] == "negative"]

        if direction == "BULLISH":
            if positive_drivers:
                summary = f"Bullish outlook driven by {', '.join(positive_drivers).lower()}"
            else:
                summary = "Slightly bullish with mixed signals"
        elif direction == "BEARISH":
            if negative_drivers:
                summary = f"Bearish outlook due to {', '.join(negative_drivers).lower()}"
            else:
                summary = "Slightly bearish with mixed signals"
        else:
            summary = "Neutral outlook — no strong directional signal from available data"

        return {
            "direction": direction,
            "confidence": confidence,
            "summary": summary,
            "key_drivers": drivers,
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors identified"],
            "support_resistance": {
                "support": round(recent_low, 2),
                "resistance": round(recent_high, 2),
            },
        }


prediction_explainer = PredictionExplainer()
