"""AI-generated stock summaries in plain English.

Transforms raw signal/prediction data into readable, Perplexity-style
explanations without needing an external LLM API.
"""

import logging

logger = logging.getLogger(__name__)


class AISummaryService:
    """Generate human-readable AI summaries from signal and fundamental data."""

    def generate_signal_summary(self, signal_data: dict, fundamentals: dict = None) -> dict:
        """Generate a plain-English summary explaining why a signal is bullish/bearish/neutral."""
        direction = signal_data.get("direction", "NEUTRAL")
        confidence = signal_data.get("confidence", 0)
        composite = signal_data.get("composite_score", 0)

        tech = signal_data.get("technical", {})
        sent = signal_data.get("sentiment", {})
        glob = signal_data.get("global_market", {})
        fund = signal_data.get("fundamental", {})
        oi = signal_data.get("oi_analysis", {})
        sr = signal_data.get("support_resistance", {})
        mtf = signal_data.get("mtf_confluence", {})
        learning = signal_data.get("stock_learning", {})

        # Build factors list
        factors = []
        risk_factors = []

        # Technical analysis
        tech_score = tech.get("score", 0)
        details = tech.get("details", {})
        rsi = details.get("rsi")
        vol_ratio = details.get("volume_ratio")
        ma_cross = details.get("ma_cross")
        bb_pos = details.get("bb_position")

        if abs(tech_score) > 20:
            impact = "positive" if tech_score > 0 else "negative"
            parts = []
            if rsi is not None:
                if rsi < 30:
                    parts.append(f"RSI is oversold at {rsi:.0f}")
                elif rsi > 70:
                    parts.append(f"RSI is overbought at {rsi:.0f}")
                else:
                    parts.append(f"RSI at {rsi:.0f}")
            if ma_cross:
                parts.append(f"moving average crossover is {ma_cross}")
            if vol_ratio and vol_ratio > 1.5:
                parts.append(f"volume is {vol_ratio:.1f}x above average")
            if bb_pos is not None:
                if bb_pos < 0.2:
                    parts.append("price near lower Bollinger Band (potential support)")
                elif bb_pos > 0.8:
                    parts.append("price near upper Bollinger Band (potential resistance)")

            detail = ". ".join(parts) if parts else f"Technical score: {tech_score:+.0f}"
            factors.append({"factor": "Technical Analysis", "impact": impact, "detail": detail})

            if rsi and rsi < 25:
                risk_factors.append("RSI deeply oversold — potential for further downside before reversal")
            if rsi and rsi > 80:
                risk_factors.append("RSI extremely overbought — pullback risk is elevated")
        elif abs(tech_score) > 5:
            factors.append({"factor": "Technical Analysis", "impact": "neutral",
                          "detail": "Technical indicators are mixed with no strong directional bias"})

        # Sentiment
        sent_score = sent.get("score", 0)
        headline_count = sent.get("headline_count", 0)
        pos_count = sent.get("positive_count", 0)
        neg_count = sent.get("negative_count", 0)

        if headline_count > 0:
            if abs(sent_score) > 15:
                impact = "positive" if sent_score > 0 else "negative"
                sentiment_word = "positive" if sent_score > 0 else "negative"
                detail = f"News sentiment is {sentiment_word} ({pos_count} positive, {neg_count} negative out of {headline_count} headlines)"
                factors.append({"factor": "News Sentiment", "impact": impact, "detail": detail})
            else:
                factors.append({"factor": "News Sentiment", "impact": "neutral",
                              "detail": f"News is mixed ({headline_count} headlines analyzed, no strong bias)"})

        # Global markets
        glob_score = glob.get("score", 0)
        news_mag = glob.get("news_magnitude", 0)

        if abs(glob_score) > 10 or news_mag > 30:
            impact = "positive" if glob_score > 0 else "negative"
            parts = []
            if news_mag >= 60:
                parts.append(f"high-impact global event detected (magnitude: {news_mag})")
            if glob_score > 15:
                parts.append("global markets are supportive")
            elif glob_score < -15:
                parts.append("global markets are showing weakness")
            detail = ". ".join(parts) if parts else f"Global market score: {glob_score:+.0f}"
            factors.append({"factor": "Global Markets", "impact": impact, "detail": detail})

            if news_mag >= 60:
                risk_factors.append(f"High-impact global event (magnitude {news_mag}) — increased volatility expected")

        # Fundamentals
        fund_score = fund.get("score", 0)
        fund_class = fund.get("classification", "balanced")

        if fundamentals and fundamentals.get("pe"):
            pe = fundamentals.get("pe", 0)
            roe = fundamentals.get("roe", 0)
            de = fundamentals.get("de", 0)
            rev_growth = fundamentals.get("rev_growth", 0)

            parts = []
            if pe and pe > 0:
                if pe < 15:
                    parts.append(f"attractively valued at {pe:.1f}x P/E")
                elif pe > 40:
                    parts.append(f"richly valued at {pe:.1f}x P/E")
                else:
                    parts.append(f"P/E at {pe:.1f}x")
            if roe and roe > 15:
                parts.append(f"strong ROE of {roe:.1f}%")
            if rev_growth and abs(rev_growth) > 5:
                direction_word = "growing" if rev_growth > 0 else "declining"
                parts.append(f"revenue {direction_word} at {rev_growth:.1f}%")
            if de and de > 1.5:
                risk_factors.append(f"High debt-to-equity ratio ({de:.1f}x)")

            if parts:
                impact = "positive" if fund_score > 5 else ("negative" if fund_score < -5 else "neutral")
                factors.append({"factor": "Fundamentals", "impact": impact,
                              "detail": ". ".join(parts) + f". Classified as '{fund_class}'"})

        # OI Analysis
        if oi.get("available"):
            pcr = oi.get("pcr")
            max_pain = oi.get("max_pain")
            oi_score = oi.get("score", 0)

            if abs(oi_score) > 10:
                impact = "positive" if oi_score > 0 else "negative"
                parts = []
                if pcr is not None:
                    if pcr > 1.2:
                        parts.append(f"PCR at {pcr:.2f} suggests bullish sentiment (more puts being written)")
                    elif pcr < 0.7:
                        parts.append(f"PCR at {pcr:.2f} suggests bearish sentiment (more calls being written)")
                if max_pain:
                    parts.append(f"max pain level at ₹{max_pain:,.0f}")
                detail = ". ".join(parts) if parts else f"OI score: {oi_score:+.0f}"
                factors.append({"factor": "Options Activity", "impact": impact, "detail": detail})

        # MTF Confluence
        mtf_level = mtf.get("level", "LOW")
        if mtf_level == "HIGH":
            timeframes = mtf.get("timeframes", [])
            tf_dirs = [f"{tf['label']}: {tf['direction']}" for tf in timeframes if tf.get("direction")]
            factors.append({"factor": "Multi-Timeframe", "impact": "positive" if composite > 0 else "negative",
                          "detail": f"Strong confluence — all timeframes agree ({', '.join(tf_dirs)})"})
        elif mtf_level == "MEDIUM":
            factors.append({"factor": "Multi-Timeframe", "impact": "neutral",
                          "detail": "Partial agreement across timeframes"})

        # Support/Resistance
        levels = sr.get("levels", {})
        if levels:
            sr_parts = []
            if levels.get("pivot"):
                sr_parts.append(f"Pivot: ₹{levels['pivot']:,.0f}")
            if levels.get("s1"):
                sr_parts.append(f"S1: ₹{levels['s1']:,.0f}")
            if levels.get("r1"):
                sr_parts.append(f"R1: ₹{levels['r1']:,.0f}")
            if sr_parts:
                factors.append({"factor": "Support/Resistance", "impact": "neutral",
                              "detail": " | ".join(sr_parts)})

        # Generate summary text
        summary = self._build_summary(direction, confidence, factors, risk_factors, learning)

        return {
            "summary": summary,
            "factors": factors,
            "risk_factors": risk_factors,
            "direction": direction,
            "confidence": confidence,
        }

    def _build_summary(self, direction: str, confidence: float, factors: list,
                       risk_factors: list, learning: dict) -> str:
        """Build a cohesive plain-English summary paragraph."""
        if direction == "BULLISH":
            opener = f"The AI signal is **bullish** with {confidence:.0f}% confidence."
        elif direction == "BEARISH":
            opener = f"The AI signal is **bearish** with {confidence:.0f}% confidence."
        else:
            opener = f"The AI signal is **neutral** ({confidence:.0f}% confidence) — no strong directional bias."

        # Key drivers
        positive = [f for f in factors if f["impact"] == "positive"]
        negative = [f for f in factors if f["impact"] == "negative"]

        parts = [opener]

        if positive:
            drivers = [f["factor"].lower() for f in positive[:3]]
            parts.append(f"Bullish drivers include {self._join_list(drivers)}.")
        if negative:
            drags = [f["factor"].lower() for f in negative[:3]]
            parts.append(f"Bearish pressure from {self._join_list(drags)}.")

        if risk_factors:
            parts.append(f"Key risk: {risk_factors[0]}")

        # Learning insight
        if learning and learning.get("available"):
            acc = learning.get("overall_accuracy", 0)
            best_tf = learning.get("best_timeframe", "")
            tf_map = {"15min": "15-minute", "30min": "30-minute", "1hr": "1-hour"}
            tf_label = tf_map.get(best_tf, best_tf)
            trend = learning.get("trend", "stable")
            if acc > 0:
                parts.append(
                    f"Historical accuracy for this stock: {acc:.0f}% "
                    f"(best at {tf_label} window, trend: {trend})."
                )

        return " ".join(parts)

    @staticmethod
    def _join_list(items: list) -> str:
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    def generate_earnings_analysis(self, fundamentals: dict) -> dict:
        """Analyze earnings data and generate insights."""
        if not fundamentals:
            return {"available": False, "summary": "No earnings data available."}

        earnings_q = fundamentals.get("earnings_quarterly", [])
        income_q = fundamentals.get("income_quarterly", [])

        if not earnings_q and not income_q:
            return {"available": False, "summary": "No quarterly earnings data available."}

        insights = []
        metrics = {}

        # EPS Analysis
        if earnings_q:
            latest = earnings_q[0] if earnings_q else {}
            surprise = latest.get("surprise_pct")
            eps = latest.get("reported_eps") or latest.get("actual")

            if surprise is not None:
                if surprise > 5:
                    insights.append(f"Beat EPS estimates by {surprise:.1f}% — strong execution")
                elif surprise < -5:
                    insights.append(f"Missed EPS estimates by {abs(surprise):.1f}% — disappointing quarter")
                else:
                    insights.append(f"Met EPS expectations (surprise: {surprise:+.1f}%)")
                metrics["eps_surprise"] = round(surprise, 1)

            if eps:
                metrics["latest_eps"] = round(float(eps), 2)

            # EPS trend (if multiple quarters)
            if len(earnings_q) >= 2:
                eps_vals = []
                for q in earnings_q[:4]:
                    e = q.get("reported_eps") or q.get("actual")
                    if e:
                        eps_vals.append(float(e))
                if len(eps_vals) >= 2:
                    if eps_vals[0] > eps_vals[1]:
                        pct = (eps_vals[0] - eps_vals[1]) / abs(eps_vals[1]) * 100 if eps_vals[1] != 0 else 0
                        insights.append(f"EPS grew {pct:.0f}% quarter-over-quarter")
                        metrics["eps_qoq_growth"] = round(pct, 1)
                    elif eps_vals[0] < eps_vals[1]:
                        pct = (eps_vals[1] - eps_vals[0]) / abs(eps_vals[1]) * 100 if eps_vals[1] != 0 else 0
                        insights.append(f"EPS declined {pct:.0f}% quarter-over-quarter")
                        metrics["eps_qoq_growth"] = round(-pct, 1)

        # Revenue Analysis
        if income_q and len(income_q) >= 2:
            rev0 = income_q[0].get("revenue") or income_q[0].get("totalRevenue")
            rev1 = income_q[1].get("revenue") or income_q[1].get("totalRevenue")

            if rev0 and rev1 and rev1 > 0:
                rev_growth = (float(rev0) - float(rev1)) / float(rev1) * 100
                metrics["revenue_qoq_growth"] = round(rev_growth, 1)
                if rev_growth > 10:
                    insights.append(f"Revenue grew {rev_growth:.0f}% QoQ — strong top-line growth")
                elif rev_growth > 0:
                    insights.append(f"Revenue grew {rev_growth:.0f}% QoQ")
                else:
                    insights.append(f"Revenue declined {abs(rev_growth):.0f}% QoQ")

                metrics["latest_revenue"] = float(rev0)

        # Margin Analysis
        profit_margin = fundamentals.get("profit_margin", 0)
        operating_margin = fundamentals.get("operating_margin", 0)

        if profit_margin:
            metrics["profit_margin"] = profit_margin
            if profit_margin > 20:
                insights.append(f"Healthy profit margin of {profit_margin:.1f}%")
            elif profit_margin < 5:
                insights.append(f"Thin profit margin of {profit_margin:.1f}% — limited pricing power")

        if operating_margin:
            metrics["operating_margin"] = operating_margin

        # Valuation context
        pe = fundamentals.get("pe", 0)
        forward_pe = fundamentals.get("forward_pe", 0)

        if pe and forward_pe and pe > 0 and forward_pe > 0:
            if forward_pe < pe * 0.8:
                insights.append(f"Forward P/E ({forward_pe:.1f}x) significantly lower than trailing ({pe:.1f}x) — market expects earnings growth")
            elif forward_pe > pe * 1.2:
                insights.append(f"Forward P/E ({forward_pe:.1f}x) higher than trailing ({pe:.1f}x) — market expects earnings decline")

        # Build summary
        if insights:
            summary = " | ".join(insights[:5])
        else:
            summary = "Limited earnings data available for analysis."

        # Rating
        positive_signals = sum(1 for i in insights if any(w in i.lower() for w in ["beat", "grew", "strong", "healthy", "growth"]))
        negative_signals = sum(1 for i in insights if any(w in i.lower() for w in ["missed", "declined", "thin", "disappointing"]))

        if positive_signals > negative_signals:
            rating = "positive"
        elif negative_signals > positive_signals:
            rating = "negative"
        else:
            rating = "neutral"

        return {
            "available": True,
            "summary": summary,
            "insights": insights,
            "metrics": metrics,
            "rating": rating,
        }


ai_summary_service = AISummaryService()
