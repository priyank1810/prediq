import logging
import numpy as np

logger = logging.getLogger(__name__)


class FundamentalModel:
    """Scores a stock's fundamentals against sector medians.
    Output: fundamental_score (-1 to +1) + value/growth classification.
    Not a price predictor — provides directional bias to meta-learner."""

    # Sector median approximations for Indian large-caps
    _sector_medians = {
        "default": {"pe": 25, "pb": 3.5, "roe": 15, "de": 0.5, "rev_growth": 10, "earn_growth": 12},
        "IT": {"pe": 28, "pb": 8, "roe": 25, "de": 0.1, "rev_growth": 12, "earn_growth": 14},
        "Banking": {"pe": 15, "pb": 2.5, "roe": 14, "de": 8, "rev_growth": 15, "earn_growth": 18},
        "Pharma": {"pe": 30, "pb": 4, "roe": 16, "de": 0.3, "rev_growth": 10, "earn_growth": 12},
        "Auto": {"pe": 22, "pb": 4, "roe": 15, "de": 0.6, "rev_growth": 12, "earn_growth": 15},
        "FMCG": {"pe": 55, "pb": 12, "roe": 30, "de": 0.2, "rev_growth": 8, "earn_growth": 10},
        "Metal": {"pe": 10, "pb": 1.5, "roe": 12, "de": 0.8, "rev_growth": 8, "earn_growth": 10},
        "Energy": {"pe": 12, "pb": 1.8, "roe": 14, "de": 0.7, "rev_growth": 8, "earn_growth": 10},
    }

    def _get_sector(self, symbol: str) -> str:
        from app.config import SECTOR_MAP
        for sector, symbols in SECTOR_MAP.items():
            if symbol in symbols:
                return sector
        return "default"

    def score(self, fundamentals: dict, symbol: str = "") -> dict:
        """Score fundamentals against sector medians.

        Args:
            fundamentals: Dict with keys like pe, pb, roe, de, rev_growth, earn_growth, div_yield
            symbol: Stock symbol for sector lookup

        Returns:
            {score: float (-1 to 1), classification: 'value'|'growth'|'balanced', details: dict}
        """
        if not fundamentals:
            return {"score": 0.0, "classification": "balanced", "details": {}}

        sector = self._get_sector(symbol)
        medians = self._sector_medians.get(sector, self._sector_medians["default"])

        scores = []
        details = {}

        # P/E ratio: lower is better (value)
        pe = fundamentals.get("pe")
        if pe and pe > 0:
            pe_score = np.clip((medians["pe"] - pe) / medians["pe"], -1, 1)
            scores.append(pe_score * 0.2)
            details["pe"] = {"value": pe, "median": medians["pe"], "score": round(pe_score, 2)}

        # P/B ratio: lower is better
        pb = fundamentals.get("pb")
        if pb and pb > 0:
            pb_score = np.clip((medians["pb"] - pb) / medians["pb"], -1, 1)
            scores.append(pb_score * 0.1)
            details["pb"] = {"value": pb, "median": medians["pb"], "score": round(pb_score, 2)}

        # ROE: higher is better
        roe = fundamentals.get("roe")
        if roe is not None:
            roe_score = np.clip((roe - medians["roe"]) / medians["roe"], -1, 1)
            scores.append(roe_score * 0.2)
            details["roe"] = {"value": roe, "median": medians["roe"], "score": round(roe_score, 2)}

        # D/E ratio: lower is better (except banking)
        de = fundamentals.get("de")
        if de is not None and sector != "Banking":
            de_score = np.clip((medians["de"] - de) / max(medians["de"], 0.1), -1, 1)
            scores.append(de_score * 0.1)
            details["de"] = {"value": de, "median": medians["de"], "score": round(de_score, 2)}

        # Revenue growth: higher is better
        rev_growth = fundamentals.get("rev_growth")
        if rev_growth is not None:
            rg_score = np.clip((rev_growth - medians["rev_growth"]) / max(medians["rev_growth"], 1), -1, 1)
            scores.append(rg_score * 0.2)
            details["rev_growth"] = {"value": rev_growth, "median": medians["rev_growth"], "score": round(rg_score, 2)}

        # Earnings growth: higher is better
        earn_growth = fundamentals.get("earn_growth")
        if earn_growth is not None:
            eg_score = np.clip((earn_growth - medians["earn_growth"]) / max(medians["earn_growth"], 1), -1, 1)
            scores.append(eg_score * 0.15)
            details["earn_growth"] = {"value": earn_growth, "median": medians["earn_growth"], "score": round(eg_score, 2)}

        # Dividend yield: positive signal
        div_yield = fundamentals.get("div_yield")
        if div_yield and div_yield > 0:
            dy_score = min(div_yield / 3.0, 1.0)  # 3% yield = max score
            scores.append(dy_score * 0.05)
            details["div_yield"] = {"value": div_yield, "score": round(dy_score, 2)}

        if not scores:
            return {"score": 0.0, "classification": "balanced", "details": details}

        final_score = float(np.clip(sum(scores), -1, 1))

        # Classification
        if pe and pe < medians["pe"] * 0.8 and (div_yield or 0) > 1.5:
            classification = "value"
        elif (rev_growth or 0) > medians["rev_growth"] * 1.3 and (pe or 0) > medians["pe"]:
            classification = "growth"
        else:
            classification = "balanced"

        return {
            "score": round(final_score, 3),
            "classification": classification,
            "details": details,
        }


fundamental_model = FundamentalModel()
