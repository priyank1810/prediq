import numpy as np
import pandas as pd
import ta


class IndicatorService:
    def compute_all(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        dates = df["date"].tolist()

        result = {}

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        result["rsi"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in rsi]}

        # MACD
        macd = ta.trend.MACD(close)
        result["macd_line"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd()]}
        result["macd_signal"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd_signal()]}
        result["macd_histogram"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd_diff()]}

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        result["bollinger_upper"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_hband()]}
        result["bollinger_middle"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_mavg()]}
        result["bollinger_lower"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_lband()]}

        # SMA
        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        result["sma_20"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in sma20]}
        result["sma_50"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in sma50]}

        # EMA
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        result["ema_20"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in ema20]}

        return result

    def compute_intraday_indicators(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty or len(df) < 26:
            return {"score": 0, "details": {}, "raw": {}}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # MACD
        macd_ind = ta.trend.MACD(close)
        macd_diff = macd_ind.macd_diff().iloc[-1]
        macd_diff = 0 if pd.isna(macd_diff) else macd_diff

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        current_price = float(close.iloc[-1])

        # 3-bar momentum (45 minutes)
        momentum_3 = ((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]) * 100 if len(close) >= 4 else 0.0

        # EMA crossover (5 vs 13)
        ema5 = ta.trend.EMAIndicator(close, window=5).ema_indicator().iloc[-1]
        ema13 = ta.trend.EMAIndicator(close, window=13).ema_indicator().iloc[-1]
        ema5 = current_price if pd.isna(ema5) else float(ema5)
        ema13 = current_price if pd.isna(ema13) else float(ema13)

        # VWAP
        typical_price = (high + low + close) / 3
        cumvol = df["volume"].cumsum()
        vwap_series = (typical_price * df["volume"]).cumsum() / cumvol.replace(0, np.nan)
        current_vwap = float(vwap_series.iloc[-1]) if not pd.isna(vwap_series.iloc[-1]) else current_price

        # --- Scoring (each -100 to +100) ---
        rsi_score = max(-100, min(100, -(rsi - 50) * (100 / 30)))

        macd_score = max(-100, min(100, (macd_diff / current_price) * 10000)) if current_price > 0 else 0

        bb_range = bb_upper - bb_lower if (not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower) else 1
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            bb_pos = (current_price - bb_lower) / bb_range
            bb_score = max(-100, min(100, -(bb_pos - 0.5) * 200))
        else:
            bb_score = 0

        momentum_score = max(-100, min(100, momentum_3 * 50))

        ema_score = max(-100, min(100, ((ema5 - ema13) / current_price) * 10000)) if current_price > 0 else 0

        vwap_score = max(-100, min(100, ((current_price - current_vwap) / current_price) * 10000)) if current_price > 0 else 0

        # Weighted composite
        technical_score = (
            0.20 * rsi_score +
            0.25 * macd_score +
            0.15 * bb_score +
            0.20 * momentum_score +
            0.10 * ema_score +
            0.10 * vwap_score
        )
        technical_score = max(-100, min(100, round(technical_score, 2)))

        details = {
            "rsi": round(rsi, 2),
            "rsi_score": round(rsi_score, 2),
            "macd_diff": round(float(macd_diff), 4),
            "macd_score": round(macd_score, 2),
            "bb_score": round(bb_score, 2),
            "momentum_3bar": round(float(momentum_3), 4),
            "momentum_score": round(momentum_score, 2),
            "ema_score": round(ema_score, 2),
            "vwap_score": round(vwap_score, 2),
            "current_price": round(current_price, 2),
            "vwap": round(current_vwap, 2),
        }

        datetimes = df["datetime_str"].tolist() if "datetime_str" in df.columns else list(range(len(df)))
        raw = {
            "datetimes": datetimes,
            "rsi": [None if pd.isna(v) else round(v, 2) for v in rsi_series],
        }

        return {"score": technical_score, "details": details, "raw": raw}


indicator_service = IndicatorService()
