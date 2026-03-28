"""XGBoost V2 — direction classifier instead of return predictor.

Key differences from v1:
1. Predicts UP/DOWN (classification) instead of exact return (regression)
2. Fewer features — only top 20 proven indicators, no candlestick/fundamental noise
3. Lower regularization — lets the model learn
4. Minimum 200 samples for training
5. Saves alongside v1 for comparison
"""

import os
import numpy as np
import pandas as pd
import ta
import joblib
from app.config import MODEL_DIR, MODEL_FRESHNESS_HOURS

import logging
logger = logging.getLogger(__name__)


class XGBoostV2Predictor:
    def __init__(self):
        pass

    def _model_path(self, symbol: str) -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"xgb_v2_{safe}.joblib")

    def _model_is_fresh(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        import time
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        return age_hours < MODEL_FRESHNESS_HOURS

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lean feature set — only 20 proven indicators."""
        feat = df.copy()
        close = feat["close"]
        high = feat["high"]
        low = feat["low"]
        volume = feat["volume"].astype(float)

        if (volume == 0).all():
            volume = pd.Series(1.0, index=volume.index)
            feat["volume"] = volume

        # Core momentum (4 features)
        feat["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        feat["stoch_k"] = stoch.stoch()
        feat["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        feat["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # Trend (4 features)
        feat["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
        macd = ta.trend.MACD(close)
        feat["macd_hist"] = macd.macd_diff()
        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        feat["sma20_ratio"] = close / sma20.replace(0, np.nan)
        feat["sma50_ratio"] = close / sma50.replace(0, np.nan)

        # Volatility (3 features)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        feat["bb_percent"] = (close - bb_lower) / bb_range
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        feat["atr_norm"] = atr / close
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std().replace(0, np.nan)
        feat["vol_regime"] = vol_20 / vol_60

        # Volume (2 features)
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        feat["volume_ratio"] = volume / vol_ma
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_sma = obv.rolling(20).mean().replace(0, np.nan)
        feat["obv_ratio"] = obv / obv_sma

        # Returns & momentum (5 features)
        for w in [1, 3, 5, 10]:
            feat[f"ret_{w}"] = close.pct_change(w)
        feat["ret_mean_5"] = returns.rolling(5).mean()

        # Price position (2 features)
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        feat["price_position"] = (close - rolling_low) / price_range
        feat["overnight_gap"] = ((feat["open"] / close.shift(1)) - 1).fillna(0)

        feat = feat.dropna().reset_index(drop=True)

        # Target: direction (1 = up, 0 = down) with 0.15% threshold
        next_return = close.pct_change().shift(-1)
        feat["target"] = (next_return > 0.0015).astype(int)
        feat = feat.dropna(subset=["target"]).reset_index(drop=True)

        return feat

    def _get_feature_cols(self, feat_df: pd.DataFrame) -> list:
        exclude = {"date", "datetime", "target", "close", "open", "high", "low", "volume"}
        return [c for c in feat_df.columns if c not in exclude]

    def predict(self, df: pd.DataFrame, symbol: str, horizon: str = "1d") -> dict:
        """Predict direction and return probability + predicted price."""
        from xgboost import XGBClassifier

        feat_df = self._build_features(df)
        feature_cols = self._get_feature_cols(feat_df)

        if len(feat_df) < 200:
            return {"direction": "NEUTRAL", "probability": 0.5, "predicted_price": None,
                    "confidence_score": 0, "version": "v2", "error": "Insufficient data"}

        X = feat_df[feature_cols].values
        y = feat_df["target"].values
        last_close = float(df["close"].iloc[-1])

        model_path = self._model_path(symbol)

        if self._model_is_fresh(model_path):
            try:
                saved = joblib.load(model_path)
                model = saved["model"]
                feature_cols = saved.get("feature_cols", feature_cols)
                X = feat_df[feature_cols].values
            except Exception:
                model = None
        else:
            model = None

        if model is None:
            # Train new model
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,       # Much lower than v1's 0.1
                reg_lambda=0.5,       # Much lower than v1's 1.0
                min_child_weight=2,
                gamma=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=20,
            )

            val_split = int(len(X_train) * 0.85)
            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            y_tr, y_val = y_train[:val_split], y_train[val_split:]

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # Feature importance pruning — keep features with >0.5% importance
            importances = model.feature_importances_
            total_imp = importances.sum()
            if total_imp > 0:
                imp_fracs = importances / total_imp
                keep_mask = imp_fracs >= 0.005  # 0.5% threshold
                if keep_mask.sum() >= 8:  # Keep at least 8 features
                    pruned_cols = [c for c, k in zip(feature_cols, keep_mask) if k]

                    # Retrain on pruned features
                    X_train_p = X_train[:, keep_mask]
                    X_tr_p, X_val_p = X_train_p[:val_split], X_train_p[val_split:]

                    model_p = XGBClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.01, reg_lambda=0.5,
                        min_child_weight=2, gamma=0.05,
                        objective="binary:logistic", eval_metric="logloss",
                        random_state=42, verbosity=0, early_stopping_rounds=20,
                    )
                    model_p.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)

                    model = model_p
                    feature_cols = pruned_cols
                    X = feat_df[feature_cols].values

            # Save
            try:
                joblib.dump({"model": model, "feature_cols": feature_cols, "version": "v2"}, model_path)
            except Exception:
                pass

        # Predict
        last_features = X[-1:].copy()
        prob = model.predict_proba(last_features)[0]

        # prob[0] = P(down), prob[1] = P(up)
        prob_up = float(prob[1]) if len(prob) > 1 else 0.5

        if prob_up > 0.6:
            direction = "BULLISH"
        elif prob_up < 0.4:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Estimate predicted price from probability
        avg_move = feat_df["ret_1"].abs().mean()  # average daily move
        if direction == "BULLISH":
            predicted_price = round(last_close * (1 + avg_move * (prob_up - 0.5) * 4), 2)
        elif direction == "BEARISH":
            predicted_price = round(last_close * (1 - avg_move * (0.5 - prob_up) * 4), 2)
        else:
            predicted_price = round(last_close, 2)

        confidence = round(abs(prob_up - 0.5) * 200, 1)  # 0-100 scale

        # Get actual feature importance
        top_features = []
        try:
            imps = model.feature_importances_
            total = sum(imps)
            if total > 0:
                pairs = sorted(zip(feature_cols, imps), key=lambda x: -x[1])
                top_features = [{"name": n, "importance": round(float(v / total * 100), 1)} for n, v in pairs[:5]]
        except Exception:
            pass

        return {
            "direction": direction,
            "probability": round(prob_up, 4),
            "predicted_price": predicted_price,
            "confidence_score": confidence,
            "version": "v2",
            "features_used": len(feature_cols),
            "top_features": top_features,
            "best_iteration": getattr(model, "best_iteration", None),
        }


xgboost_v2 = XGBoostV2Predictor()
