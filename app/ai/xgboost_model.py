import os
import numpy as np
import pandas as pd
import ta
import joblib
from sklearn.preprocessing import MinMaxScaler
from app.config import MODEL_DIR, PREDICTION_HORIZONS, MODEL_FRESHNESS_HOURS


class XGBoostPredictor:
    def __init__(self):
        pass

    def _model_path(self, symbol: str, suffix: str = "") -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"xgb_{safe}{suffix}.joblib")

    def _model_is_fresh(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        import time
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        return age_hours < MODEL_FRESHNESS_HOURS

    def _horizon_to_steps(self, horizon: str) -> int:
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            return cfg.get("bars", 1)
        return cfg.get("days", 1)

    def _walk_forward_validate(self, feat_df: pd.DataFrame, feature_cols: list, n_splits: int = 5) -> float:
        """Anchored walk-forward validation: train on [0:t], predict [t:t+fold], slide forward.
        Returns average MAPE across all folds."""
        from xgboost import XGBRegressor

        X = feat_df[feature_cols].values
        y = feat_df["target"].values
        close_prices = feat_df["close"].values
        n = len(X)
        fold_size = n // (n_splits + 1)
        if fold_size < 20:
            return 5.0

        mapes = []
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_end = min(train_end + fold_size, n)
            if test_end <= train_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            test_close = close_prices[train_end:test_end]

            model = XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0,
            )
            model.fit(X_train, y_train, verbose=False)

            pred_returns = model.predict(X_test)
            pred_prices = test_close * (1 + pred_returns)
            actual_prices = test_close * (1 + y_test)

            nonzero = actual_prices != 0
            if nonzero.any():
                fold_mape = float(np.mean(np.abs(
                    (actual_prices[nonzero] - pred_prices[nonzero]) / actual_prices[nonzero]
                )) * 100)
                mapes.append(fold_mape)

        return float(np.mean(mapes)) if mapes else 5.0

    def _build_tabular_features(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Build tabular feature matrix from OHLCV data (no sequences needed)."""
        from app.ai.preprocessing import StockDataPreprocessor

        feat_df = df.copy()
        close = feat_df["close"]
        high = feat_df["high"]
        low = feat_df["low"]
        volume = feat_df["volume"].astype(float)

        # Synthetic volume for indices
        if (volume == 0).all():
            volume = pd.Series(1.0, index=volume.index)
            feat_df["volume"] = volume

        # Add context features
        feat_df = StockDataPreprocessor._add_context_features(feat_df, symbol)

        # Price-based
        feat_df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        macd_ind = ta.trend.MACD(close)
        feat_df["macd"] = macd_ind.macd()
        feat_df["macd_signal"] = macd_ind.macd_signal()
        feat_df["macd_hist"] = macd_ind.macd_diff()

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        feat_df["bb_percent"] = (close - bb_lower) / bb_range

        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()

        feat_df["sma20_ratio"] = close / sma20.replace(0, np.nan)
        feat_df["sma50_ratio"] = close / sma50.replace(0, np.nan)
        feat_df["ema20_ratio"] = close / ema20.replace(0, np.nan)

        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        feat_df["atr_norm"] = atr / close

        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        feat_df["volume_ratio"] = volume / vol_ma

        feat_df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        feat_df["stoch_k"] = stoch.stoch()
        feat_df["stoch_d"] = stoch.stoch_signal()

        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_sma = obv.rolling(20).mean().replace(0, np.nan)
        feat_df["obv_ratio"] = obv / obv_sma

        feat_df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # Williams %R (14)
        feat_df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        # CCI (20)
        feat_df["cci"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()

        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        feat_df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        feat_df["ichimoku_a"] = ichimoku.ichimoku_a()
        feat_df["ichimoku_b"] = ichimoku.ichimoku_b()

        # MA crossover signals (binary)
        sma5 = ta.trend.SMAIndicator(close, window=5).sma_indicator()
        sma200 = ta.trend.SMAIndicator(close, window=200).sma_indicator()
        feat_df["cross_5_20"] = (sma5 > sma20).astype(float)
        feat_df["cross_20_50"] = (sma20 > sma50).astype(float)
        feat_df["cross_50_200"] = (sma50 > sma200).astype(float)

        # SMA(200) ratio
        feat_df["sma200_ratio"] = close / sma200.replace(0, np.nan)

        # Gap analysis (overnight gap %)
        prev_close = close.shift(1)
        feat_df["overnight_gap"] = ((feat_df["open"] / prev_close) - 1).fillna(0)

        # Returns at various horizons (lagged features)
        for w in [1, 2, 3, 5, 10, 20]:
            feat_df[f"ret_{w}"] = close.pct_change(w)

        # Rolling stats
        returns = close.pct_change()
        for w in [5, 10, 20]:
            feat_df[f"ret_mean_{w}"] = returns.rolling(w).mean()
            feat_df[f"ret_std_{w}"] = returns.rolling(w).std()
            feat_df[f"vol_ratio_{w}"] = volume / volume.rolling(w).mean().replace(0, np.nan)

        # Volatility regime
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std().replace(0, np.nan)
        feat_df["vol_regime"] = vol_20 / vol_60

        # Price position
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        feat_df["price_position"] = (close - rolling_low) / price_range

        # Day of week (cyclical)
        if "date" in feat_df.columns:
            dates = pd.to_datetime(feat_df["date"])
            feat_df["day_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 5)
            feat_df["day_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 5)

        feat_df = feat_df.dropna().reset_index(drop=True)

        # Target: next-day return
        feat_df["target"] = close.pct_change().shift(-1)
        feat_df = feat_df.dropna(subset=["target"]).reset_index(drop=True)

        return feat_df

    def _get_feature_cols(self, feat_df: pd.DataFrame) -> list:
        """Return columns used as features (exclude metadata and target)."""
        exclude = {"date", "datetime", "target", "close", "open", "high", "low", "volume"}
        return [c for c in feat_df.columns if c not in exclude]

    def predict(self, df: pd.DataFrame, symbol: str, horizon: str = "1d") -> dict:
        from xgboost import XGBRegressor

        feat_df = self._build_tabular_features(df, symbol=symbol)
        feature_cols = self._get_feature_cols(feat_df)

        if len(feat_df) < 100:
            raise ValueError(f"Not enough data for XGBoost. Got {len(feat_df)}, need 100+")

        X = feat_df[feature_cols].values
        y = feat_df["target"].values  # next-day return
        last_close = df["close"].iloc[-1]

        # Train/test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_path = self._model_path(symbol)
        if self._model_is_fresh(model_path):
            saved = joblib.load(model_path)
            model = saved["model"]
            feature_cols = saved.get("feature_cols", feature_cols)
            X = feat_df[feature_cols].values
            split = int(len(X) * 0.8)
            X_test = X[split:]
        else:
            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                early_stopping_rounds=20,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            # Feature importance pruning: drop features below 1% importance, keep min 10
            importances = model.feature_importances_
            total_imp = importances.sum()
            if total_imp > 0:
                imp_fracs = importances / total_imp
                keep_mask = imp_fracs >= 0.01
                if keep_mask.sum() < 10:
                    top_indices = np.argsort(imp_fracs)[-10:]
                    keep_mask = np.zeros(len(imp_fracs), dtype=bool)
                    keep_mask[top_indices] = True
                if keep_mask.sum() < len(feature_cols):
                    X_train_pruned = X_train[:, keep_mask]
                    X_test_pruned = X_test[:, keep_mask]
                    model_pruned = XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        verbosity=0,
                        early_stopping_rounds=20,
                    )
                    model_pruned.fit(
                        X_train_pruned, y_train,
                        eval_set=[(X_test_pruned, y_test)],
                        verbose=False,
                    )
                    model = model_pruned
                    pruned_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
                    feature_cols = pruned_cols
                    X = feat_df[feature_cols].values
                    X_test = X[split:]

            joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

        # Walk-forward MAPE (anchored, 5-fold)
        try:
            mape = self._walk_forward_validate(feat_df, feature_cols, n_splits=5)
        except Exception:
            mape = 5.0

        confidence = max(0, min(100, 100 - mape))

        # Multi-step prediction with feature rollover
        steps = self._horizon_to_steps(horizon)
        predictions = []
        current_features = X[-1:].copy()
        current_price = float(last_close)

        # Build column name -> index mapping for feature updates
        col_idx = {c: i for i, c in enumerate(feature_cols)}

        for step in range(steps):
            pred_return = model.predict(current_features)[0]
            next_price = current_price * (1 + pred_return)
            predictions.append(next_price)

            # Roll features forward for next step
            f = current_features[0].copy()

            # Shift lagged returns: ret_1 gets current predicted return
            for lag in [20, 10, 5, 3, 2]:
                src = f"ret_{lag}"
                # Approximate: higher lags decay toward zero
                if src in col_idx:
                    f[col_idx[src]] *= 0.95
            if "ret_1" in col_idx:
                f[col_idx["ret_1"]] = pred_return

            # Update rolling mean returns with EMA-style update
            for w in [5, 10, 20]:
                key = f"ret_mean_{w}"
                if key in col_idx:
                    alpha = 2.0 / (w + 1)
                    f[col_idx[key]] = alpha * pred_return + (1 - alpha) * f[col_idx[key]]

            # Adjust price-ratio features by predicted return factor
            for ratio_key in ["sma20_ratio", "sma50_ratio", "ema20_ratio"]:
                if ratio_key in col_idx:
                    f[col_idx[ratio_key]] *= (1 + pred_return)

            # RSI mean-reverts toward 50
            if "rsi" in col_idx:
                f[col_idx["rsi"]] += (50.0 - f[col_idx["rsi"]]) * 0.067  # half-life ~10 days

            current_features = f.reshape(1, -1)
            current_price = next_price

        # Generate dates
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            last_dt = pd.to_datetime(df.get("datetime", df.get("date")).iloc[-1])
            dates = [
                (last_dt + pd.Timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%d %H:%M")
                for i in range(steps)
            ]
        else:
            last_date = pd.to_datetime(df["date"].iloc[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            dates = [d.strftime("%Y-%m-%d") for d in future_dates]

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_score": round(float(confidence), 2),
            "mape": round(float(mape), 2),
        }

    def predict_intraday(self, df: pd.DataFrame, symbol: str, horizon: str = "15m") -> dict:
        """XGBoost prediction for intraday data."""
        from xgboost import XGBRegressor
        from app.ai.preprocessing import StockDataPreprocessor

        feat_df = df.copy()
        close = feat_df["close"]
        high = feat_df["high"]
        low = feat_df["low"]
        volume = feat_df["volume"].astype(float)

        # Synthetic volume for indices
        if (volume == 0).all():
            volume = pd.Series(1.0, index=volume.index)
            feat_df["volume"] = volume

        # Add context features
        feat_df = StockDataPreprocessor._add_context_features(feat_df, symbol)

        # Lighter feature set for intraday
        close = feat_df["close"]
        high = feat_df["high"]
        low = feat_df["low"]
        volume = feat_df["volume"].astype(float)
        feat_df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        feat_df["macd"] = macd_ind.macd()

        ema5 = ta.trend.EMAIndicator(close, window=5).ema_indicator()
        ema13 = ta.trend.EMAIndicator(close, window=13).ema_indicator()
        feat_df["ema_ratio"] = ema5 / ema13.replace(0, np.nan)

        vol_ma = volume.rolling(10).mean().replace(0, np.nan)
        feat_df["volume_ratio"] = volume / vol_ma
        feat_df["momentum"] = close.pct_change(3)

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        feat_df["stoch_k"] = stoch.stoch()

        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        feat_df["atr_norm"] = atr / close

        for w in [1, 3, 5, 10]:
            feat_df[f"ret_{w}"] = close.pct_change(w)

        returns = close.pct_change()
        for w in [5, 10]:
            feat_df[f"ret_mean_{w}"] = returns.rolling(w).mean()
            feat_df[f"ret_std_{w}"] = returns.rolling(w).std()

        feat_df["target"] = close.pct_change().shift(-1)
        feat_df = feat_df.dropna().reset_index(drop=True)

        feature_cols = [c for c in feat_df.columns
                        if c not in {"date", "datetime", "datetime_str", "target",
                                     "close", "open", "high", "low", "volume"}]

        if len(feat_df) < 50:
            raise ValueError(f"Not enough intraday data for XGBoost. Got {len(feat_df)}")

        X = feat_df[feature_cols].values
        y = feat_df["target"].values
        last_close = df["close"].iloc[-1]

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_path = self._model_path(symbol, suffix="_intraday")
        if self._model_is_fresh(model_path):
            saved = joblib.load(model_path)
            model = saved["model"]
            feature_cols = saved.get("feature_cols", feature_cols)
            X = feat_df[feature_cols].values
            split = int(len(X) * 0.8)
            X_test = X[split:]
        else:
            model = XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.15,
                reg_lambda=1.5,
                verbosity=0,
                early_stopping_rounds=15,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

        # MAPE
        if len(X_test) > 0:
            test_pred_returns = model.predict(X_test)
            test_actual_prices = feat_df["close"].iloc[split:split + len(y_test)].values
            test_pred_prices = test_actual_prices * (1 + test_pred_returns)
            test_actual_next = test_actual_prices * (1 + y_test)
            nonzero = test_actual_next != 0
            if nonzero.any():
                mape = float(np.mean(np.abs(
                    (test_actual_next[nonzero] - test_pred_prices[nonzero]) / test_actual_next[nonzero]
                )) * 100)
            else:
                mape = 5.0
        else:
            mape = 5.0

        confidence = max(0, min(100, 100 - mape))

        steps = self._horizon_to_steps(horizon)
        predictions = []
        current_features = X[-1:].copy()
        current_price = float(last_close)

        # Build column name -> index mapping for intraday feature updates
        col_idx = {c: i for i, c in enumerate(feature_cols)}

        for step in range(steps):
            pred_return = model.predict(current_features)[0]
            next_price = current_price * (1 + pred_return)
            predictions.append(next_price)

            # Roll intraday features forward
            f = current_features[0].copy()

            # Shift lagged returns
            for lag in [10, 5, 3]:
                src = f"ret_{lag}"
                if src in col_idx:
                    f[col_idx[src]] *= 0.95
            if "ret_1" in col_idx:
                f[col_idx["ret_1"]] = pred_return

            # Update rolling mean returns (EMA-style)
            for w in [5, 10]:
                key = f"ret_mean_{w}"
                if key in col_idx:
                    alpha = 2.0 / (w + 1)
                    f[col_idx[key]] = alpha * pred_return + (1 - alpha) * f[col_idx[key]]

            # EMA ratio adjusts by predicted return
            if "ema_ratio" in col_idx:
                f[col_idx["ema_ratio"]] *= (1 + pred_return)

            # RSI mean-reverts toward 50
            if "rsi" in col_idx:
                f[col_idx["rsi"]] += (50.0 - f[col_idx["rsi"]]) * 0.067

            # Momentum update
            if "momentum" in col_idx:
                f[col_idx["momentum"]] = pred_return

            current_features = f.reshape(1, -1)
            current_price = next_price

        last_dt = pd.to_datetime(df["datetime"].iloc[-1]) if "datetime" in df.columns else pd.Timestamp.now()
        dates = [
            (last_dt + pd.Timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%d %H:%M")
            for i in range(steps)
        ]

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_score": round(float(confidence), 2),
            "mape": round(float(mape), 2),
        }
