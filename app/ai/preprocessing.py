import logging
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))

    @staticmethod
    def _winsorize(df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based Winsorization: clip extreme daily returns and volume spikes."""
        df = df.copy()
        returns = df["close"].pct_change()

        if len(returns.dropna()) < 10:
            return df

        lower = returns.quantile(0.01)
        upper = returns.quantile(0.99)
        clipped = returns.clip(lower=lower, upper=upper)
        outlier_count = int((returns != clipped).sum())

        if outlier_count > 0:
            # Recalculate close prices from clipped returns
            new_close = [df["close"].iloc[0]]
            for i in range(1, len(df)):
                new_close.append(new_close[-1] * (1 + clipped.iloc[i]))
            df["close"] = new_close
            logger.info(f"Winsorized {outlier_count} outlier returns (clipped to [{lower:.4f}, {upper:.4f}])")

        # Clip volume spikes above 99th percentile
        volume = df["volume"].astype(float)
        if (volume > 0).any():
            vol_cap = volume.quantile(0.99)
            if vol_cap > 0:
                df["volume"] = volume.clip(upper=vol_cap)

        return df

    @staticmethod
    def _flag_gaps(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
        """Detect overnight gaps where open/prev_close - 1 > threshold."""
        df = df.copy()
        prev_close = df["close"].shift(1)
        gap_ratio = (df["open"] / prev_close - 1).abs()
        df["gap_flag"] = (gap_ratio > threshold).astype(float)
        df["gap_flag"] = df["gap_flag"].fillna(0.0)
        return df

    @staticmethod
    def _add_context_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Fetch and merge external context: sentiment, VIX, S&P 500, USD/INR."""
        df = df.copy()

        # 1. Sentiment score
        sentiment_val = 0.0
        try:
            from app.services.sentiment_service import sentiment_service
            sentiment_data = sentiment_service.get_sentiment(symbol)
            sentiment_val = float(sentiment_data.get("score", 0.0))
        except Exception:
            pass
        df["sentiment_score"] = sentiment_val

        # 2. India VIX level (normalized by rolling mean)
        vix_val = 0.0
        try:
            import yfinance as yf
            vix_ticker = yf.Ticker("^INDIAVIX")
            vix_hist = vix_ticker.history(period="3mo")
            if len(vix_hist) > 0:
                current_vix = float(vix_hist["Close"].iloc[-1])
                mean_vix = float(vix_hist["Close"].mean())
                vix_val = (current_vix / mean_vix) - 1.0 if mean_vix > 0 else 0.0
        except Exception:
            pass
        df["vix_level"] = vix_val

        # 3. S&P 500 change
        sp500_val = 0.0
        try:
            import yfinance as yf
            sp_ticker = yf.Ticker("^GSPC")
            sp_hist = sp_ticker.history(period="5d")
            if len(sp_hist) >= 2:
                sp500_val = float(
                    (sp_hist["Close"].iloc[-1] - sp_hist["Close"].iloc[-2])
                    / sp_hist["Close"].iloc[-2]
                )
        except Exception:
            pass
        df["sp500_change"] = sp500_val

        # 4. USD/INR change
        usdinr_val = 0.0
        try:
            import yfinance as yf
            usd_ticker = yf.Ticker("USDINR=X")
            usd_hist = usd_ticker.history(period="5d")
            if len(usd_hist) >= 2:
                usdinr_val = float(
                    (usd_hist["Close"].iloc[-1] - usd_hist["Close"].iloc[-2])
                    / usd_hist["Close"].iloc[-2]
                )
        except Exception:
            pass
        df["usdinr_change"] = usdinr_val

        logger.info(
            f"Added 4 context features (sentiment={sentiment_val:.1f}, "
            f"vix={vix_val:.3f}, sp500={sp500_val:.4f}, usdinr={usdinr_val:.4f})"
        )
        return df

    def prepare_lstm_data(self, df: pd.DataFrame):
        """Legacy single-feature (close price only) preprocessing."""
        close_prices = df["close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled)):
            X.append(scaled[i - self.sequence_length:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X)
        y = np.array(y)
        split = int(len(X) * 0.8)

        X_train = X[:split].reshape(-1, self.sequence_length, 1)
        y_train = y[:split]
        X_test = X[split:].reshape(-1, self.sequence_length, 1)
        y_test = y[split:]

        return X_train, y_train, X_test, y_test, self.scaler, scaled

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame, close: pd.Series, volume: pd.Series, windows=(5, 20)):
        """Add multi-timescale rolling statistics."""
        cols = []
        returns = close.pct_change()
        for w in windows:
            col_ret_mean = f"ret_mean_{w}"
            col_ret_std = f"ret_std_{w}"
            col_vol_mean = f"vol_mean_{w}"
            df[col_ret_mean] = returns.rolling(w).mean()
            df[col_ret_std] = returns.rolling(w).std()
            vol_ma = volume.rolling(w).mean().replace(0, np.nan)
            df[col_vol_mean] = volume / vol_ma
            cols.extend([col_ret_mean, col_ret_std, col_vol_mean])
        return cols

    def prepare_multifeature_lstm_data(self, df: pd.DataFrame, symbol: str = ""):
        """Build ~32-feature sequences for LSTM using OHLCV + technical indicators + context.

        Features: close, open, high, low, volume_ratio, RSI(14), MACD, MACD_signal,
                  BB_percent, SMA20_ratio, SMA50_ratio, EMA20_ratio, ATR_norm,
                  ADX, Stoch_K, Stoch_D, OBV_ratio, MFI, VWAP_dev,
                  vol_regime, price_position, ret_5, ret_20,
                  + rolling features + gap_flag + context features (4)

        Returns:
            X_train, y_train, X_test, y_test, close_scaler, num_features,
            full_scaled_data, feature_scaler
        """
        # Outlier removal and gap detection
        feat_df = self._winsorize(df)
        feat_df = self._flag_gaps(feat_df)
        feat_df = self._add_context_features(feat_df, symbol)

        close = feat_df["close"]
        high = feat_df["high"]
        low = feat_df["low"]
        volume = feat_df["volume"].astype(float)

        # Indices have no volume — fill with synthetic volume to avoid NaN indicators
        if (volume == 0).all():
            volume = pd.Series(1.0, index=volume.index)
            feat_df["volume"] = volume

        # --- Original 13 features ---
        feat_df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        macd_ind = ta.trend.MACD(close)
        feat_df["macd"] = macd_ind.macd()
        feat_df["macd_signal"] = macd_ind.macd_signal()

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

        # --- New features (13 → 22+) ---

        # ADX — trend strength
        feat_df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        feat_df["stoch_k"] = stoch.stoch()
        feat_df["stoch_d"] = stoch.stoch_signal()

        # OBV ratio (OBV / OBV SMA20)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_sma = obv.rolling(20).mean().replace(0, np.nan)
        feat_df["obv_ratio"] = obv / obv_sma

        # Money Flow Index
        feat_df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # VWAP deviation
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        feat_df["vwap_dev"] = (close - vwap) / close.replace(0, np.nan)

        # Volatility regime (20-day vol / 60-day vol)
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std().replace(0, np.nan)
        feat_df["vol_regime"] = vol_20 / vol_60

        # Price position in recent range
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        feat_df["price_position"] = (close - rolling_low) / price_range

        # Multi-bar returns
        feat_df["ret_5"] = close.pct_change(5)
        feat_df["ret_20"] = close.pct_change(20)

        # Multi-timescale rolling features
        rolling_cols = self._add_rolling_features(feat_df, close, volume, windows=(5, 20))

        feature_cols = [
            "close", "open", "high", "low", "volume_ratio",
            "rsi", "macd", "macd_signal", "bb_percent",
            "sma20_ratio", "sma50_ratio", "ema20_ratio", "atr_norm",
            "adx", "stoch_k", "stoch_d", "obv_ratio", "mfi", "vwap_dev",
            "vol_regime", "price_position", "ret_5", "ret_20",
        ] + rolling_cols + [
            "gap_flag",
            "sentiment_score", "vix_level", "sp500_change", "usdinr_change",
        ]

        feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(feat_df) < self.sequence_length + 20:
            raise ValueError(
                f"Not enough data after indicator computation. "
                f"Got {len(feat_df)}, need {self.sequence_length + 20}"
            )

        feature_matrix = feat_df[feature_cols].values
        close_vals = feat_df["close"].values.reshape(-1, 1)

        # --- FIX DATA LEAKAGE: fit scalers on training portion only ---
        total_sequences = len(feature_matrix) - self.sequence_length
        split = int(total_sequences * 0.8)
        train_end = split + self.sequence_length  # raw data index

        self.feature_scaler.fit(feature_matrix[:train_end])
        scaled_features = self.feature_scaler.transform(feature_matrix)

        self.close_scaler.fit(close_vals[:train_end])

        # Build sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i - self.sequence_length:i])
            y.append(scaled_features[i, 0])  # target = scaled close (col 0)

        X = np.array(X)
        y = np.array(y)

        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]

        num_features = len(feature_cols)

        return X_train, y_train, X_test, y_test, self.close_scaler, num_features, scaled_features, self.feature_scaler

    def prepare_multifeature_intraday(self, df: pd.DataFrame, symbol: str = ""):
        """20-feature preprocessing for intraday (15-min candle) data.

        Features: close, open, high, low, volume_ratio, RSI(14), MACD, ema_ratio, momentum,
                  stoch_k, vwap_dev, vol_accel, atr_norm, ret_5, ret_10, spread
                  + rolling features + context features (4)

        Uses shorter indicator windows to preserve more bars.
        """
        seq_len = min(30, self.sequence_length)
        feat_df = self._winsorize(df)
        feat_df = self._add_context_features(feat_df, symbol)
        close = feat_df["close"]
        high = feat_df["high"]
        low = feat_df["low"]
        volume = feat_df["volume"].astype(float)

        # Indices have no volume — fill with synthetic volume to avoid NaN indicators
        if (volume == 0).all():
            volume = pd.Series(1.0, index=volume.index)
            feat_df["volume"] = volume

        # --- Original 9 features ---
        feat_df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        feat_df["macd"] = macd_ind.macd()

        ema5 = ta.trend.EMAIndicator(close, window=5).ema_indicator()
        ema13 = ta.trend.EMAIndicator(close, window=13).ema_indicator()
        feat_df["ema_ratio"] = ema5 / ema13.replace(0, np.nan)

        vol_ma = volume.rolling(10).mean().replace(0, np.nan)
        feat_df["volume_ratio"] = volume / vol_ma

        feat_df["momentum"] = close.pct_change(3)

        # --- New features (9 → 16+) ---

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        feat_df["stoch_k"] = stoch.stoch()

        # VWAP deviation
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        feat_df["vwap_dev"] = (close - vwap) / close.replace(0, np.nan)

        # Volume acceleration
        vol_shift = volume.shift(5).replace(0, np.nan)
        feat_df["vol_accel"] = volume / vol_shift

        # ATR normalized
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        feat_df["atr_norm"] = atr / close

        # Multi-bar returns
        feat_df["ret_5"] = close.pct_change(5)
        feat_df["ret_10"] = close.pct_change(10)

        # Spread placeholder (0 when Angel One not active)
        feat_df["spread"] = 0.0

        # Multi-timescale rolling features (shorter windows for intraday)
        rolling_cols = self._add_rolling_features(feat_df, close, volume, windows=(5, 10))

        feature_cols = [
            "close", "open", "high", "low", "volume_ratio",
            "rsi", "macd", "ema_ratio", "momentum",
            "stoch_k", "vwap_dev", "vol_accel", "atr_norm",
            "ret_5", "ret_10", "spread",
        ] + rolling_cols + [
            "sentiment_score", "vix_level", "sp500_change", "usdinr_change",
        ]

        feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(feat_df) < seq_len + 10:
            raise ValueError(
                f"Not enough intraday data after indicators. "
                f"Got {len(feat_df)}, need {seq_len + 10}"
            )

        feature_matrix = feat_df[feature_cols].values
        close_vals = feat_df["close"].values.reshape(-1, 1)

        # --- FIX DATA LEAKAGE: fit scalers on training portion only ---
        total_sequences = len(feature_matrix) - seq_len
        split = int(total_sequences * 0.8)
        train_end = split + seq_len

        self.feature_scaler.fit(feature_matrix[:train_end])
        scaled_features = self.feature_scaler.transform(feature_matrix)

        self.close_scaler.fit(close_vals[:train_end])

        X, y = [], []
        for i in range(seq_len, len(scaled_features)):
            X.append(scaled_features[i - seq_len:i])
            y.append(scaled_features[i, 0])

        X = np.array(X)
        y = np.array(y)

        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]

        num_features = len(feature_cols)

        return X_train, y_train, X_test, y_test, self.close_scaler, num_features, scaled_features, seq_len, self.feature_scaler

    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        prophet_df = df[["date", "close"]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        return prophet_df
