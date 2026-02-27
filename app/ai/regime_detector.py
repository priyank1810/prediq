import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn not installed -- MarketRegimeDetector will use ADX-based heuristic fallback"
    )

# Canonical regime labels ordered from most bearish to most bullish.
# State-labeling logic sorts HMM states by mean return and maps them to these.
REGIME_LABELS = ["bear", "sideways", "volatile", "bull"]


class MarketRegimeDetector:
    """Detect the current market regime using a 4-state Gaussian HMM.

    States
    ------
    bull      : strong upward trend (ADX > 25, positive returns)
    bear      : strong downward trend (ADX > 25, negative returns)
    sideways  : low-volatility, range-bound market
    volatile  : elevated short-term volatility relative to longer-term

    The model is refit on every call to ``detect()``; upstream callers are
    expected to cache the OHLCV DataFrame so the cost stays bounded.
    """

    N_STATES = 4

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute the four regime-detection features from an OHLCV DataFrame.

        Features
        --------
        adx         : Average Directional Index (14-period) -- trend strength.
        vol_ratio   : Ratio of 20-day return volatility to 60-day return
                      volatility -- volatility regime indicator.
        price_pos   : Price position within the rolling 20-day high/low range
                      -- market-breadth proxy.
        returns     : Simple 1-period percentage return.
        """
        feat = df.copy()

        close = feat["close"]
        high = feat["high"]
        low = feat["low"]

        # 1. ADX (14)
        feat["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

        # 2. Volatility ratio (20-day / 60-day)
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std().replace(0, np.nan)
        feat["vol_ratio"] = vol_20 / vol_60

        # 3. Price position (breadth proxy) -- where close sits in recent range
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        feat["price_pos"] = (close - rolling_low) / price_range

        # 4. Returns
        feat["returns"] = returns

        feat = feat.dropna(subset=["adx", "vol_ratio", "price_pos", "returns"])
        feat = feat.reset_index(drop=True)

        return feat

    # ------------------------------------------------------------------
    # HMM-based detection
    # ------------------------------------------------------------------

    def _detect_hmm(self, features: np.ndarray) -> dict:
        """Fit a 4-state GaussianHMM and return the regime for the last row."""
        model = GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )

        model.fit(features)

        # Posterior state probabilities for the last observation
        state_probs = model.predict_proba(features)
        last_probs = state_probs[-1]  # shape (N_STATES,)

        # ----- State labelling -----
        # The HMM states are unordered.  We label them by sorting on the mean
        # return component of each state (last column of ``model.means_``).
        # Index -1 corresponds to the ``returns`` feature.
        mean_returns = model.means_[:, -1]  # one per state
        sorted_state_indices = np.argsort(mean_returns)  # ascending

        # Map: sorted position -> canonical label
        # sorted_state_indices[0] has the lowest mean return -> "bear"
        # sorted_state_indices[1] -> "sideways"
        # sorted_state_indices[2] -> "volatile"
        # sorted_state_indices[3] -> "bull"
        state_to_label = {}
        for rank, state_idx in enumerate(sorted_state_indices):
            state_to_label[state_idx] = REGIME_LABELS[rank]

        # Determine the most-likely state for the last observation
        best_state = int(np.argmax(last_probs))
        regime = state_to_label[best_state]
        confidence = float(last_probs[best_state])

        # Transition matrix reordered to canonical label order
        transmat = model.transmat_  # shape (N, N)
        reordered = np.zeros_like(transmat)
        for i_rank, i_state in enumerate(sorted_state_indices):
            for j_rank, j_state in enumerate(sorted_state_indices):
                reordered[i_rank, j_rank] = transmat[i_state, j_state]

        transition_probs = reordered.tolist()

        regime_onehot = {label: 0 for label in REGIME_LABELS}
        regime_onehot[regime] = 1

        return {
            "regime": regime,
            "confidence": round(confidence, 4),
            "transition_probs": transition_probs,
            "regime_onehot": regime_onehot,
        }

    # ------------------------------------------------------------------
    # Heuristic fallback (no hmmlearn)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_heuristic(last_row: pd.Series) -> dict:
        """ADX-based heuristic when hmmlearn is not available.

        Rules
        -----
        * ADX > 25 and positive returns  -> bull
        * ADX > 25 and negative returns  -> bear
        * vol_ratio > 1.2               -> volatile
        * otherwise                      -> sideways
        """
        adx = float(last_row["adx"])
        vol_ratio = float(last_row["vol_ratio"])
        ret = float(last_row["returns"])

        if adx > 25 and ret >= 0:
            regime = "bull"
        elif adx > 25 and ret < 0:
            regime = "bear"
        elif vol_ratio > 1.2:
            regime = "volatile"
        else:
            regime = "sideways"

        regime_onehot = {label: 0 for label in REGIME_LABELS}
        regime_onehot[regime] = 1

        # No probabilistic model -- report full confidence in the selected
        # regime and a uniform transition matrix as a neutral placeholder.
        uniform = 1.0 / len(REGIME_LABELS)
        transition_probs = [
            [uniform] * len(REGIME_LABELS) for _ in REGIME_LABELS
        ]

        return {
            "regime": regime,
            "confidence": 1.0,
            "transition_probs": transition_probs,
            "regime_onehot": regime_onehot,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> dict:
        """Detect the current market regime from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``close``, ``high``, and ``low`` columns
            with enough rows (>= 80 recommended) to compute 60-day rolling
            indicators.

        Returns
        -------
        dict
            ``regime``           -- one of ``"bull"``, ``"bear"``,
                                    ``"sideways"``, ``"volatile"``
            ``confidence``       -- probability of the detected state (0-1)
            ``transition_probs`` -- 4x4 transition probability matrix ordered
                                    ``[bear, sideways, volatile, bull]``
            ``regime_onehot``    -- ``{bull: 0/1, bear: 0/1, sideways: 0/1,
                                    volatile: 0/1}``
        """
        feat_df = self._build_features(df)
        feature_cols = ["adx", "vol_ratio", "price_pos", "returns"]

        if len(feat_df) < 30:
            logger.warning(
                "Insufficient data for regime detection (%d rows after "
                "feature computation). Falling back to heuristic.",
                len(feat_df),
            )
            return self._detect_heuristic(feat_df.iloc[-1])

        features = feat_df[feature_cols].values

        if _HMM_AVAILABLE:
            try:
                result = self._detect_hmm(features)
                logger.info(
                    "HMM regime detection: regime=%s  confidence=%.2f",
                    result["regime"],
                    result["confidence"],
                )
                return result
            except Exception:
                logger.exception(
                    "HMM fitting failed -- falling back to heuristic"
                )

        result = self._detect_heuristic(feat_df.iloc[-1])
        logger.info(
            "Heuristic regime detection: regime=%s", result["regime"]
        )
        return result
