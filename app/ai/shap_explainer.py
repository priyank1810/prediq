"""SHAP-based model explainability for XGBoost and LSTM predictions.

Provides feature-level impact analysis using SHAP (SHapley Additive exPlanations)
to surface the top drivers behind each model's predictions.
"""

import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Graceful import: shap is an optional heavy dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning("shap package not installed — SHAP explanations will be unavailable")


class SHAPExplainer:
    """Generate SHAP-based feature importance explanations for prediction models.

    Supports:
      - XGBoost (TreeExplainer — exact, fast)
      - LSTM / Keras (GradientExplainer — approximate, cached)
    """

    TOP_K = 5

    def explain_xgboost(
        self,
        model,
        X: np.ndarray,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute SHAP values for an XGBRegressor and return the top feature drivers.

        Args:
            model: A fitted ``xgboost.XGBRegressor`` instance.
            X: 2-D feature array (n_samples, n_features) used for explanation.
                Typically the most recent sample(s) from the dataset.
            feature_cols: Column names corresponding to each feature index.

        Returns:
            A list of up to 5 dicts, each containing:
              - ``feature``: human-readable feature name
              - ``impact_value``: mean |SHAP value| for that feature
              - ``direction``: ``"positive"`` or ``"negative"``
            Returns an empty list on any error.
        """
        try:
            if not SHAP_AVAILABLE:
                logger.debug("SHAP not available; skipping XGBoost explanation")
                return []

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            return self._top_k_drivers(shap_values, feature_cols)

        except Exception:
            logger.exception("Failed to compute SHAP explanation for XGBoost model")
            return []

    def explain_lstm(
        self,
        model,
        X: np.ndarray,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute approximate SHAP values for a Keras LSTM model.

        Uses ``shap.GradientExplainer`` which is well-suited for deep-learning
        models.  Background data is sampled from ``X`` to keep computation
        tractable.  Results are cached via ``lru_cache`` on the wrapper so
        repeated calls with the same model/data are fast.

        Args:
            model: A compiled Keras ``Model`` (e.g. the attention-LSTM from
                ``LSTMPredictor``).
            X: 3-D array (n_samples, seq_len, n_features) of scaled input
                sequences.
            feature_cols: Feature names matching the last axis of ``X``.

        Returns:
            A list of up to 5 dicts (same schema as ``explain_xgboost``).
            Returns an empty list on any error.
        """
        try:
            if not SHAP_AVAILABLE:
                logger.debug("SHAP not available; skipping LSTM explanation")
                return []

            # Use the cached inner function keyed on model id and data shape
            return self._cached_lstm_explain(
                id(model), X.shape, model, X, feature_cols,
            )

        except Exception:
            logger.exception("Failed to compute SHAP explanation for LSTM model")
            return []

    @staticmethod
    @lru_cache(maxsize=32)
    def _cached_lstm_explain(
        model_id: int,
        data_shape: tuple,
        model,
        X: np.ndarray,
        feature_cols: tuple,
    ) -> List[Dict[str, Any]]:
        """Inner cached computation for LSTM SHAP values.

        ``model_id`` and ``data_shape`` form a lightweight cache key so we
        avoid recomputing when the same model+data pair is requested again.
        """
        try:
            # Sample background data (up to 100 instances) for speed
            n_background = min(100, len(X))
            background_indices = np.random.default_rng(42).choice(
                len(X), size=n_background, replace=False,
            )
            background = X[background_indices]

            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(X)

            # GradientExplainer may return a list for multi-output; take first
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # For 3-D SHAP output (samples, seq_len, features), aggregate
            # across the sequence dimension so we get per-feature importance.
            if shap_values.ndim == 3:
                shap_values = shap_values.mean(axis=1)

            # Convert feature_cols back from tuple (needed for hashing) to list
            feature_cols_list = list(feature_cols)

            return SHAPExplainer._top_k_drivers(shap_values, feature_cols_list)

        except Exception:
            logger.exception(
                "Failed inside cached LSTM SHAP computation"
            )
            return []

    def explain(
        self,
        model_type: str,
        model,
        X: np.ndarray,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Dispatch to the appropriate SHAP method based on model type.

        Args:
            model_type: One of ``"xgboost"`` or ``"lstm"``.
            model: The fitted model object.
            X: Feature data (2-D for XGBoost, 3-D for LSTM).
            feature_cols: Ordered feature names.

        Returns:
            List of dicts ``[{feature, impact_value, direction}, ...]``
            (up to 5 entries).  Returns an empty list on unrecognised
            ``model_type`` or any error.
        """
        try:
            model_type_lower = model_type.strip().lower()

            if model_type_lower == "xgboost":
                return self.explain_xgboost(model, X, feature_cols)

            if model_type_lower == "lstm":
                # lru_cache requires hashable args — convert feature_cols
                return self.explain_lstm(model, X, feature_cols)

            logger.warning("Unsupported model_type for SHAP explanation: %s", model_type)
            return []

        except Exception:
            logger.exception(
                "Unexpected error in SHAP explain() dispatcher for model_type=%s",
                model_type,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_k_drivers(
        shap_values: np.ndarray,
        feature_cols: List[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank features by mean |SHAP value| and return the top *k*.

        Args:
            shap_values: Array of shape ``(n_samples, n_features)``.
            feature_cols: Feature names aligned with the columns of
                ``shap_values``.
            k: Number of top features to return.

        Returns:
            Sorted list of dicts with keys ``feature``, ``impact_value``,
            and ``direction``.
        """
        # Mean absolute SHAP value per feature across all samples
        mean_abs = np.abs(shap_values).mean(axis=0)

        # Mean signed SHAP value (used to determine direction)
        mean_signed = shap_values.mean(axis=0)

        # Indices of top-k features by absolute impact (descending)
        top_indices = np.argsort(mean_abs)[::-1][:k]

        drivers: List[Dict[str, Any]] = []
        for idx in top_indices:
            if idx >= len(feature_cols):
                continue
            drivers.append({
                "feature": feature_cols[idx],
                "impact_value": round(float(mean_abs[idx]), 6),
                "direction": "positive" if mean_signed[idx] >= 0 else "negative",
            })

        return drivers


# Module-level singleton for convenient reuse
shap_explainer = SHAPExplainer()
