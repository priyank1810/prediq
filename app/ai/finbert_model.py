"""
FinBERT sentiment scorer — wraps ProsusAI/finbert for financial headline analysis.

Lazy-loads the model on first use.  Falls back to 0.0 scores when the model
cannot be loaded (missing libs, download failure, or FINBERT_ENABLED=False).
"""

import logging
from typing import Optional

from app.config import FINBERT_ENABLED

logger = logging.getLogger(__name__)


class FinBERTScorer:
    """Singleton-style scorer that lazy-loads ProsusAI/finbert from HuggingFace."""

    _instance: Optional["FinBERTScorer"] = None

    def __new__(cls) -> "FinBERTScorer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Guard against re-initialising on repeated __init__ calls
        if hasattr(self, "_initialised"):
            return
        self._initialised = True

        self._tokenizer = None
        self._model = None
        self._labels: list[str] = []  # populated on load ("positive","negative","neutral")
        self._loaded = False
        self._available = False  # True only after a successful load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load the model the first time it is needed.

        Returns ``True`` if the model is ready, ``False`` otherwise.
        """
        if self._loaded:
            return self._available

        self._loaded = True  # only attempt once

        if not FINBERT_ENABLED:
            logger.info("FinBERT disabled via config (FINBERT_ENABLED=False)")
            return False

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch  # noqa: F401 — ensures torch is importable
        except ImportError:
            logger.warning(
                "transformers and/or torch not installed — "
                "FinBERT unavailable, returning fallback scores"
            )
            return False

        try:
            model_name = "ProsusAI/finbert"
            logger.info("Loading FinBERT model '%s' …", model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()

            # ProsusAI/finbert label order: positive(0), negative(1), neutral(2)
            self._labels = list(self._model.config.id2label.values())
            self._available = True
            logger.info(
                "FinBERT loaded successfully (labels=%s)", self._labels
            )
            return True
        except Exception:
            logger.exception("Failed to load FinBERT model — falling back to 0.0 scores")
            return False

    @staticmethod
    def _logits_to_score(probs, labels: list[str]) -> float:
        """Convert a softmax probability vector into a single -1..+1 score.

        Mapping:
            +prob(positive) contributes positively
            -prob(negative) contributes negatively
            prob(neutral)   is ignored (adds zero)
        """
        score = 0.0
        for idx, label in enumerate(labels):
            label_lower = label.lower()
            if label_lower == "positive":
                score += float(probs[idx])
            elif label_lower == "negative":
                score -= float(probs[idx])
            # neutral contributes 0
        return max(-1.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_headlines(self, headlines: list[str]) -> list[float]:
        """Batch-score a list of headlines.

        Returns a list of floats in the range ``[-1.0, +1.0]``, one per
        headline.  Positive values indicate bullish sentiment; negative
        values indicate bearish sentiment.

        If the model is unavailable every headline receives ``0.0``.
        """
        if not headlines:
            return []

        if not self._ensure_loaded():
            return [0.0] * len(headlines)

        try:
            import torch
        except ImportError:
            logger.warning("torch disappeared after initial load — returning fallback scores")
            return [0.0] * len(headlines)

        try:
            inputs = self._tokenizer(
                headlines,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            scores: list[float] = []
            for prob_row in probabilities:
                scores.append(self._logits_to_score(prob_row, self._labels))

            logger.debug(
                "Scored %d headlines (sample: %.3f)", len(scores), scores[0]
            )
            return scores

        except Exception:
            logger.exception("FinBERT inference failed — returning fallback scores")
            return [0.0] * len(headlines)

    def score_single(self, text: str) -> float:
        """Score a single headline / text snippet.

        Convenience wrapper around :meth:`score_headlines` that returns a
        single float in ``[-1.0, +1.0]``.
        """
        if not text or not text.strip():
            return 0.0
        results = self.score_headlines([text])
        return results[0]

    @property
    def is_available(self) -> bool:
        """Check whether the model loaded (or would load) successfully."""
        return self._ensure_loaded()


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------
finbert_scorer = FinBERTScorer()
