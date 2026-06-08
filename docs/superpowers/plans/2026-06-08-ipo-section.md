# IPO Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "IPO" section that lists upcoming NSE IPOs, emits an explainable Apply/Neutral/Avoid verdict (GMP + subscription + fundamentals), and tracks post-listing performance as a scorecard.

**Architecture:** Mirror the existing service + router + tab pattern. `ipo_advisor.py` is a pure scoring engine (no I/O). `ipo_service.py` fetches free NSE IPO JSON + a single GMP scrape (each behind a circuit breaker), enriches with `fundamental_service`, scores, and persists to two new tables. A daily lifespan task refreshes the list and backfills listing-day prices via Yahoo. Frontend is a lazy-loaded vanilla-JS tab.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, requests, pytest; vanilla JS + Jinja2 + Tailwind CDN frontend.

**Spec:** `docs/superpowers/specs/2026-06-08-ipo-section-design.md`

---

## File Structure

- Create `app/ai/ipo_advisor.py` — pure scoring engine (`score(ipo: dict) -> dict`).
- Create `app/services/ipo_service.py` — fetch/enrich/persist/scorecard.
- Create `app/routers/ipo.py` — `/api/ipo/*` endpoints.
- Modify `app/models.py` — add `IPO` + `IPORecommendation` models.
- Modify `app/config.py` — add `IPO_WEIGHTS`, `IPO_VERDICT_THRESHOLDS`, `IPO_REFRESH_HOUR_IST`, `IPO_GMP_SOURCE_URL`, `IPO_APPLY_GAIN_THRESHOLD`.
- Modify `main.py` — register router + add `daily_ipo_refresh` lifespan task.
- Create `static/js/ipo.js` + `static/css/ipo.css` (if a separate css is warranted) — frontend module.
- Modify `templates/index.html` — add IPO tab + nav entry.
- Create `tests/test_ipo_advisor.py`, `tests/test_ipo_service.py`, `tests/test_ipo_api.py`.

---

## Task 1: Config constants

**Files:**
- Modify: `app/config.py`

- [ ] **Step 1: Add IPO config block** (append near other service constants)

```python
# ---------------------------------------------------------------------------
# IPO advisor
# ---------------------------------------------------------------------------
IPO_WEIGHTS = {"gmp": 0.40, "subscription": 0.35, "fundamentals": 0.25}
IPO_VERDICT_THRESHOLDS = {"apply": 65, "avoid": 40}  # >=apply APPLY, <=avoid AVOID, else NEUTRAL
IPO_REFRESH_HOUR_IST = 16            # daily refresh hour (IST), after market close
IPO_APPLY_GAIN_THRESHOLD = 0.0      # listing gain % above this counts an APPLY as a hit
IPO_GMP_SOURCE_URL = "https://www.investorgain.com/report/live-ipo-gmp/331/ipo/"  # single GMP scrape source; validate/adjust at build time
```

- [ ] **Step 2: Commit**

```bash
git add app/config.py
git commit -m "feat(ipo): add IPO advisor config constants"
```

---

## Task 2: Database models

**Files:**
- Modify: `app/models.py`
- Test: `tests/test_models.py` (extend) — but primary coverage via service tests.

- [ ] **Step 1: Add models** (append at end of `app/models.py`, before any trailing helpers)

```python
class IPO(Base):
    __tablename__ = "ipos"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=True)
    name = Column(String, nullable=False)
    open_date = Column(Date, nullable=True)
    close_date = Column(Date, nullable=True)
    listing_date = Column(Date, nullable=True)
    price_band_low = Column(Float, nullable=True)
    price_band_high = Column(Float, nullable=True)
    lot_size = Column(Integer, nullable=True)
    issue_size = Column(Float, nullable=True)        # ₹ cr
    is_ofs = Column(Boolean, default=False)
    is_fresh = Column(Boolean, default=False)
    gmp = Column(Float, nullable=True)               # latest snapshot, ₹ premium
    subscription_total = Column(Float, nullable=True)  # oversubscription multiple
    listing_price = Column(Float, nullable=True)
    listing_gain_pct = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=now_ist, onupdate=now_ist)

    recommendations = relationship("IPORecommendation", back_populates="ipo",
                                   cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("name", "open_date", name="uq_ipo_name_open"),)


class IPORecommendation(Base):
    __tablename__ = "ipo_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    ipo_id = Column(Integer, ForeignKey("ipos.id"), nullable=False, index=True)
    verdict = Column(String, nullable=False)          # APPLY / NEUTRAL / AVOID
    confidence = Column(Integer, nullable=False)       # 0-100
    final_score = Column(Float, nullable=False)
    score_gmp = Column(Float, nullable=True)
    score_subscription = Column(Float, nullable=True)
    score_fundamentals = Column(Float, nullable=True)
    reasons = Column(Text, nullable=True)              # JSON-encoded list[str]
    risk_flags = Column(Text, nullable=True)           # JSON-encoded list[str]
    created_at = Column(DateTime, default=now_ist)

    ipo = relationship("IPO", back_populates="recommendations")
```

- [ ] **Step 2: Verify tables create** (conftest calls `Base.metadata.create_all`)

Run: `python3 -m pytest tests/test_models.py -q`
Expected: PASS (no errors importing new models).

- [ ] **Step 3: Commit**

```bash
git add app/models.py
git commit -m "feat(ipo): add IPO and IPORecommendation models"
```

---

## Task 3: Scoring engine (pure, TDD)

The advisor is pure: input a normalized IPO dict, output a verdict dict. No I/O, fully unit-testable.

**Input dict keys** (any may be `None`): `gmp_pct` (estimated listing gain % from GMP), `subscription_total` (multiple), `fundamentals` (dict from `fundamental_service.get_fundamentals` or `None`), `is_ofs`, `is_fresh`, `issue_size`, `peer_pe` (optional sector median P/E or `None`).

**Output dict:** `{verdict, confidence, final_score, components: {gmp, subscription, fundamentals}, reasons: [...], risk_flags: [...]}`.

**Files:**
- Create: `app/ai/ipo_advisor.py`
- Test: `tests/test_ipo_advisor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ipo_advisor.py
"""Unit tests for the pure IPO scoring engine."""
import pytest
from app.ai.ipo_advisor import score


def test_strong_all_signals_apply():
    out = score({
        "gmp_pct": 35.0,             # big grey-market premium
        "subscription_total": 6.0,    # 6x oversubscribed
        "fundamentals": {"pe": 22, "rev_growth": 0.25, "de": 0.3},
        "peer_pe": 28,
        "is_ofs": False, "is_fresh": True, "issue_size": 800,
    })
    assert out["verdict"] == "APPLY"
    assert out["final_score"] >= 65
    assert 0 <= out["confidence"] <= 100
    assert out["reasons"]


def test_weak_all_signals_avoid():
    out = score({
        "gmp_pct": -5.0,
        "subscription_total": 0.4,
        "fundamentals": {"pe": 80, "rev_growth": -0.1, "de": 2.5},
        "peer_pe": 25,
        "is_ofs": True, "is_fresh": False, "issue_size": 200,
    })
    assert out["verdict"] == "AVOID"
    assert out["final_score"] <= 40
    assert any("OFS" in f for f in out["risk_flags"])


def test_missing_gmp_redistributes_weight():
    out = score({
        "gmp_pct": None,              # scrape failed
        "subscription_total": 5.0,
        "fundamentals": {"pe": 20, "rev_growth": 0.2, "de": 0.5},
        "peer_pe": 26,
        "is_ofs": False, "is_fresh": True, "issue_size": 500,
    })
    assert out["components"]["gmp"] is None
    assert out["verdict"] in ("APPLY", "NEUTRAL", "AVOID")
    assert any("GMP" in r for r in out["reasons"] + out["risk_flags"])


def test_only_fundamentals_low_confidence():
    out = score({
        "gmp_pct": None,
        "subscription_total": None,   # not open yet
        "fundamentals": {"pe": 24, "rev_growth": 0.15, "de": 0.6},
        "peer_pe": 25,
        "is_ofs": False, "is_fresh": True, "issue_size": 400,
    })
    assert out["confidence"] <= 50
    assert out["components"]["subscription"] is None


def test_all_missing_returns_neutral_zero_confidence():
    out = score({"gmp_pct": None, "subscription_total": None, "fundamentals": None})
    assert out["verdict"] == "NEUTRAL"
    assert out["confidence"] == 0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python3 -m pytest tests/test_ipo_advisor.py -q`
Expected: FAIL — `ModuleNotFoundError: app.ai.ipo_advisor`.

- [ ] **Step 3: Implement the engine**

```python
# app/ai/ipo_advisor.py
"""Pure, explainable IPO Apply/Neutral/Avoid scoring engine. No I/O."""
from app.config import IPO_WEIGHTS, IPO_VERDICT_THRESHOLDS


def _gmp_subscore(gmp_pct):
    """Map estimated GMP listing-gain % to 0-100. ~30%+ saturates high."""
    if gmp_pct is None:
        return None
    return max(0.0, min(100.0, 50.0 + gmp_pct * (50.0 / 30.0)))


def _subscription_subscore(mult):
    """Map total oversubscription multiple to 0-100. 1x ~ 50, 10x ~ 100."""
    if mult is None:
        return None
    if mult <= 0:
        return 0.0
    return max(0.0, min(100.0, 40.0 + (mult ** 0.5) * 19.0))


def _fundamentals_subscore(fund, peer_pe):
    """Map fundamentals to 0-100 from valuation, growth, and leverage."""
    if not fund:
        return None
    score = 50.0
    pe = fund.get("pe") or 0
    if peer_pe and pe > 0:
        if pe < peer_pe:
            score += 15
        elif pe > peer_pe * 1.5:
            score -= 20
    elif pe and pe > 60:
        score -= 15
    rg = fund.get("rev_growth")
    if rg is not None:
        score += max(-15.0, min(15.0, rg * 60.0))
    de = fund.get("de")
    if de is not None:
        if de > 2.0:
            score -= 15
        elif de < 0.5:
            score += 5
    return max(0.0, min(100.0, score))


def _redistribute(components):
    """Return effective weights over only the non-None components (sum to 1)."""
    present = {k: v for k, v in components.items() if v is not None}
    if not present:
        return {}
    base = {k: IPO_WEIGHTS[k] for k in present}
    total = sum(base.values())
    return {k: w / total for k, w in base.items()}


def _reasons_and_flags(ipo, components):
    reasons, flags = [], []
    g = ipo.get("gmp_pct")
    if g is None:
        flags.append("GMP unavailable — verdict leans on subscription & fundamentals")
    elif g >= 15:
        reasons.append(f"GMP ~{g:.0f}% suggests a strong listing pop")
    elif g <= 0:
        reasons.append(f"GMP ~{g:.0f}% signals weak/negative listing demand")
    else:
        reasons.append(f"GMP ~{g:.0f}% suggests a modest listing gain")

    s = ipo.get("subscription_total")
    if s is None:
        reasons.append("Subscription not yet open — demand unknown")
    elif s >= 3:
        reasons.append(f"Subscribed {s:.1f}x overall — healthy demand")
    elif s < 1:
        flags.append(f"Undersubscribed ({s:.1f}x) — weak demand")
    else:
        reasons.append(f"Subscribed {s:.1f}x overall")

    fund = ipo.get("fundamentals")
    peer = ipo.get("peer_pe")
    if fund:
        pe = fund.get("pe") or 0
        if peer and pe > 0 and pe > peer * 1.5:
            flags.append(f"P/E {pe:.0f} vs peer median {peer:.0f} — richly valued")
        elif peer and pe > 0 and pe < peer:
            reasons.append(f"P/E {pe:.0f} below peer median {peer:.0f} — reasonable valuation")

    if ipo.get("is_ofs") and not ipo.get("is_fresh"):
        flags.append("Pure OFS — no fresh capital raised for the business")
    if ipo.get("issue_size") is not None and ipo["issue_size"] < 100:
        flags.append("Small issue size — higher volatility risk")
    return reasons, flags


def score(ipo: dict) -> dict:
    components = {
        "gmp": _gmp_subscore(ipo.get("gmp_pct")),
        "subscription": _subscription_subscore(ipo.get("subscription_total")),
        "fundamentals": _fundamentals_subscore(ipo.get("fundamentals"), ipo.get("peer_pe")),
    }
    weights = _redistribute(components)
    reasons, risk_flags = _reasons_and_flags(ipo, components)

    if not weights:
        return {"verdict": "NEUTRAL", "confidence": 0, "final_score": 50.0,
                "components": components, "reasons": reasons or ["No data available yet"],
                "risk_flags": risk_flags}

    final = sum(components[k] * w for k, w in weights.items())
    apply_t = IPO_VERDICT_THRESHOLDS["apply"]
    avoid_t = IPO_VERDICT_THRESHOLDS["avoid"]
    verdict = "APPLY" if final >= apply_t else "AVOID" if final <= avoid_t else "NEUTRAL"

    n_present = len(weights)
    confidence = int(round({1: 35, 2: 65, 3: 90}.get(n_present, 0)
                           * (1.0 if abs(final - 50) > 10 else 0.85)))

    return {"verdict": verdict, "confidence": confidence, "final_score": round(final, 1),
            "components": components, "reasons": reasons, "risk_flags": risk_flags}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_ipo_advisor.py -q`
Expected: PASS (5 passed). Adjust subscore constants only if a boundary test legitimately misses; keep thresholds in config.

- [ ] **Step 5: Commit**

```bash
git add app/ai/ipo_advisor.py tests/test_ipo_advisor.py
git commit -m "feat(ipo): add explainable IPO scoring engine"
```

---

## Task 4: IPO service — fetch, enrich, persist (TDD with mocks)

**Files:**
- Create: `app/services/ipo_service.py`
- Test: `tests/test_ipo_service.py`

The service exposes:
- `fetch_upcoming() -> list[dict]` — NSE JSON → normalized dicts (behind `ipo_nse_breaker`).
- `_fetch_gmp_map() -> dict[str, float]` — scrape → `{name_or_symbol: gmp_pct}` (behind `ipo_gmp_breaker`).
- `refresh_and_score(db) -> dict` — fetch, enrich, score, upsert IPO rows, freeze recommendation at close. Returns summary counts.
- `backfill_listing_perf(db) -> int` — fill listing prices via Yahoo for listed rows.
- `get_upcoming(db) -> list[dict]`, `get_detail(db, symbol_or_id) -> dict`, `get_scorecard(db) -> dict`.

First add two circuit breakers.

- [ ] **Step 1: Add IPO breakers** in `app/utils/circuit_breaker.py` (after `nse_breaker`)

```python
ipo_nse_breaker = CircuitBreaker("ipo_nse", failure_threshold=5, recovery_timeout=300)
ipo_gmp_breaker = CircuitBreaker("ipo_gmp", failure_threshold=5, recovery_timeout=300)
```

- [ ] **Step 2: Write failing tests** (mock all network)

```python
# tests/test_ipo_service.py
"""Tests for ipo_service — all network mocked."""
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from app.models import IPO, IPORecommendation
from app.services.ipo_service import ipo_service


SAMPLE_UPCOMING = [{
    "symbol": "ACME", "name": "Acme Tech Ltd",
    "open_date": date.today(), "close_date": date.today() + timedelta(days=2),
    "price_band_low": 100.0, "price_band_high": 110.0, "lot_size": 135,
    "issue_size": 500.0, "is_ofs": False, "is_fresh": True,
    "subscription_total": 5.0,
}]


@patch.object(ipo_service, "_fetch_gmp_map", return_value={"ACME": 30.0})
@patch.object(ipo_service, "fetch_upcoming", return_value=SAMPLE_UPCOMING)
@patch("app.services.ipo_service.fundamental_service")
def test_refresh_upserts_and_scores(mock_fund, _f1, _f2, db):
    mock_fund.get_fundamentals.return_value = {"pe": 20, "rev_growth": 0.25, "de": 0.4}
    summary = ipo_service.refresh_and_score(db)
    rows = db.query(IPO).all()
    assert len(rows) == 1
    assert rows[0].name == "Acme Tech Ltd"
    assert rows[0].gmp == 30.0
    assert summary["upserted"] == 1


@patch.object(ipo_service, "_fetch_gmp_map", return_value={"ACME": 30.0})
@patch.object(ipo_service, "fetch_upcoming", return_value=SAMPLE_UPCOMING)
@patch("app.services.ipo_service.fundamental_service")
def test_refresh_is_idempotent(mock_fund, _f1, _f2, db):
    mock_fund.get_fundamentals.return_value = {"pe": 20, "rev_growth": 0.25, "de": 0.4}
    ipo_service.refresh_and_score(db)
    ipo_service.refresh_and_score(db)
    assert db.query(IPO).count() == 1


@patch.object(ipo_service, "_fetch_gmp_map", return_value={})
def test_freeze_recommendation_at_close(_gmp, db):
    closed = [{
        "symbol": "DONE", "name": "Done Ltd",
        "open_date": date.today() - timedelta(days=5),
        "close_date": date.today() - timedelta(days=1),
        "price_band_low": 50.0, "price_band_high": 55.0, "lot_size": 200,
        "issue_size": 300.0, "is_ofs": False, "is_fresh": True,
        "subscription_total": 4.0,
    }]
    with patch.object(ipo_service, "fetch_upcoming", return_value=closed), \
         patch("app.services.ipo_service.fundamental_service") as mf:
        mf.get_fundamentals.return_value = {"pe": 18, "rev_growth": 0.2, "de": 0.3}
        ipo_service.refresh_and_score(db)
        ipo_service.refresh_and_score(db)  # second pass must not double-freeze
    assert db.query(IPORecommendation).count() == 1


def test_backfill_listing_perf(db):
    ipo = IPO(name="List Ltd", symbol="LIST",
              open_date=date.today() - timedelta(days=10),
              close_date=date.today() - timedelta(days=8),
              listing_date=date.today() - timedelta(days=5),
              price_band_high=100.0, price_band_low=95.0, lot_size=100)
    db.add(ipo); db.commit()
    with patch("app.services.ipo_service.yahoo_quote", return_value={"price": 130.0}):
        n = ipo_service.backfill_listing_perf(db)
    db.refresh(ipo)
    assert n == 1
    assert ipo.listing_price == 130.0
    assert round(ipo.listing_gain_pct, 1) == 30.0   # vs price_band_high 100


def test_scorecard_hit_miss(db):
    ipo = IPO(name="Score Ltd", symbol="SCOR", price_band_high=100.0,
              listing_price=120.0, listing_gain_pct=20.0,
              listing_date=date.today() - timedelta(days=2))
    db.add(ipo); db.commit()
    db.add(IPORecommendation(ipo_id=ipo.id, verdict="APPLY", confidence=80,
                             final_score=70.0)); db.commit()
    card = ipo_service.get_scorecard(db)
    assert card["total_graded"] == 1
    assert card["hits"] == 1
    assert card["accuracy_pct"] == 100.0
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `python3 -m pytest tests/test_ipo_service.py -q`
Expected: FAIL — module/methods missing.

- [ ] **Step 4: Implement the service**

```python
# app/services/ipo_service.py
"""Upcoming-IPO ingestion, scoring, persistence, and scorecard."""
import json
import logging
from datetime import date

import requests

from app.ai.ipo_advisor import score as score_ipo
from app.config import IPO_APPLY_GAIN_THRESHOLD, IPO_GMP_SOURCE_URL
from app.models import IPO, IPORecommendation
from app.services.fundamental_service import fundamental_service
from app.utils.circuit_breaker import ipo_gmp_breaker, ipo_nse_breaker
from app.utils.yahoo_api import yahoo_quote

logger = logging.getLogger(__name__)

NSE_IPO_URL = "https://www.nseindia.com/api/all-upcoming-issues?category=ipo"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _to_date(s):
    from datetime import datetime
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return None


class IPOService:
    # ---- fetch -----------------------------------------------------------
    def fetch_upcoming(self) -> list:
        """Normalized upcoming-IPO dicts from NSE. [] on failure."""
        if not ipo_nse_breaker.allow_request():
            return []
        try:
            s = requests.Session()
            s.headers.update(_NSE_HEADERS)
            s.get("https://www.nseindia.com", timeout=10)  # seed cookies
            r = s.get(NSE_IPO_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            ipo_nse_breaker.record_success()
        except Exception as e:  # noqa: BLE001
            ipo_nse_breaker.record_failure()
            logger.warning("NSE IPO fetch failed: %s", e)
            return []

        rows = data if isinstance(data, list) else data.get("data", [])
        out = []
        for it in rows:
            try:
                low, high = self._parse_band(it.get("priceBand") or it.get("issuePrice"))
                out.append({
                    "symbol": (it.get("symbol") or "").strip() or None,
                    "name": (it.get("companyName") or it.get("name") or "").strip(),
                    "open_date": _to_date(it.get("issueStartDate") or it.get("openDate")),
                    "close_date": _to_date(it.get("issueEndDate") or it.get("closeDate")),
                    "price_band_low": low, "price_band_high": high,
                    "lot_size": self._to_int(it.get("lotSize") or it.get("marketLot")),
                    "issue_size": self._to_float(it.get("issueSize")),
                    "is_ofs": "OFS" in str(it.get("series") or it.get("issueType") or "").upper(),
                    "is_fresh": True,
                    "subscription_total": self._to_float(it.get("noOfSharesBid")) and None,
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("skip malformed IPO row: %s", e)
        return [r for r in out if r["name"]]

    def _fetch_gmp_map(self) -> dict:
        """Scrape one GMP source → {symbol_or_name_upper: gmp_pct}. {} on failure."""
        if not ipo_gmp_breaker.allow_request():
            return {}
        try:
            r = requests.get(IPO_GMP_SOURCE_URL, timeout=10,
                             headers={"User-Agent": _NSE_HEADERS["User-Agent"]})
            r.raise_for_status()
            ipo_gmp_breaker.record_success()
            return self._parse_gmp(r.text)
        except Exception as e:  # noqa: BLE001
            ipo_gmp_breaker.record_failure()
            logger.warning("GMP scrape failed: %s", e)
            return {}

    def _parse_gmp(self, html: str) -> dict:
        """Best-effort parse of GMP source. Returns {KEY: gmp_pct}.

        Implementation note: the exact selectors depend on IPO_GMP_SOURCE_URL.
        Parse the table of (company, estimated listing gain %) and key by the
        uppercased company name. Return {} if structure not recognized.
        """
        try:
            import re
            out = {}
            # Rows like: Acme Tech ... 27.27%  → capture name + trailing percent
            for m in re.finditer(r">([A-Za-z][A-Za-z0-9 &.\-]{2,60})<[^%]*?([\-]?\d{1,3}(?:\.\d+)?)%", html):
                name, pct = m.group(1).strip().upper(), float(m.group(2))
                if name and name not in out:
                    out[name] = pct
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ---- enrich + persist ------------------------------------------------
    def refresh_and_score(self, db) -> dict:
        upcoming = self.fetch_upcoming()
        gmp_map = self._fetch_gmp_map()
        upserted = frozen = 0
        for item in upcoming:
            gmp_pct = self._lookup_gmp(gmp_map, item)
            fund = None
            if item.get("symbol"):
                try:
                    fund = fundamental_service.get_fundamentals(item["symbol"])
                except Exception:  # noqa: BLE001
                    fund = None
            verdict = score_ipo({
                "gmp_pct": gmp_pct,
                "subscription_total": item.get("subscription_total"),
                "fundamentals": fund,
                "peer_pe": None,
                "is_ofs": item.get("is_ofs"), "is_fresh": item.get("is_fresh"),
                "issue_size": item.get("issue_size"),
            })
            row = self._upsert(db, item, gmp_pct)
            upserted += 1
            if self._should_freeze(db, row):
                self._freeze(db, row, verdict)
                frozen += 1
        db.commit()
        return {"upserted": upserted, "frozen": frozen, "fetched": len(upcoming)}

    def _upsert(self, db, item, gmp_pct) -> IPO:
        row = (db.query(IPO)
               .filter(IPO.name == item["name"], IPO.open_date == item.get("open_date"))
               .first())
        if row is None:
            row = IPO(name=item["name"], open_date=item.get("open_date"))
            db.add(row)
        row.symbol = item.get("symbol") or row.symbol
        row.close_date = item.get("close_date")
        row.price_band_low = item.get("price_band_low")
        row.price_band_high = item.get("price_band_high")
        row.lot_size = item.get("lot_size")
        row.issue_size = item.get("issue_size")
        row.is_ofs = bool(item.get("is_ofs"))
        row.is_fresh = bool(item.get("is_fresh"))
        if gmp_pct is not None:
            row.gmp = gmp_pct
        if item.get("subscription_total") is not None:
            row.subscription_total = item["subscription_total"]
        db.flush()
        return row

    def _should_freeze(self, db, row) -> bool:
        if not row.close_date or row.close_date > date.today():
            return False
        existing = (db.query(IPORecommendation)
                    .filter(IPORecommendation.ipo_id == row.id).first())
        return existing is None

    def _freeze(self, db, row, verdict):
        db.add(IPORecommendation(
            ipo_id=row.id, verdict=verdict["verdict"], confidence=verdict["confidence"],
            final_score=verdict["final_score"],
            score_gmp=verdict["components"]["gmp"],
            score_subscription=verdict["components"]["subscription"],
            score_fundamentals=verdict["components"]["fundamentals"],
            reasons=json.dumps(verdict["reasons"]),
            risk_flags=json.dumps(verdict["risk_flags"]),
        ))

    # ---- listing backfill ------------------------------------------------
    def backfill_listing_perf(self, db) -> int:
        rows = (db.query(IPO)
                .filter(IPO.listing_date != None, IPO.listing_date <= date.today(),  # noqa: E711
                        IPO.listing_price == None).all())  # noqa: E711
        n = 0
        for row in rows:
            if not row.symbol:
                continue
            try:
                q = yahoo_quote(row.symbol)
            except Exception:  # noqa: BLE001
                q = None
            price = (q or {}).get("price")
            if not price:
                continue
            row.listing_price = float(price)
            base = row.price_band_high or row.price_band_low
            if base:
                row.listing_gain_pct = round((float(price) - base) / base * 100, 2)
            n += 1
        db.commit()
        return n

    # ---- reads -----------------------------------------------------------
    def get_upcoming(self, db) -> list:
        rows = (db.query(IPO)
                .filter((IPO.close_date == None) | (IPO.close_date >= date.today()))  # noqa: E711
                .order_by(IPO.open_date).all())
        return [self._serialize(db, r) for r in rows]

    def get_detail(self, db, key) -> dict:
        q = db.query(IPO)
        row = (q.filter(IPO.id == int(key)).first() if str(key).isdigit()
               else q.filter(IPO.symbol == str(key).upper()).first())
        return self._serialize(db, row, detail=True) if row else None

    def get_scorecard(self, db) -> dict:
        graded = (db.query(IPO, IPORecommendation)
                  .join(IPORecommendation, IPORecommendation.ipo_id == IPO.id)
                  .filter(IPO.listing_gain_pct != None).all())  # noqa: E711
        hits = total = 0
        history = []
        for ipo, rec in graded:
            if rec.verdict == "NEUTRAL":
                history.append(self._grade_row(ipo, rec, None))
                continue
            total += 1
            gain = ipo.listing_gain_pct
            hit = (rec.verdict == "APPLY" and gain > IPO_APPLY_GAIN_THRESHOLD) or \
                  (rec.verdict == "AVOID" and gain <= IPO_APPLY_GAIN_THRESHOLD)
            hits += int(hit)
            history.append(self._grade_row(ipo, rec, hit))
        return {"total_graded": total, "hits": hits,
                "accuracy_pct": round(hits / total * 100, 1) if total else 0.0,
                "history": history}

    # ---- helpers ---------------------------------------------------------
    def _lookup_gmp(self, gmp_map, item):
        if not gmp_map:
            return None
        for key in (item.get("symbol"), item.get("name")):
            if key and key.upper() in gmp_map:
                return gmp_map[key.upper()]
        nm = (item.get("name") or "").upper()
        for k, v in gmp_map.items():
            if nm and (nm in k or k in nm):
                return v
        return None

    def _serialize(self, db, row, detail=False):
        rec = (db.query(IPORecommendation)
               .filter(IPORecommendation.ipo_id == row.id)
               .order_by(IPORecommendation.created_at.desc()).first())
        d = {
            "id": row.id, "symbol": row.symbol, "name": row.name,
            "open_date": row.open_date.isoformat() if row.open_date else None,
            "close_date": row.close_date.isoformat() if row.close_date else None,
            "price_band_low": row.price_band_low, "price_band_high": row.price_band_high,
            "lot_size": row.lot_size, "issue_size": row.issue_size,
            "lot_cost": (row.lot_size * row.price_band_high)
                        if row.lot_size and row.price_band_high else None,
            "gmp": row.gmp, "subscription_total": row.subscription_total,
            "is_ofs": row.is_ofs, "is_fresh": row.is_fresh,
            "listing_gain_pct": row.listing_gain_pct,
            "verdict": rec.verdict if rec else None,
            "confidence": rec.confidence if rec else None,
        }
        if detail and rec:
            d["reasons"] = json.loads(rec.reasons or "[]")
            d["risk_flags"] = json.loads(rec.risk_flags or "[]")
            d["components"] = {"gmp": rec.score_gmp, "subscription": rec.score_subscription,
                               "fundamentals": rec.score_fundamentals}
        return d

    def _grade_row(self, ipo, rec, hit):
        return {"name": ipo.name, "symbol": ipo.symbol, "verdict": rec.verdict,
                "listing_gain_pct": ipo.listing_gain_pct, "hit": hit}

    @staticmethod
    def _parse_band(s):
        import re
        if not s:
            return None, None
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s))]
        if not nums:
            return None, None
        return min(nums), max(nums)

    @staticmethod
    def _to_int(v):
        try:
            return int(float(str(v).replace(",", "")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(v):
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return None


ipo_service = IPOService()
```

- [ ] **Step 5: Run tests, verify pass**

Run: `python3 -m pytest tests/test_ipo_service.py -q`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add app/services/ipo_service.py tests/test_ipo_service.py app/utils/circuit_breaker.py
git commit -m "feat(ipo): add IPO ingestion, scoring persistence, and scorecard service"
```

---

## Task 5: API router (TDD)

**Files:**
- Create: `app/routers/ipo.py`
- Modify: `main.py` (register router)
- Test: `tests/test_ipo_api.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ipo_api.py
from datetime import date, timedelta
from app.models import IPO, IPORecommendation


def _seed(db):
    ipo = IPO(name="Acme Tech Ltd", symbol="ACME",
              open_date=date.today(), close_date=date.today() + timedelta(days=2),
              price_band_low=100.0, price_band_high=110.0, lot_size=135, gmp=25.0,
              subscription_total=5.0, is_fresh=True)
    db.add(ipo); db.commit()
    db.add(IPORecommendation(ipo_id=ipo.id, verdict="APPLY", confidence=80,
                             final_score=72.0, reasons='["good"]', risk_flags='[]',
                             score_gmp=90.0, score_subscription=80.0,
                             score_fundamentals=60.0)); db.commit()
    return ipo


def test_upcoming_endpoint(client, db):
    _seed(db)
    r = client.get("/api/ipo/upcoming")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["name"] == "Acme Tech Ltd"
    assert body[0]["verdict"] == "APPLY"


def test_detail_endpoint(client, db):
    _seed(db)
    r = client.get("/api/ipo/ACME")
    assert r.status_code == 200
    assert r.json()["reasons"] == ["good"]


def test_detail_404(client, db):
    r = client.get("/api/ipo/NOPE")
    assert r.status_code == 404


def test_scorecard_endpoint(client, db):
    r = client.get("/api/ipo/scorecard")
    assert r.status_code == 200
    assert "accuracy_pct" in r.json()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python3 -m pytest tests/test_ipo_api.py -q`
Expected: FAIL — 404s / router not registered.

- [ ] **Step 3: Implement router**

```python
# app/routers/ipo.py
"""IPO section API: upcoming list, detail, scorecard."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.ipo_service import ipo_service

router = APIRouter()


@router.get("/upcoming")
def upcoming(db: Session = Depends(get_db)):
    return ipo_service.get_upcoming(db)


@router.get("/scorecard")
def scorecard(db: Session = Depends(get_db)):
    return ipo_service.get_scorecard(db)


@router.get("/{key}")
def detail(key: str, db: Session = Depends(get_db)):
    out = ipo_service.get_detail(db, key)
    if not out:
        raise HTTPException(status_code=404, detail="IPO not found")
    return out
```

Note: declare `/scorecard` before `/{key}` so the static path is not captured by the param route.

- [ ] **Step 4: Register router in `main.py`**

Add import near other router imports (around line 32):

```python
from app.routers.ipo import router as ipo_router
```

Add registration in the `include_router` block (after the mtf line ~1035):

```python
app.include_router(ipo_router, prefix="/api/ipo", tags=["ipo"])
```

- [ ] **Step 5: Run tests, verify pass**

Run: `python3 -m pytest tests/test_ipo_api.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add app/routers/ipo.py main.py tests/test_ipo_api.py
git commit -m "feat(ipo): add /api/ipo endpoints and register router"
```

---

## Task 6: Daily refresh background task

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the task** (place near `daily_stock_learner`, ~line 433)

```python
async def daily_ipo_refresh():
    """Refresh upcoming IPOs, freeze close-date recommendations, backfill listings.

    Runs once daily after market close (IPO_REFRESH_HOUR_IST), mirroring
    daily_stock_learner's IST-gated loop.
    """
    import asyncio
    from app.config import IPO_REFRESH_HOUR_IST
    from app.database import SessionLocal
    from app.services.ipo_service import ipo_service
    from app.utils.helpers import now_ist

    await asyncio.sleep(600)  # let services warm up
    last_run_date = None
    while True:
        try:
            now = now_ist()
            if now.hour >= IPO_REFRESH_HOUR_IST and last_run_date != now.date():
                db = SessionLocal()
                try:
                    summary = await asyncio.to_thread(ipo_service.refresh_and_score, db)
                    backfilled = await asyncio.to_thread(ipo_service.backfill_listing_perf, db)
                    logger.info("IPO refresh: %s, backfilled=%s", summary, backfilled)
                    last_run_date = now.date()
                finally:
                    db.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("daily_ipo_refresh error: %s", e)
        await asyncio.sleep(1800)  # check every 30 min
```

- [ ] **Step 2: Schedule it in lifespan** — find where `daily_stock_learner` is launched via `asyncio.create_task(...)` and add alongside:

```python
        asyncio.create_task(daily_ipo_refresh())
```

(Match the exact indentation/context of the existing `create_task` calls in the lifespan startup block.)

- [ ] **Step 3: Smoke check import** (no network)

Run: `python3 -c "import main; print('ok')"`
Expected: prints `ok` (module imports without error).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(ipo): schedule daily IPO refresh + listing backfill task"
```

---

## Task 7: Frontend tab

Follow the existing single-page lazy-module pattern. First read how an existing tab registers (e.g. screener or watchlist) in `templates/index.html` and `static/js/`, then mirror it exactly. Do NOT invent a new pattern.

**Files:**
- Create: `static/js/ipo.js`
- Modify: `templates/index.html`

- [ ] **Step 1: Inspect the existing tab pattern**

Run: `grep -n "screener\|Lazy.loadAndInit\|data-tab\|tab-content" templates/index.html | head -40`
Read one existing `static/js/<tab>.js` to copy its init/fetch/render shape.

- [ ] **Step 2: Add nav entry + empty tab container in `index.html`**

Add an "IPO" entry to the tab navigation and a matching `<div>` content panel with a container the JS fills (e.g. `<div id="ipo-list"></div>` and `<div id="ipo-scorecard"></div>`), matching the markup of an existing tab. Wire it into the same `Lazy.loadAndInit('ipo', ...)` mechanism the other tabs use.

- [ ] **Step 3: Implement `static/js/ipo.js`**

Mirror the existing module shape. Behavior:
- On init, `fetch('/api/ipo/upcoming')` → render cards: name, symbol, dates, price band, lot cost, verdict badge (green APPLY / amber NEUTRAL / red AVOID), confidence.
- Card click → `fetch('/api/ipo/' + (symbol||id))` → expand reasons + risk_flags + component scores.
- A "Scorecard" toggle → `fetch('/api/ipo/scorecard')` → render accuracy %, sample count, history table (verdict vs actual listing gain, hit/miss).
- Use the same Tailwind classes and badge/card helpers other tabs use. Mobile-first: cards stack on small screens.

Keep all DOM strings escaped the way the existing modules do (reuse any shared `escapeHtml`/render helper if present).

- [ ] **Step 4: Manual verification**

Run the app: `python3 -m uvicorn main:app --host 0.0.0.0 --port 8000`
Open `http://localhost:8000`, click the IPO tab. With an empty DB it should show an empty/"No upcoming IPOs" state without console errors. (Seed a row via a Python shell using `ipo_service.refresh_and_score` with mocked sources if a populated view is needed.)

- [ ] **Step 5: Commit**

```bash
git add static/js/ipo.js templates/index.html
git commit -m "feat(ipo): add IPO tab UI with verdicts and scorecard"
```

---

## Task 8: Full suite + final commit

- [ ] **Step 1: Run the whole test suite**

Run: `python3 -m pytest tests/ -q`
Expected: all pass (existing + new IPO tests).

- [ ] **Step 2: Fix any regressions** introduced by model/router/main changes, then re-run.

- [ ] **Step 3: Final commit (if any fixups)**

```bash
git add -A
git commit -m "test(ipo): green full suite after IPO section"
```

---

## Self-Review

**Spec coverage:**
- Upcoming list (NSE free JSON) → Task 4 `fetch_upcoming`. ✓
- GMP via one isolated scrape, null-safe reweight → Task 4 `_fetch_gmp_map` + Task 3 redistribution. ✓
- Subscription + fundamentals signals → Task 3 subscores. ✓
- Explainable Apply/Neutral/Avoid + confidence + reasons + risk flags → Task 3. ✓
- Two tables, frozen recommendation → Task 2 + Task 4 `_freeze`/`_should_freeze`. ✓
- Daily refresh + listing backfill → Task 6 + Task 4 `backfill_listing_perf`. ✓
- Scorecard accuracy → Task 4 `get_scorecard` + Task 5 endpoint. ✓
- API endpoints → Task 5. ✓
- Frontend tab + scorecard view → Task 7. ✓
- Tests (thresholds, redistribution, scorecard math, mocked fetch, idempotency) → Tasks 3–5. ✓
- Config-tunable weights/thresholds → Task 1. ✓

**Known build-time validations (not placeholders, real external unknowns):**
- Exact NSE IPO JSON field names (`issueStartDate`, `priceBand`, etc.) and the GMP source HTML structure are best-effort; verify against live payloads during Task 4 and adjust `fetch_upcoming`/`_parse_gmp` mapping. Tests mock these, so the contract (normalized dict shape) is what's pinned.
- `subscription_total` from NSE upcoming feed is often absent pre-close; left `None` → reweights. Populate from a bid-details call later if desired (out of v1 scope).

**Type consistency:** verdict dict keys (`verdict`, `confidence`, `final_score`, `components{gmp,subscription,fundamentals}`, `reasons`, `risk_flags`) are identical across advisor → service `_freeze` → serializer → API → tests. ✓
