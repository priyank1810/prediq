# IPO Section — Upcoming IPOs + AI Apply/Avoid Advisor

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan

## Goal

Add a new "IPO" section to PrediQ that:

1. Lists upcoming Indian (NSE) IPOs with key issue details.
2. Produces an explainable **Apply / Neutral / Avoid** recommendation per IPO ("AI advisor").
3. Tracks how listed IPOs actually performed and grades past recommendations (scorecard).

There is **no external LLM** in this stack. "AI" here means a transparent, explainable
weighted scoring engine that emits a verdict, a confidence, a component breakdown, and
plain-English reason lines — consistent with the existing rule-based explanation style
(e.g. `explainer.py`, signal reasoning).

## Non-Goals (YAGNI)

- No XGBoost/ML training for IPO verdicts (insufficient labeled data; opacity hurts trust).
- No BSE-only IPOs in v1 (NSE coverage first).
- No brokerage integration to actually place IPO applications.
- No real-time GMP streaming; GMP refreshed on the daily job cadence.

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Upcoming IPO list, dates, price band, lot size, issue size, OFS/fresh split | Free NSE IPO JSON endpoint | Reuse existing NSE fetch style: cookie/header handling + circuit breaker (mirror `nse_breaker` in `data_fetcher.py`). |
| Subscription % (retail/QIB/HNI, total) | Free NSE JSON (subscription/bid-details) | Only meaningful near issue close; null early. |
| GMP (Grey Market Premium) | **One dedicated scrape site** | Not on official NSE. Isolated `_fetch_gmp()` with its own circuit breaker. Failure → null → auto-reweight. |
| Fundamentals (P/E vs peers, growth, debt) | Existing `fundamental_service` | For longer-term valuation view. |
| Listing-day open/close price | Yahoo Finance (existing fallback in `data_fetcher`) | Fetched once IPO has listed, to compute actual listing gain. |

**Resilience principle:** every external source is wrapped so a single source failing
degrades gracefully (component goes null and its weight redistributes) rather than
breaking the whole section. This matches the existing Angel → NSE → Yahoo fallback design.

## Architecture

Mirrors the existing service + router + tab pattern.

```
NSE free IPO JSON ──► ipo_service.fetch_upcoming()
   (circuit breaker + cookie headers)
        │  upcoming: name, symbol, open/close dates, price band,
        │  lot size, issue size, OFS/fresh, subscription%
        ▼
   enrich:
     • _fetch_gmp(symbol)            (own breaker, null-safe)
     • fundamental_service           (P/E vs peer, growth, debt)
        ▼
   ipo_advisor.score(ipo) ──► {
        verdict: APPLY | NEUTRAL | AVOID,
        confidence: 0-100,
        component_scores: {gmp, subscription, fundamentals},
        reasons: [str, ...],
        risk_flags: [str, ...]
   }
        ▼
   persist: upsert IPO row + freeze IPORecommendation row
        ▼
   API ──► IPO tab (cards + verdict badges) + scorecard sub-view
```

### New files / modules

- `app/services/ipo_service.py`
  - `fetch_upcoming()` — NSE JSON → normalized IPO dicts (with breaker).
  - `_fetch_gmp(symbol)` — isolated GMP scrape (own breaker, null-safe).
  - `refresh_and_score()` — fetch + enrich + score + persist (called by daily job and on-demand).
  - `backfill_listing_perf()` — for listed IPOs with null `listing_price`, fetch via Yahoo, compute gain.
  - `get_scorecard()` — verdict-vs-actual accuracy stats + history.
- `app/ai/ipo_advisor.py`
  - `score(ipo: dict) -> dict` — pure, deterministic, unit-testable scoring engine.
- `app/routers/ipo.py` — API endpoints (below).
- Frontend: new "IPO" tab in `templates/index.html` + lazy JS module + CSS.
- Tests under `tests/`.

### Config (in `app/config.py`)

Tunable, like existing `adaptive_weights` constants:

```python
IPO_WEIGHTS = {"gmp": 0.40, "subscription": 0.35, "fundamentals": 0.25}
IPO_VERDICT_THRESHOLDS = {"apply": 65, "avoid": 40}   # >=65 APPLY, <=40 AVOID, else NEUTRAL
IPO_REFRESH_HOUR_IST = 16   # daily refresh after market close (~4 PM IST)
IPO_GMP_SOURCE_URL = "<one scrape source>"
```

## Scoring Engine (`ipo_advisor.score`)

Weighted 0–100 score from three components (weights from `IPO_WEIGHTS`):

1. **GMP component (~40%)** — estimated listing gain % from grey market premium, mapped
   to 0–100 (e.g. higher GMP % → higher sub-score, capped). Strongest short-term signal.
2. **Subscription component (~35%)** — total oversubscription multiple mapped to 0–100.
   Only meaningful near close; null early.
3. **Fundamentals component (~25%)** — valuation vs sector peers, revenue/profit growth,
   debt load → 0–100. Longer-term anchor.

**Weight redistribution:** if any component is null (GMP scrape failed, subscription not
yet open, fundamentals unavailable for a pre-listing entity), its weight is redistributed
proportionally across the remaining available components, and a note is added to `reasons`.
If only fundamentals is available very early, confidence is reported as low.

**Verdict mapping:** `final_score >= apply_threshold → APPLY`;
`<= avoid_threshold → AVOID`; else `NEUTRAL`.

**Confidence:** function of how many components were available and their agreement
(e.g. all three present + aligned → high; single component → low).

**Reasons & risk flags:** each component emits one plain-English line
(e.g. "GMP ~22% suggests a strong listing pop", "Subscribed 4.1x overall — healthy demand",
"P/E 38 vs sector median 24 — richly valued"). Risk flags appended for: pure OFS (no fresh
capital), thin/low subscription near close, rich valuation, very small issue size, GMP missing.

## Data Model (SQLAlchemy, declarative `Base` in `app/models.py`)

### `ipos`
| Column | Type | Notes |
|--------|------|-------|
| id | int PK | |
| symbol | str, indexed | NSE symbol/ticker (nullable until assigned) |
| name | str | Company / issue name |
| open_date | date | |
| close_date | date | |
| listing_date | date, nullable | |
| price_band_low / price_band_high | float | |
| lot_size | int | |
| issue_size | float, nullable | in ₹ cr |
| is_ofs / is_fresh | bool | OFS vs fresh-issue flags |
| gmp | float, nullable | latest snapshot |
| subscription_total | float, nullable | latest snapshot, oversubscription multiple |
| listing_price | float, nullable | filled after listing |
| listing_gain_pct | float, nullable | computed after listing |
| updated_at | datetime | |

Unique constraint on (`name`, `open_date`) for upsert idempotency (symbol may be absent pre-listing).

### `ipo_recommendations`
Frozen at issue close so the scorecard is honest.

| Column | Type | Notes |
|--------|------|-------|
| id | int PK | |
| ipo_id | int FK → ipos.id | |
| verdict | str | APPLY / NEUTRAL / AVOID |
| confidence | int | 0–100 |
| score_gmp / score_subscription / score_fundamentals | float, nullable | component sub-scores at freeze time |
| final_score | float | |
| reasons | JSON/text | |
| risk_flags | JSON/text | |
| created_at | datetime | freeze timestamp |

## Post-Listing Tracking & Scorecard

Daily background task `daily_ipo_refresh` in `main.py` lifespan, mirroring
`daily_stock_learner` (IST-gated, runs after market close at `IPO_REFRESH_HOUR_IST`):

1. **Refresh upcoming** — `ipo_service.refresh_and_score()`: fetch list, enrich, re-score
   still-open IPOs, upsert rows. Freeze an `ipo_recommendations` row when an IPO reaches
   its close_date (capture the final verdict once).
2. **Backfill performance** — `backfill_listing_perf()`: for IPOs past `listing_date` with
   null `listing_price`, fetch listing-day open/close via Yahoo, compute `listing_gain_pct`.
3. **Scorecard** — `get_scorecard()`: compare each frozen verdict against actual
   `listing_gain_pct`:
   - APPLY is a "hit" if listing gain > 0 (configurable threshold), "miss" otherwise.
   - AVOID is a "hit" if listing gain <= 0.
   - NEUTRAL excluded from hit/miss or counted separately.
   - Return rolling accuracy %, sample count, and per-IPO history (verdict vs actual).

## API (`app/routers/ipo.py`)

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/ipo/upcoming` | List of upcoming IPOs with verdict, confidence, key facts |
| GET | `/api/ipo/{symbol_or_id}` | Full detail: all fields, component breakdown, reasons, risk flags |
| GET | `/api/ipo/scorecard` | Accuracy stats + per-IPO verdict-vs-actual history |

Router registered via `include_router` in `main.py` like existing routers.

## Frontend

- New **"IPO"** tab in the single-page `templates/index.html` (Jinja2 — HTML changes need app restart).
- Lazy JS module loaded via `Lazy.loadAndInit()` pattern; vanilla JS, no build step.
- **Cards** per IPO: name, open/close dates, price band, lot cost (lot_size × band_high),
  verdict badge (green APPLY / amber NEUTRAL / red AVOID), confidence, expandable section
  with reason lines + risk flags + subscription/GMP/fundamental detail.
- **Scorecard sub-view**: rolling accuracy %, sample size, table of past IPOs
  (verdict vs actual listing gain).
- Mobile-first responsive (grids/cards collapse), consistent with existing tabs.
- CSS in `static/`; served by nginx directly (no app restart for CSS/JS).

## Error Handling

- Each external fetch wrapped in a circuit breaker; failures logged, component nulled,
  weights redistributed — section never hard-fails on a single source.
- NSE anti-bot / shape changes: defensive parsing, log + skip malformed entries.
- GMP scrape change: isolated; null GMP, reason note "GMP unavailable".
- Yahoo listing-price miss: leave `listing_price` null, retry next daily run.
- Empty upcoming list: section shows "No upcoming IPOs" rather than erroring.

## Testing (`tests/`)

- Scoring thresholds → verdict mapping (APPLY/NEUTRAL/AVOID boundaries).
- Weight redistribution when one/two components null (sums to 1.0; confidence lowered).
- Reason/risk-flag generation for representative inputs (OFS-only, rich P/E, thin subscription).
- Scorecard hit/miss math for APPLY/AVOID/NEUTRAL across sample histories.
- `ipo_service` fetch parsing with **mocked** NSE JSON + mocked GMP scrape (no live network).
- Upsert idempotency (same IPO refreshed twice → one row, recommendation frozen once).

## Build Order

1. Data model (`ipos`, `ipo_recommendations`) + config constants.
2. `ipo_advisor.score()` (pure, TDD) — no I/O.
3. `ipo_service` fetch/enrich/persist (mocked sources, TDD).
4. Router endpoints.
5. Daily refresh + listing backfill + scorecard.
6. Frontend tab + scorecard view.
7. Integration pass + responsive check.
