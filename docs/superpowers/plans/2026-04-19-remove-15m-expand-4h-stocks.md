# Remove 15m Signals & Expand 4h Stock Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the 15m intraday signal entirely from signal computation, API, and UI; add ~30 Nifty Midcap 100 stocks to the 4h scan universe.

**Architecture:** Signal removal is purely subtractive — delete the `intraday_15m` computation and its downstream references while keeping `df_15m` data fetch (30m resampling still needs it). Stock expansion is additive — append symbols to `POPULAR_STOCKS` in config. MTF dashboard has its own 15m path that must also be cleaned.

**Tech Stack:** Python/FastAPI backend, Vanilla JS frontend, SQLAlchemy, Jinja2 templates.

---

### Task 1: Remove 15m from signal_service.py

**Files:**
- Modify: `app/services/signal_service.py`

- [ ] **Step 1: Remove `intraday_15m` from `_TIMEFRAME_HORIZON_MAP`**

In `app/services/signal_service.py`, find the `_TIMEFRAME_HORIZON_MAP` dict (around line 485) and remove the 15m entry:

```python
_TIMEFRAME_HORIZON_MAP = {
    "intraday_30m": "30m",
    "short_1h": "1h",
    "short_4h": "4h",
}
```

- [ ] **Step 2: Remove `intraday_15m` signal computation**

In `get_multi_timeframe_signals`, remove the `intraday_15m` compute block and update the comment (around line 730):

Remove these lines entirely:
```python
# 15 min — triggers (15m candles)
intraday_15m = self._compute_timeframe_signal(
    label="15 Min (Triggers)", df=df_15m,
    current_price=current_price,
    sentiment_score=sentiment_score, global_score=global_score,
    fundamental_score=fundamental_score, news_magnitude=news_magnitude,
    timeframe="intraday", symbol=symbol,
    prediction=pred_results.get("intraday_15m", {}),
)
```

Update the comment above `intraday_30m` to:
```python
# ── INTRADAY SIGNALS ──
# 30 min — trend direction (1-hour resampled for broader view)
```

- [ ] **Step 3: Remove `intraday_15m` from `all_signals` dict**

Find the `all_signals` dict (around line 761) and update to:

```python
all_signals = {
    "intraday_30m": intraday_30m,
    "short_1h": short_1h,
    "short_4h": short_4h,
}
```

- [ ] **Step 4: Remove `"15m"` from the return dict's `intraday` key**

Find the return statement's `intraday` dict and update to:

```python
return {
    "symbol": symbol,
    "current_price": round(current_price, 2),
    "timestamp": now_ist().isoformat(),
    "market_open": is_market_open(),
    "intraday": {
        "30m": intraday_30m,
    },
    "short_term": {
        "1h": short_1h,
        "4h": short_4h,
    },
}
```

- [ ] **Step 5: Verify no remaining `intraday_15m` or bare `15m` signal references**

```bash
grep -n 'intraday_15m\|"15m"\|15 Min' app/services/signal_service.py
```

Expected: only `df_15m` variable references remain (data fetch + resampling), no signal-related hits.

- [ ] **Step 6: Commit**

```bash
git add app/services/signal_service.py
git commit -m "feat: remove 15m signal from MTF computation"
```

---

### Task 2: Clean up MTF dashboard router

**Files:**
- Modify: `app/routers/mtf_dashboard.py`

- [ ] **Step 1: Remove 15m from the cache TTL map**

Find the timeframe TTL dict (around line 27) and remove the 15m entry:

```python
_TF_CACHE_TTL = {
    "1h": 300,
    "4h": 900,
    "1d": 3600,
}
```

- [ ] **Step 2: Remove 15m signal computation**

In the MTF endpoint function (around line 243), remove:
```python
tf_15m = await asyncio.to_thread(_compute_tf, sym, "15m", df_15m)
```

And update the `timeframes` list:
```python
timeframes = [tf_1h, tf_4h, tf_1d]
```

- [ ] **Step 3: Update docstring/comment**

Update the module docstring (line 3) and endpoint docstring (line 209) to say `1h, 4h, and 1D` instead of `15m, 1h, 4h, and 1D`.

- [ ] **Step 4: Verify**

```bash
grep -n '15m\|tf_15m' app/routers/mtf_dashboard.py
```

Expected: no results.

- [ ] **Step 5: Commit**

```bash
git add app/routers/mtf_dashboard.py
git commit -m "feat: remove 15m from MTF dashboard"
```

---

### Task 3: Clean up UI — templates and JS

**Files:**
- Modify: `templates/index.html`
- Modify: `static/js/insights.js`
- Modify: `static/js/mtf.js`
- Modify: `static/js/signals.js`

- [ ] **Step 1: Remove 15m from index.html dropdowns and buttons**

In `templates/index.html`:

Remove the 15m horizon button (around line 362):
```html
<!-- DELETE THIS LINE -->
<button class="horizon-btn" data-horizon="15m">15 Min</button>
```

Remove both `<option value="intraday_15m">15m</option>` entries (around lines 839 and 956).

- [ ] **Step 2: Update tfShort maps in insights.js**

In `static/js/insights.js`, all occurrences of:
```js
const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };
```

Change to (remove `intraday_15m: '15m'`):
```js
const tfShort = { intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };
```

Do this for ALL occurrences (lines ~31, 306, 695, 786, 1057, 1173).

- [ ] **Step 3: Update tfOrder in insights.js**

Find (around line 138):
```js
const tfOrder = ['15m','30m','1h','4h','1d'];
```

Change to:
```js
const tfOrder = ['30m','1h','4h','1d'];
```

- [ ] **Step 4: Update scanLabel in insights.js**

Find (around line 742):
```js
const scanLabel = { intraday: 'Intraday (15m/30m)', short: 'Short-term (1h/4h)', full: 'Full scan' };
```

Change to:
```js
const scanLabel = { intraday: 'Intraday (30m)', short: 'Short-term (1h/4h)', full: 'Full scan' };
```

- [ ] **Step 5: Update mtf.js comment**

In `static/js/mtf.js` line 4, change:
```js
 * Displays a grid of 4 timeframes (15m, 1h, 4h, 1D) with signal direction,
```
To:
```js
 * Displays a grid of 3 timeframes (1h, 4h, 1D) with signal direction,
```

- [ ] **Step 6: Update signals.js tfLabel**

In `static/js/signals.js` around line 437-442, remove the 15min case:

```js
const tfLabel = best === '30min' ? '30 Min' : best === '1h' ? '1 Hour' : '4 Hours';
```

Also on line 637, update the timeframe iteration:
```js
for (const key of ['1h', '4h']) {
```

- [ ] **Step 7: Verify no stray 15m UI references**

```bash
grep -rn '15m\|15min\|intraday_15m' templates/ static/js/
```

Expected: no signal-related hits (only CSS class names or unrelated occurrences are acceptable).

- [ ] **Step 8: Commit**

```bash
git add templates/index.html static/js/insights.js static/js/mtf.js static/js/signals.js
git commit -m "feat: remove 15m from UI — dropdowns, labels, MTF grid"
```

---

### Task 4: Update tests

**Files:**
- Modify: `tests/test_worker_telegram.py`
- Modify: `tests/test_trade_tracker.py`
- Modify: `tests/test_telegram_service.py`

- [ ] **Step 1: Update test_worker_telegram.py**

In `tests/test_worker_telegram.py`, the mock `intraday` dict currently has `"15m"` and `"1h"` keys. Update all mock signal dicts to use `"30m"` only:

Find (around line 80):
```python
"intraday": {"15m": mock_signal, "1h": mock_signal},
```
Change to:
```python
"intraday": {"30m": mock_signal},
```

Similarly around lines 120 and 163:
```python
"intraday": {"15m": bearish_signal, "1h": bearish_signal},
```
Change to:
```python
"intraday": {"30m": bearish_signal},
```

And around line 60-65, update the timeframe in the direct `_fire_telegram_signal` test:
```python
_fire_telegram_signal({"symbol": "HDFC", "timeframe": "intraday_30m", **_make_signal()})
```
```python
assert captured[0]["timeframe"] == "intraday_30m"
```

- [ ] **Step 2: Update test_trade_tracker.py**

In `tests/test_trade_tracker.py`, replace all `"intraday_15m"` with `"intraday_30m"`:

```bash
sed -i 's/intraday_15m/intraday_30m/g' tests/test_trade_tracker.py
```

- [ ] **Step 3: Update test_telegram_service.py**

In `tests/test_telegram_service.py` around line 15:
```python
"timeframe": "intraday_30m",
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_worker_telegram.py tests/test_trade_tracker.py tests/test_telegram_service.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_worker_telegram.py tests/test_trade_tracker.py tests/test_telegram_service.py
git commit -m "test: update 15m → 30m in signal tests"
```

---

### Task 5: Expand POPULAR_STOCKS with Nifty Midcap 100

**Files:**
- Modify: `app/config.py`

- [ ] **Step 1: Add missing Nifty Midcap 100 stocks**

In `app/config.py`, append a new section to `POPULAR_STOCKS` before the closing `]`:

```python
    # ── Additional Nifty Midcap 100 ──
    "IRCTC", "CGPOWER", "JIOFIN", "ANGELONE", "AUBANK",
    "BANDHANBNK", "ESCORTS", "LICI", "NAUKRI", "SUNDARMFIN",
    "TORNTPOWER", "TIINDIA", "ZYDUSLIFE", "INDHOTEL", "POLICYBZR",
    "SWIGGY", "DELHIVERY", "YESBANK", "TRIDENT", "RITES",
    "SOBHA", "RADICO", "WOCKPHARMA", "GILLETTE", "GLAXO",
    "PGHH", "KAJARIACER", "KANSAINER", "LINDEINDIA", "CONCOR",
```

- [ ] **Step 2: Verify no duplicates**

```bash
python3 -c "
from app.config import POPULAR_STOCKS
dupes = [s for s in POPULAR_STOCKS if POPULAR_STOCKS.count(s) > 1]
print('Dupes:', set(dupes) if dupes else 'None')
print('Total stocks:', len(POPULAR_STOCKS))
"
```

Expected: `Dupes: None`, total ~190 stocks.

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add 30 Nifty Midcap 100 stocks to 4h scan universe"
```

---

### Task 6: Full test suite + push

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass (or only pre-existing failures).

- [ ] **Step 2: Push to deploy**

```bash
git push origin main
```

- [ ] **Step 3: Verify on prod after deploy (~2 min)**

```bash
ssh -i ~/Downloads/LightsailDefaultKey-ap-south-1.pem ubuntu@65.2.125.8 \
  "cd /home/ubuntu/stock-tracker && python3 -c \"
from app.services.signal_service import signal_service
result = signal_service.get_multi_timeframe_signals('RELIANCE')
print('Keys:', list(result['intraday'].keys()))
print('Expected: [30m] only')
\""
```

Expected output: `Keys: ['30m']`
