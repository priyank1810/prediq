"""Upcoming-IPO ingestion, scoring, persistence, and scorecard."""
import json
import logging
import re
from datetime import date, datetime

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
                    "subscription_total": None,
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("skip malformed IPO row: %s", e)
        return [r for r in out if r["name"]]

    def _fetch_gmp_map(self) -> dict:
        """Scrape one GMP source -> {NAME_UPPER: gmp_pct}. {} on failure."""
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
        """Best-effort parse of the GMP source into {NAME_UPPER: gmp_pct}.

        The exact selectors depend on IPO_GMP_SOURCE_URL; parse rows of
        (company name, estimated listing gain %) and key by uppercased name.
        Returns {} if the structure is not recognized.
        """
        try:
            out = {}
            pattern = r">([A-Za-z][A-Za-z0-9 &.\-]{2,60})<[^%]*?([\-]?\d{1,3}(?:\.\d+)?)%"
            for m in re.finditer(pattern, html):
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
            price = (q or {}).get("ltp")
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
