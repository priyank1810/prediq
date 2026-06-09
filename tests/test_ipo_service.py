"""Tests for ipo_service — all network mocked."""
from datetime import date, timedelta
from unittest.mock import patch

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


@patch.object(ipo_service, "_fetch_gmp_map", return_value={"ACME": 30.0})
@patch.object(ipo_service, "fetch_upcoming", return_value=SAMPLE_UPCOMING)
@patch("app.services.ipo_service.fundamental_service")
def test_open_ipo_gets_live_unfrozen_verdict(mock_fund, _f1, _f2, db):
    mock_fund.get_fundamentals.return_value = {"pe": 20, "rev_growth": 0.25, "de": 0.4}
    ipo_service.refresh_and_score(db)
    rec = db.query(IPORecommendation).one()
    assert rec.frozen is False           # still open → mutable
    assert rec.verdict in ("APPLY", "NEUTRAL", "AVOID")
    # serialized upcoming exposes the live verdict
    assert ipo_service.get_upcoming(db)[0]["verdict"] == rec.verdict


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
    db.add(ipo)
    db.commit()
    with patch("app.services.ipo_service.yahoo_quote", return_value={"ltp": 130.0}):
        n = ipo_service.backfill_listing_perf(db)
    db.refresh(ipo)
    assert n == 1
    assert ipo.listing_price == 130.0
    assert round(ipo.listing_gain_pct, 1) == 30.0   # vs price_band_high 100


def test_scorecard_hit_miss(db):
    ipo = IPO(name="Score Ltd", symbol="SCOR", price_band_high=100.0,
              listing_price=120.0, listing_gain_pct=20.0,
              listing_date=date.today() - timedelta(days=2))
    db.add(ipo)
    db.commit()
    db.add(IPORecommendation(ipo_id=ipo.id, verdict="APPLY", confidence=80,
                             final_score=70.0))
    db.commit()
    card = ipo_service.get_scorecard(db)
    assert card["total_graded"] == 1
    assert card["hits"] == 1
    assert card["accuracy_pct"] == 100.0
