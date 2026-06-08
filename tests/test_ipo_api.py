"""Tests for IPO endpoints: /api/ipo."""
from datetime import date, timedelta

from app.models import IPO, IPORecommendation


def _seed(db):
    ipo = IPO(name="Acme Tech Ltd", symbol="ACME",
              open_date=date.today(), close_date=date.today() + timedelta(days=2),
              price_band_low=100.0, price_band_high=110.0, lot_size=135, gmp=25.0,
              subscription_total=5.0, is_fresh=True)
    db.add(ipo)
    db.commit()
    db.add(IPORecommendation(ipo_id=ipo.id, verdict="APPLY", confidence=80,
                             final_score=72.0, reasons='["good"]', risk_flags='[]',
                             score_gmp=90.0, score_subscription=80.0,
                             score_fundamentals=60.0))
    db.commit()
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
