"""Unit tests for the pure IPO scoring engine."""
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
