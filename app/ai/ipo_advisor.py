"""Pure, explainable IPO Apply/Neutral/Avoid scoring engine. No I/O."""
from app.config import IPO_WEIGHTS, IPO_VERDICT_THRESHOLDS


def _gmp_subscore(gmp_pct):
    """Map estimated GMP listing-gain % to 0-100. ~30%+ saturates high."""
    if gmp_pct is None:
        return None
    return max(0.0, min(100.0, 50.0 + gmp_pct * (50.0 / 30.0)))


def _subscription_subscore(mult):
    """Map total oversubscription multiple to 0-100. 1x ~ 59, 10x ~ 100."""
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
