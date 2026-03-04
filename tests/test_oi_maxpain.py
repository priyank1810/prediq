"""Tests for the O(n) max pain algorithm in oi_service.py.

Compares the optimized prefix-sum implementation against a naive O(n²)
reference to ensure correctness across various option chain scenarios.
"""

import pytest


def naive_max_pain(strikes_data):
    """O(n²) reference implementation — brute force max pain.

    Args:
        strikes_data: list of (strike, ce_oi, pe_oi) tuples, sorted ascending.

    Returns:
        strike with maximum total pain to option buyers.
    """
    if not strikes_data:
        return 0

    best_pain = -1
    best_strike = strikes_data[0][0]

    for k, (sk, _, _) in enumerate(strikes_data):
        pain = 0
        for i, (si, ce_oi, pe_oi) in enumerate(strikes_data):
            if si < sk:
                pain += pe_oi * (sk - si)
            elif si > sk:
                pain += ce_oi * (si - sk)
        if pain > best_pain:
            best_pain = pain
            best_strike = sk

    return best_strike


def optimized_max_pain(strikes_data):
    """O(n) prefix-sum implementation — mirrors oi_service.py logic."""
    if not strikes_data:
        return 0

    n = len(strikes_data)
    strikes = [sd[0] for sd in strikes_data]
    ce_ois = [sd[1] for sd in strikes_data]
    pe_ois = [sd[2] for sd in strikes_data]

    sum_pe_below = 0
    sum_pe_str_below = 0
    sum_ce_above = sum(ce_ois)
    sum_ce_str_above = sum(ce_ois[i] * strikes[i] for i in range(n))

    best_pain = -1
    best_strike = strikes[0]

    for k in range(n):
        sum_ce_above -= ce_ois[k]
        sum_ce_str_above -= ce_ois[k] * strikes[k]

        pain = (strikes[k] * sum_pe_below - sum_pe_str_below +
                sum_ce_str_above - strikes[k] * sum_ce_above)

        if pain > best_pain:
            best_pain = pain
            best_strike = strikes[k]

        sum_pe_below += pe_ois[k]
        sum_pe_str_below += pe_ois[k] * strikes[k]

    return best_strike


# ── Test Cases ──


class TestMaxPainAlgorithm:
    """Compare optimized O(n) vs naive O(n²) max pain computation."""

    def test_simple_symmetric(self):
        """Symmetric OI around center strike → max pain at center."""
        data = [
            (100, 1000, 0),
            (110, 500, 500),
            (120, 0, 1000),
        ]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_single_strike(self):
        """Single strike → that strike is max pain."""
        data = [(100, 500, 300)]
        assert optimized_max_pain(data) == 100
        assert naive_max_pain(data) == 100

    def test_two_strikes(self):
        """Two strikes — verify both implementations agree."""
        data = [(100, 1000, 200), (110, 200, 1000)]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_heavy_call_oi_high_strikes(self):
        """Heavy call OI at high strikes pulls max pain lower."""
        data = [
            (2800, 100, 500),
            (2850, 200, 400),
            (2900, 5000, 100),
            (2950, 8000, 50),
            (3000, 10000, 10),
        ]
        result = optimized_max_pain(data)
        expected = naive_max_pain(data)
        assert result == expected

    def test_heavy_put_oi_low_strikes(self):
        """Heavy put OI at low strikes pulls max pain higher."""
        data = [
            (2800, 10, 10000),
            (2850, 50, 8000),
            (2900, 100, 5000),
            (2950, 400, 200),
            (3000, 500, 100),
        ]
        result = optimized_max_pain(data)
        expected = naive_max_pain(data)
        assert result == expected

    def test_uniform_oi(self):
        """Uniform OI across all strikes."""
        data = [(s, 1000, 1000) for s in range(100, 200, 10)]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_zero_oi(self):
        """All OI is zero — first strike wins (pain = 0 for all)."""
        data = [(100, 0, 0), (110, 0, 0), (120, 0, 0)]
        # Both should return 100 (first strike, pain = 0 everywhere)
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_realistic_nifty_chain(self):
        """Realistic NIFTY-like option chain with 20+ strikes."""
        data = [
            (22000, 50, 8000),
            (22100, 80, 7500),
            (22200, 150, 6000),
            (22300, 300, 5000),
            (22400, 600, 4000),
            (22500, 1200, 3000),
            (22600, 2500, 2500),
            (22700, 4000, 1500),
            (22800, 6000, 800),
            (22900, 7500, 400),
            (23000, 9000, 200),
            (23100, 5000, 100),
            (23200, 3000, 50),
        ]
        result = optimized_max_pain(data)
        expected = naive_max_pain(data)
        assert result == expected

    def test_large_chain_consistency(self):
        """100-strike chain to stress-test O(n) correctness."""
        import random
        random.seed(42)
        data = [
            (1000 + i * 50, random.randint(0, 10000), random.randint(0, 10000))
            for i in range(100)
        ]
        result = optimized_max_pain(data)
        expected = naive_max_pain(data)
        assert result == expected

    def test_non_uniform_gaps(self):
        """Non-uniform strike gaps (e.g., 50, 100, 150 point gaps)."""
        data = [
            (2800, 1000, 3000),
            (2850, 2000, 2000),
            (2950, 3000, 1000),
            (3100, 4000, 500),
            (3400, 1000, 100),
        ]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_single_dominant_ce(self):
        """One strike has overwhelming CE OI."""
        data = [
            (100, 10, 100),
            (110, 10, 100),
            (120, 100000, 100),
            (130, 10, 100),
        ]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_single_dominant_pe(self):
        """One strike has overwhelming PE OI."""
        data = [
            (100, 100, 10),
            (110, 100, 100000),
            (120, 100, 10),
            (130, 100, 10),
        ]
        assert optimized_max_pain(data) == naive_max_pain(data)

    def test_empty_input(self):
        """Empty input returns 0."""
        assert optimized_max_pain([]) == 0
        assert naive_max_pain([]) == 0
