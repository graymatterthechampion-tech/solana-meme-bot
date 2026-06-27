"""Tests for the dry-run fill simulator (execution.fill_simulator).

Pure function: verifies constant-product price impact, fee/drift deductions,
the slippage-cap flag vs. actual rejection, and the no-op / illiquid edges.
"""

from __future__ import annotations

import pytest

from execution.fill_simulator import FillResult, simulate_fill


def test_tiny_sell_into_deep_pool_has_negligible_impact() -> None:
    r = simulate_fill(1.0, 1.0, 1_000_000.0)
    assert isinstance(r, FillResult)
    assert r.filled is True
    assert r.exceeded_slippage_cap is False
    assert r.price_impact_pct < 1e-4           # deep pool -> almost no impact
    assert r.total_slippage_pct < 0.01         # under the 1% cap
    assert r.net_proceeds_usd < r.notional_usd  # fees/drift still bite
    assert r.effective_price < r.quoted_price


def test_price_impact_matches_constant_product_model() -> None:
    # S=500 into L=100_000 -> f=2S/L=0.01, impact=f/(1+f)=0.009900...
    r = simulate_fill(500.0, 1.0, 100_000.0)
    assert r.notional_usd == pytest.approx(500.0)
    assert r.price_impact_pct == pytest.approx(0.01 / 1.01, rel=1e-6)
    # total slippage = 1 - (1-impact)(1-drift); drift default 0.001
    expected_slip = 1.0 - (1.0 - 0.01 / 1.01) * (1.0 - 0.001)
    assert r.total_slippage_pct == pytest.approx(expected_slip, rel=1e-6)
    # ~1.09% slippage exceeds the 1% cap flag, but default policy still fills.
    assert r.exceeded_slippage_cap is True
    assert r.filled is True
    # net = gross - swap_fee - network_fee
    gross = 500.0 * (1.0 - r.total_slippage_pct)
    expected_net = gross - gross * 0.0025 - 0.05
    assert r.net_proceeds_usd == pytest.approx(expected_net, rel=1e-6)


def test_reject_over_cap_blocks_the_fill() -> None:
    r = simulate_fill(500.0, 1.0, 100_000.0, reject_over_cap=True)
    assert r.exceeded_slippage_cap is True
    assert r.filled is False
    assert r.net_proceeds_usd == 0.0
    assert r.effective_price == 0.0


def test_within_cap_fills_under_reject_policy() -> None:
    # Small sell into a deep pool stays under the cap -> fills even when strict.
    r = simulate_fill(10.0, 1.0, 5_000_000.0, reject_over_cap=True)
    assert r.exceeded_slippage_cap is False
    assert r.filled is True
    assert r.net_proceeds_usd > 0.0


def test_bigger_sell_has_more_slippage() -> None:
    small = simulate_fill(100.0, 1.0, 100_000.0)
    big = simulate_fill(2_000.0, 1.0, 100_000.0)
    assert big.total_slippage_pct > small.total_slippage_pct
    assert big.price_impact_pct > small.price_impact_pct


def test_zero_tokens_is_noop() -> None:
    r = simulate_fill(0.0, 1.0, 100_000.0)
    assert r.filled is False
    assert r.net_proceeds_usd == 0.0
    assert r.notional_usd == 0.0


def test_zero_liquidity_is_fully_illiquid() -> None:
    r = simulate_fill(100.0, 1.0, 0.0, reject_over_cap=True)
    assert r.price_impact_pct == pytest.approx(1.0)
    assert r.exceeded_slippage_cap is True
    assert r.filled is False
