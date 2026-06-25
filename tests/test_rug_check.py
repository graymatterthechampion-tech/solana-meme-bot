"""Tests for the read-only token safety module.

Covers: a clean token passes; each individual failure condition fails; and a
fail-closed case where the data source errors. A read-only async stub gatherer
is injected so no network is touched.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from safety.rug_check import evaluate_token_safety


def safe_payload() -> Dict[str, Any]:
    """A clean token: every check passes."""
    return {
        "lp_locked_or_burned": True,
        "lp_lock_detail": "LP burned (100%)",
        "buy_tax_pct": 0.0,
        "sell_tax_pct": 0.0,
        "holders": [
            {"address": "Whale1", "pct": 3.0, "funded_by": "srcA"},
            {"address": "Whale2", "pct": 2.5, "funded_by": "srcB"},
            {"address": "Whale3", "pct": 2.0, "funded_by": "srcC"},
            {"address": "Whale4", "pct": 1.5, "funded_by": "srcD"},
            {"address": "Whale5", "pct": 1.0, "funded_by": "srcE"},
        ],
        "volume_24h_usd": 40_000.0,
        "market_cap_usd": 500_000.0,
        "price_change_24h_pct": 22.0,
    }


def gatherer_for(payload: Dict[str, Any]):
    async def _gather(mint_address: str) -> Dict[str, Any]:
        return payload

    return _gather


def evaluate(payload: Dict[str, Any]):
    return asyncio.run(
        evaluate_token_safety("MINT", data_gatherer=gatherer_for(payload))
    )


def test_clean_token_passes() -> None:
    report = evaluate(safe_payload())
    assert report.passed is True
    assert report.reasons == []
    assert report.lp_locked_or_burned is True
    assert report.holder_concentration_pass is True
    assert report.funding_source_clustered is False
    assert report.farmed_volume_flag is False
    assert report.tax_pct == pytest.approx(0.0)
    assert report.top10_holder_pct == pytest.approx(10.0)
    assert report.volume_to_mcap_ratio == pytest.approx(0.08)


def test_unlocked_lp_fails() -> None:
    payload = safe_payload()
    payload["lp_locked_or_burned"] = False
    payload["lp_lock_detail"] = "LP unlocked, dev holds 100%"
    report = evaluate(payload)
    assert report.passed is False
    assert report.lp_locked_or_burned is False
    assert any("LP not locked" in r for r in report.reasons)


def test_high_tax_fails() -> None:
    payload = safe_payload()
    payload["sell_tax_pct"] = 10.0  # >= 1%
    report = evaluate(payload)
    assert report.passed is False
    assert report.tax_pct == pytest.approx(10.0)
    assert any("tax too high" in r for r in report.reasons)


def test_holder_concentration_fails() -> None:
    payload = safe_payload()
    # Seven distinct whales at 3% each = 21% (>= 15%), distinct funders.
    payload["holders"] = [
        {"address": f"W{i}", "pct": 3.0, "funded_by": f"src{i}"} for i in range(7)
    ]
    report = evaluate(payload)
    assert report.passed is False
    assert report.holder_concentration_pass is False
    assert report.top10_holder_pct == pytest.approx(21.0)
    assert report.funding_source_clustered is False  # distinct funders


def test_funding_clustering_fails() -> None:
    payload = safe_payload()
    # Low concentration (10% < 15%) but all funded from one wallet.
    payload["holders"] = [
        {"address": f"W{i}", "pct": 2.0, "funded_by": "sameWallet"} for i in range(5)
    ]
    report = evaluate(payload)
    assert report.passed is False
    assert report.holder_concentration_pass is True  # concentration is fine
    assert report.funding_source_clustered is True
    assert any("same source" in r for r in report.reasons)


def test_farmed_volume_fails() -> None:
    payload = safe_payload()
    payload["volume_24h_usd"] = 1_000_000.0  # ratio 2.0 vs 500k mcap
    payload["price_change_24h_pct"] = 1.0     # flat price
    report = evaluate(payload)
    assert report.passed is False
    assert report.farmed_volume_flag is True
    assert report.volume_to_mcap_ratio == pytest.approx(2.0)
    assert any("farmed volume" in r for r in report.reasons)


def test_missing_field_marks_only_that_check_failed() -> None:
    """A missing field fails its own check (fail-closed), not the whole eval."""
    payload = safe_payload()
    del payload["sell_tax_pct"]  # tax data incomplete
    report = evaluate(payload)
    assert report.passed is False
    assert any("tax check failed" in r for r in report.reasons)
    # Other checks still evaluated normally.
    assert report.lp_locked_or_burned is True
    assert report.holder_concentration_pass is True


def test_fail_closed_when_data_source_errors() -> None:
    """If the data source raises, the report is fully UNSAFE (never safe)."""

    async def boom(mint_address: str) -> Dict[str, Any]:
        raise RuntimeError("rpc down")

    report = asyncio.run(evaluate_token_safety("MINT", data_gatherer=boom))
    assert report.passed is False
    assert report.lp_locked_or_burned is False
    assert report.holder_concentration_pass is False
    assert report.funding_source_clustered is True
    assert report.farmed_volume_flag is True
    assert any("unavailable" in r for r in report.reasons)
