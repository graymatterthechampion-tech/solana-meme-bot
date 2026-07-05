"""Tests for the real RugCheck + Helius wiring in safety/rug_check.

Uses httpx.MockTransport to simulate the live endpoints (no network) with a
fixture shaped like a real RugCheck v1 report. Covers rugged/risk/LP/tax/
holder/mcap parsing, the knownAccounts pool-exclusion fix, rate-limit retry,
timeouts, and per-provider fail-closed behavior.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

import config
from safety import rug_check
from safety.rug_check import SYSTEM_ADDRESS, evaluate_token_safety

HELIUS_TEST_URL = "https://helius.test/rpc"


# --- RugCheck fixture (shaped like the real v1 report) -----------------------

def default_top_holders() -> List[Dict[str, Any]]:
    # A pool (AMM) and the system address dominate by raw pct but must be
    # excluded; the five real holders sum to 10%.
    return [
        {"owner": "POOLOWNER1", "pct": 60.0},
        {"owner": SYSTEM_ADDRESS, "pct": 5.0},
        *[{"owner": f"Hold{i}", "pct": 2.0} for i in range(5)],
    ]


def default_known_accounts() -> Dict[str, Any]:
    return {"POOLOWNER1": {"name": "Pump Fun AMM", "type": "AMM"}}


def rugcheck_report(
    *,
    rugged: bool = False,
    score: float = 10.0,
    risks: Optional[List[Dict[str, Any]]] = None,
    lp_locked_pct: float = 100.0,
    tax_pct: float = 0.0,
    top_holders: Any = "default",
    known_accounts: Optional[Dict[str, Any]] = None,
    price: float = 0.0005,
    supply: int = 10 ** 15,
    decimals: int = 6,
    include_top_holders: bool = True,
) -> Dict[str, Any]:
    # NOTE: volume / price change are NOT in the RugCheck report — they come
    # from Dexscreener now (see dexscreener_payload).
    report: Dict[str, Any] = {
        "tokenProgram": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
        "rugged": rugged,
        "score_normalised": score,
        "risks": [] if risks is None else risks,
        "token": {
            "supply": supply,
            "decimals": decimals,
            "mintAuthority": None,
            "freezeAuthority": None,
        },
        "token_extensions": {"transferFeeConfig": None},
        "transferFee": {"pct": tax_pct},
        "price": price,
        "markets": [{"lp": {"lpLockedPct": lp_locked_pct, "lpLockedUSD": 25_000.0}}],
    }
    if include_top_holders:
        report["topHolders"] = (
            default_top_holders() if top_holders == "default" else top_holders
        )
        report["knownAccounts"] = (
            default_known_accounts() if known_accounts is None else known_accounts
        )
    return report


# --- Dexscreener fixture (volume / price change) -----------------------------

def dexscreener_payload(
    *,
    volume: float = 40_000.0,
    price_change: float = 22.0,
    omit_volume: bool = False,
    no_pairs: bool = False,
) -> Dict[str, Any]:
    if no_pairs:
        return {"pairs": []}
    pair: Dict[str, Any] = {"volume": {}, "priceChange": {"h24": price_change}}
    if not omit_volume:
        pair["volume"]["h24"] = volume
    return {"pairs": [pair]}


# --- Helius (fallback) mock helpers ------------------------------------------

def helius_largest(accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "result": {"value": accounts}}


def helius_supply(ui_amount: float) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "result": {"value": {"uiAmount": ui_amount}}}


def default_accounts() -> List[Dict[str, Any]]:
    return [{"address": f"acct{i}", "uiAmount": 20_000.0} for i in range(5)]


# --- Mock transport ----------------------------------------------------------

def make_handler(
    *,
    rug_json: Optional[Dict[str, Any]] = None,
    accounts: Optional[List[Dict[str, Any]]] = None,
    supply_ui: float = 1_000_000.0,
    rug_status_sequence: Optional[List[int]] = None,
    rug_raises: Optional[Exception] = None,
    helius_error: bool = False,
    dex_json: Optional[Dict[str, Any]] = None,
    dex_raises: Optional[Exception] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    rug_json = rugcheck_report() if rug_json is None else rug_json
    accounts = default_accounts() if accounts is None else accounts
    dex_json = dexscreener_payload() if dex_json is None else dex_json
    state = {"rug_call": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)

        if "dexscreener.com" in url:
            if dex_raises is not None:
                raise dex_raises
            return httpx.Response(200, json=dex_json)

        if "rugcheck.xyz" in url:
            # Assert RugCheck is called WITHOUT an auth header.
            assert "authorization" not in {k.lower() for k in request.headers}
            if rug_raises is not None:
                raise rug_raises
            if rug_status_sequence:
                idx = min(state["rug_call"], len(rug_status_sequence) - 1)
                status = rug_status_sequence[idx]
                state["rug_call"] += 1
                if status == 429:
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={})
                if status >= 500:
                    return httpx.Response(status, json={})
            return httpx.Response(200, json=rug_json)

        if "helius.test" in url:
            body = json.loads(request.content)
            method = body.get("method")
            if helius_error:
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": 1, "error": {"message": "boom"}}
                )
            if method == "getTokenLargestAccounts":
                return httpx.Response(200, json=helius_largest(accounts))
            if method == "getTokenSupply":
                return httpx.Response(200, json=helius_supply(supply_ui))

        return httpx.Response(404, json={})

    return handler


@pytest.fixture(autouse=True)
def _fast_and_configured(monkeypatch):
    monkeypatch.setattr(rug_check, "RATE_LIMIT_BACKOFF", 0.0)
    monkeypatch.setattr(config, "RPC_URL", HELIUS_TEST_URL)


def evaluate(handler) -> rug_check.SafetyReport:
    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await evaluate_token_safety("MINT", client=client)
        finally:
            await client.aclose()

    return asyncio.run(_run())


# --- Tests -------------------------------------------------------------------

def test_clean_token_passes_end_to_end_with_dexscreener_volume() -> None:
    # Volume now comes from Dexscreener; clean token passes EVERY check.
    report = evaluate(make_handler())
    assert report.passed is True
    assert report.reasons == []
    assert report.lp_locked_or_burned is True
    assert report.tax_pct == pytest.approx(0.0)
    # Pool (AMM) and system address excluded -> only the 5 real holders (10%).
    assert report.top10_holder_pct == pytest.approx(10.0)
    # mcap = 1e9 ui-supply * 0.0005 = 500k; Dexscreener volume 40k -> 0.08.
    assert report.volume_to_mcap_ratio == pytest.approx(0.08)
    assert report.farmed_volume_flag is False


def test_dexscreener_failure_fails_farmed_check_closed() -> None:
    # Dexscreener errors -> no volume -> farmed-volume check fails closed.
    report = evaluate(make_handler(dex_raises=httpx.TimeoutException("timed out")))
    assert report.passed is False
    assert report.farmed_volume_flag is True
    assert any("volume check failed" in r for r in report.reasons)


def test_dexscreener_omits_volume_fails_closed() -> None:
    # Dexscreener responds but without an h24 volume -> farmed fails closed.
    report = evaluate(make_handler(dex_json=dexscreener_payload(omit_volume=True)))
    assert report.passed is False
    assert report.farmed_volume_flag is True
    assert any("volume check failed" in r for r in report.reasons)


def test_farmed_volume_detected_high_vol_flat_price() -> None:
    # High volume vs mcap (ratio >= 1) with a flat price -> farmed flag fires.
    dex = dexscreener_payload(volume=1_000_000.0, price_change=1.0)
    report = evaluate(make_handler(dex_json=dex))
    assert report.passed is False
    assert report.farmed_volume_flag is True
    assert report.volume_to_mcap_ratio == pytest.approx(2.0)
    assert any("farmed volume" in r for r in report.reasons)


def test_rugged_flag_fails_immediately() -> None:
    report = evaluate(make_handler(rug_json=rugcheck_report(rugged=True)))
    assert report.passed is False
    assert any("RUGGED" in r for r in report.reasons)


def test_high_risk_score_fails() -> None:
    report = evaluate(make_handler(rug_json=rugcheck_report(score=85.0)))
    assert report.passed is False
    assert any("risk score" in r for r in report.reasons)


def test_high_severity_risk_flag_fails() -> None:
    risks = [{"name": "Mint Authority still enabled", "level": "danger"}]
    report = evaluate(make_handler(rug_json=rugcheck_report(risks=risks)))
    assert report.passed is False
    assert any("high-severity risks" in r for r in report.reasons)


def test_warn_level_risk_does_not_fail() -> None:
    # A non-high-severity ("warn") risk should not, by itself, fail the token.
    risks = [{"name": "Low liquidity", "level": "warn"}]
    report = evaluate(make_handler(rug_json=rugcheck_report(risks=risks)))
    assert report.passed is True


def test_unlocked_lp_fails() -> None:
    report = evaluate(make_handler(rug_json=rugcheck_report(lp_locked_pct=0.0)))
    assert report.passed is False
    assert report.lp_locked_or_burned is False
    assert any("LP not locked" in r for r in report.reasons)


def test_high_tax_fails() -> None:
    report = evaluate(make_handler(rug_json=rugcheck_report(tax_pct=5.0)))
    assert report.passed is False
    assert report.tax_pct == pytest.approx(5.0)
    assert any("tax too high" in r for r in report.reasons)


def test_amm_pool_excluded_from_concentration() -> None:
    # Pool owns 80% but is AMM-labeled; real holders are 10%. Must PASS, proving
    # the pool is not counted as a whale.
    holders = [
        {"owner": "POOL", "pct": 80.0},
        *[{"owner": f"H{i}", "pct": 2.0} for i in range(5)],
    ]
    known = {"POOL": {"name": "Raydium Pool", "type": "AMM"}}
    report = evaluate(
        make_handler(rug_json=rugcheck_report(top_holders=holders, known_accounts=known))
    )
    assert report.top10_holder_pct == pytest.approx(10.0)
    assert report.holder_concentration_pass is True
    assert report.passed is True


def test_real_whale_concentration_fails() -> None:
    # A non-pool whale at 92% (>= 30% threshold) must fail concentration.
    holders = [{"owner": "Whale", "pct": 92.0}, {"owner": "H1", "pct": 1.0}]
    report = evaluate(
        make_handler(rug_json=rugcheck_report(top_holders=holders, known_accounts={}))
    )
    assert report.passed is False
    assert report.top10_holder_pct == pytest.approx(93.0)
    assert report.holder_concentration_pass is False


def test_rate_limit_retries_then_succeeds() -> None:
    report = evaluate(make_handler(rug_status_sequence=[429, 200]))
    assert report.passed is True
    assert report.lp_locked_or_burned is True


def test_rate_limit_exhausted_fails_closed() -> None:
    report = evaluate(make_handler(rug_status_sequence=[429]))
    assert report.passed is False
    assert report.lp_locked_or_burned is False
    assert any("rugged check failed" in r for r in report.reasons)


def test_rugcheck_timeout_fails_closed() -> None:
    report = evaluate(make_handler(rug_raises=httpx.TimeoutException("timed out")))
    assert report.passed is False
    assert report.lp_locked_or_burned is False


def test_helius_fallback_used_when_rugcheck_omits_topholders() -> None:
    # RugCheck report has no topHolders -> holders come from Helius fallback.
    report = evaluate(
        make_handler(rug_json=rugcheck_report(include_top_holders=False))
    )
    assert report.holder_concentration_pass is True  # 5 * 2% = 10% via Helius
    assert report.passed is True
