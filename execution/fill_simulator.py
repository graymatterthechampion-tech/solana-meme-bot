"""Dry-run fill simulation: model the friction a real swap would incur.

The strategy modules decide *how many tokens* to sell; on their own they assume
a frictionless fill at the quoted price, which makes dry-run PnL fantasy. This
module turns an intended sell into a realistic :class:`FillResult` by modelling:

    * PRICE IMPACT  — selling into a constant-product (x*y=k) pool moves the
      price against you. For a sell whose notional is ``S`` USD into a pool of
      ``L`` USD total liquidity (≈ ``L/2`` per side), the fraction of the
      token-side reserve sold is ``2S/L`` and the slippage is ``f/(1+f)`` with
      ``f = 2S/L``. Bigger sells relative to liquidity cost more — exactly why
      CLAUDE.md sizes positions as a small fraction of the pool.
    * FILL DELAY    — price can drift adversely between decision and on-chain
      fill; modelled as a small fixed adverse move.
    * SWAP FEE      — the AMM/LP + route fee, a percentage of proceeds.
    * NETWORK FEE   — base + priority fee for landing the transaction, a flat
      per-fill USD cost.
    * SLIPPAGE CAP  — CLAUDE.md mandates a strict cap; the result flags whether
      total slippage exceeded it. ``reject_over_cap`` makes the (simulated) fill
      fail when it does, mirroring a live swap that would be rejected.

This module is pure and side-effect free: no network, no signing, no broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Defaults (tunable) -----------------------------------------------------
DEFAULT_SWAP_FEE_PCT: float = 0.0025          # 0.25% AMM/LP + route fee
DEFAULT_NETWORK_FEE_USD: float = 0.05         # base + priority fee per fill
DEFAULT_FILL_DELAY_ADVERSE_PCT: float = 0.001  # 0.1% adverse drift before fill
DEFAULT_MAX_SLIPPAGE_PCT: float = 0.01        # CLAUDE.md strict cap (1%)


@dataclass(frozen=True)
class FillResult:
    """The modelled outcome of executing one sell, in USD terms.

    ``filled`` is False only when ``reject_over_cap`` is set and the modelled
    total slippage exceeded ``max_slippage_pct`` — in which case the sell would
    not land and ``net_proceeds_usd`` is 0.
    """

    requested_tokens: float
    quoted_price: float
    notional_usd: float
    price_impact_pct: float
    fill_delay_adverse_pct: float
    total_slippage_pct: float
    swap_fee_usd: float
    network_fee_usd: float
    gross_proceeds_usd: float
    net_proceeds_usd: float
    effective_price: float
    exceeded_slippage_cap: bool
    filled: bool


def _empty_result(tokens: float, quoted_price: float, cap: float) -> FillResult:
    """A no-op fill (nothing to sell / bad inputs): zero proceeds, not filled."""
    return FillResult(
        requested_tokens=max(0.0, tokens),
        quoted_price=max(0.0, quoted_price),
        notional_usd=0.0,
        price_impact_pct=0.0,
        fill_delay_adverse_pct=0.0,
        total_slippage_pct=0.0,
        swap_fee_usd=0.0,
        network_fee_usd=0.0,
        gross_proceeds_usd=0.0,
        net_proceeds_usd=0.0,
        effective_price=0.0,
        exceeded_slippage_cap=False,
        filled=False,
    )


def simulate_fill(
    tokens: float,
    quoted_price: float,
    pool_liquidity_usd: float,
    *,
    swap_fee_pct: float = DEFAULT_SWAP_FEE_PCT,
    network_fee_usd: float = DEFAULT_NETWORK_FEE_USD,
    fill_delay_adverse_pct: float = DEFAULT_FILL_DELAY_ADVERSE_PCT,
    max_slippage_pct: float = DEFAULT_MAX_SLIPPAGE_PCT,
    reject_over_cap: bool = False,
) -> FillResult:
    """Model the realistic proceeds of selling ``tokens`` at ``quoted_price``.

    ``pool_liquidity_usd`` is the current pool liquidity (USD); deeper pools
    incur less price impact. Returns a :class:`FillResult`. With
    ``reject_over_cap=True`` a fill whose modelled slippage exceeds
    ``max_slippage_pct`` is reported as not filled (zero proceeds), mirroring a
    live swap rejected by its slippage guard.

    Pure: no network, no signing, no broadcast.
    """
    if tokens <= 0 or quoted_price <= 0:
        return _empty_result(tokens, quoted_price, max_slippage_pct)

    notional = tokens * quoted_price

    # Constant-product price impact: f = 2S/L, slippage = f/(1+f). A non-positive
    # pool is treated as fully illiquid (100% impact) -> always over cap.
    if pool_liquidity_usd <= 0:
        price_impact_pct = 1.0
    else:
        f = (2.0 * notional) / pool_liquidity_usd
        price_impact_pct = f / (1.0 + f)

    drift = max(0.0, fill_delay_adverse_pct)
    # Received fraction compounds impact and adverse drift multiplicatively.
    received_fraction = (1.0 - price_impact_pct) * (1.0 - drift)
    received_fraction = max(0.0, received_fraction)
    total_slippage_pct = 1.0 - received_fraction

    gross_proceeds = notional * received_fraction
    swap_fee = gross_proceeds * max(0.0, swap_fee_pct)
    net_proceeds = max(0.0, gross_proceeds - swap_fee - max(0.0, network_fee_usd))

    exceeded = total_slippage_pct > max_slippage_pct
    filled = not (reject_over_cap and exceeded)
    if not filled:
        net_proceeds = 0.0

    effective_price = net_proceeds / tokens if (filled and tokens > 0) else 0.0

    return FillResult(
        requested_tokens=tokens,
        quoted_price=quoted_price,
        notional_usd=notional,
        price_impact_pct=price_impact_pct,
        fill_delay_adverse_pct=drift,
        total_slippage_pct=total_slippage_pct,
        swap_fee_usd=swap_fee,
        network_fee_usd=max(0.0, network_fee_usd) if filled else 0.0,
        gross_proceeds_usd=gross_proceeds if filled else 0.0,
        net_proceeds_usd=net_proceeds,
        effective_price=effective_price,
        exceeded_slippage_cap=exceeded,
        filled=filled,
    )
