"""Configuration loader.

Reads all sensitive credentials from a local `.env` file via python-dotenv.
NEVER hardcode private keys, seed phrases, or RPC secret URLs here — keep
them in `.env`, which is excluded from version control.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file into the process environment.
load_dotenv()


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, returning `default` if unset."""
    return os.environ.get(name, default)


def _require(name: str) -> str:
    """Read a required environment variable or raise a clear error."""
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Missing required environment variable: {name!r}. "
            f"Add it to your local .env file."
        )
    return value


def _get_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on unset/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on unset/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Network -----------------------------------------------------------------
# One of: "devnet" | "mainnet-beta". Defaults to the safe network.
SOLANA_NETWORK: str = _get("SOLANA_NETWORK", "devnet") or "devnet"

# RPC endpoint (e.g. a Helius / QuickNode URL). Kept secret — read from env.
RPC_URL: Optional[str] = _get("RPC_URL")

# --- Wallet ------------------------------------------------------------------
# Path to the keypair JSON file, or the secret itself — never commit either.
WALLET_KEYPAIR_PATH: Optional[str] = _get("WALLET_KEYPAIR_PATH")
WALLET_PRIVATE_KEY: Optional[str] = _get("WALLET_PRIVATE_KEY")

# --- APIs --------------------------------------------------------------------
JUPITER_API_URL: str = _get(
    "JUPITER_API_URL", "https://quote-api.jup.ag/v6"
) or "https://quote-api.jup.ag/v6"

# Market-data provider credentials (optional until the real feeds are wired).
# Never hardcode — read from .env. Dexscreener needs no key.
BIRDEYE_API_KEY: Optional[str] = _get("BIRDEYE_API_KEY")
HELIUS_API_KEY: Optional[str] = _get("HELIUS_API_KEY")

# --- Curated KOL / smart-money wallets (optional) ----------------------------
# Comma-separated Solana wallet addresses read from .env, used ONLY as a
# read-only meta signal (does a tracked wallet hold a token). Never a trigger.
# Empty when unset. Never hardcode real addresses here — keep them in .env.
KOL_WALLETS: list[str] = [
    w.strip() for w in (_get("KOL_WALLETS", "") or "").split(",") if w.strip()
]

# --- Trade journal -----------------------------------------------------------
# Append-only local file where each completed (dry-run) trade is journalled so
# performance can be analysed across runs (see reporting.trade_journal /
# reporting.journal_summary). It holds only simulated PnL and market metrics —
# never keys or secrets — so it is safe on disk, but it is runtime output and is
# git-ignored. JSON Lines (one JSON object per line) so appends never rewrite
# the file. Override the location with the TRADE_JOURNAL_PATH env var.
TRADE_JOURNAL_PATH: str = _get("TRADE_JOURNAL_PATH", "trades.jsonl") or "trades.jsonl"

# --- Re-entry guard (churn protection) ---------------------------------------
# Prevents the bot from re-buying the same mint over and over (the observed
# ACM x7 churn). See safety.reentry.ReentryGuard. All read-only / dry-run safe.
#
#   COOLDOWN     : after ENTERING a mint, refuse to re-enter it for this many
#                  seconds regardless of outcome (default 1h). Stops rapid churn.
#   BLACKLIST_HARD_EXITS : after this many defensive hard-exits on a mint, it is
#                  blacklisted (a hard exit means it flash-crashed / rugged /
#                  volume-collapsed — a clear "avoid" signal). Default 1.
#   BLACKLIST_SECONDS    : how long a blacklist lasts (default 24h). Set to 0 (or
#                  negative) to blacklist for the whole process (permanent).
REENTRY_COOLDOWN_SECONDS: float = _get_float("REENTRY_COOLDOWN_SECONDS", 3600.0)
REENTRY_BLACKLIST_HARD_EXITS: int = _get_int("REENTRY_BLACKLIST_HARD_EXITS", 1)
REENTRY_BLACKLIST_SECONDS: float = _get_float("REENTRY_BLACKLIST_SECONDS", 86_400.0)
