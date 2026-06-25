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
