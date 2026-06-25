"""Solana meme bot entry point.

Structure only — no trading logic yet.

Safety: the bot runs in dry-run mode by default. Live execution requires the
explicit `--live` flag (see CLAUDE.md: "live execution requires an explicit
flag"). Before any live network commands, confirm devnet vs mainnet-beta.
"""

from __future__ import annotations

import argparse

import config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Dry-run defaults to ON. `--dry-run` is accepted for explicitness, while
    `--live` is the explicit opt-in required to disable dry-run.
    """
    parser = argparse.ArgumentParser(description="Solana meme bot")

    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Simulate without broadcasting transactions (default: ON).",
    )
    parser.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Disable dry-run and execute live transactions (explicit opt-in).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[startup] mode={mode} network={config.SOLANA_NETWORK}")

    if not args.dry_run:
        # Live execution path intentionally left unimplemented for now.
        print("[startup] LIVE mode requested — trading logic not implemented yet.")

    # TODO: wire up strategy modules from strategies/ once trading logic exists.


if __name__ == "__main__":
    main()
