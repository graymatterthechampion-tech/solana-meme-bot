"""Read-only candidate-discovery layer.

Modules here ONLY discover and surface candidate token mint addresses to feed
into the safety -> entry pipeline. They never buy, sell, sign, or broadcast
anything, and they import no signer, keypair, or transaction-building code. See
``token_scanner`` for the Dexscreener-backed scanner.
"""
