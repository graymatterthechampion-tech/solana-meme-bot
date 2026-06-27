"""Execution layer.

Currently DRY-RUN ONLY: :mod:`execution.fill_simulator` models the market
friction a real swap would incur (price impact, swap/priority fees, fill-delay
drift) so simulated PnL reflects reality. Nothing here signs or broadcasts a
transaction — live execution is a separate, explicit-flag concern not yet built.
"""
