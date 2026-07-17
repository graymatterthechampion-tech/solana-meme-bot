"""Trade reporting.

Turns the raw per-tick loop outcomes of a managed position into a single,
clean, human-readable trade report answering the two questions the cumulative
INFO stream buries: *why did the position exit* and *what was the per-trade
PnL*. Pure and read-only — it derives everything from an already-completed
:class:`main.TradeSession`; it never trades, signs, or broadcasts.
"""
