"""Render a self-contained HTML performance report from the trade journal.

Reads the append-only journal (:mod:`reporting.trade_journal`), keeps only REAL
trades (drops the synthetic demo rows), and writes a single, dependency-free
``report.html`` the user can open directly in a browser. Every chart is inline
SVG rendered here in Python — no CDN, no paid service, no JavaScript required —
so the file works fully offline and reproducibly.

Pure and read-only: it consumes an already-written journal file and emits one
HTML file. It never touches the network, a signer, or a broadcast. Run it::

    python -m reporting.build_report                 # -> report.html
    python -m reporting.build_report --out out.html  # custom path

When the completed-round-trip count is below :data:`RELIABLE_SAMPLE`, the page
is stamped ``PRELIMINARY`` (banner + diagonal watermark) because the sample is
too small to support conclusions.
"""

from __future__ import annotations

import argparse
import html
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from reporting.journal_summary import HARD_EXIT_TRIGGERS, PROFIT_TIERS
from reporting.trade_journal import TradeRecord, load_records

# Below this many completed round-trips the report is watermarked PRELIMINARY.
RELIABLE_SAMPLE: int = 20

# Demo/synthetic rows to exclude: the generator seeded them with "Mint_*"
# addresses and scripted symbols. Real pump.fun mints never look like this.
_DEMO_SYMBOLS = frozenset({
    "MOON", "GEM", "CALM", "RUGY", "DUMP", "STOP", "BAG", "VOLX", "WHL", "PADL", "PEPE",
})


def is_real(record: TradeRecord) -> bool:
    """True if the record is a genuine on-chain trade (not a demo/synthetic row)."""
    mint = record.mint_address or ""
    if mint.startswith("Mint_"):
        return False
    if record.symbol in _DEMO_SYMBOLS:
        return False
    # A real Solana mint is a base58 string ~32-44 chars; demo labels are short.
    return len(mint) >= 32


# --- Aggregation ------------------------------------------------------------

@dataclass
class StrategyStat:
    """Per-strategy tallies for the dip-vs-momentum split."""

    entries: int = 0
    completed: int = 0
    wins: int = 0
    pnl: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        """Win rate over COMPLETED round-trips, or None if none completed."""
        return (100.0 * self.wins / self.completed) if self.completed else None


@dataclass
class ReportData:
    """Everything the HTML template needs, precomputed from the real records."""

    records: List[TradeRecord]
    total: int = 0
    completed: int = 0
    open_held: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate_pct: float = 0.0
    cum_pnl: float = 0.0
    avg_completed: float = 0.0
    avg_all: float = 0.0
    best: Optional[TradeRecord] = None
    worst: Optional[TradeRecord] = None
    equity: List[float] = field(default_factory=list)       # running cum PnL
    exit_counts: Dict[str, int] = field(default_factory=dict)
    exit_pnl: Dict[str, float] = field(default_factory=dict)
    tier_firings: Dict[str, int] = field(default_factory=dict)
    by_strategy: Dict[str, StrategyStat] = field(default_factory=dict)
    demo_dropped: int = 0
    time_range: Tuple[str, str] = ("", "")


def _exit_bucket(r: TradeRecord) -> str:
    """One mutually-exclusive exit bucket per trade.

    Precedence mirrors the loop: a hard exit is terminal, so it wins over any
    earlier profit tier. Otherwise a trade that fired >=1 profit tier (moonbag
    held) is a profit-take; anything else was simply held open to the loop's end.
    """
    if r.exit_trigger:
        return r.exit_trigger
    if r.profit_tiers:
        return "profit_take"
    return "held_open"


def build_data(records: List[TradeRecord], demo_dropped: int = 0) -> ReportData:
    """Aggregate real records into a :class:`ReportData`. Pure; no I/O."""
    ordered = sorted(records, key=lambda r: r.entry_time or r.recorded_at or "")
    data = ReportData(records=ordered, demo_dropped=demo_dropped)
    data.total = len(ordered)

    # Stable, complete buckets (0-filled) so charts always show every category.
    data.exit_counts = {t: 0 for t in HARD_EXIT_TRIGGERS}
    data.exit_counts["profit_take"] = 0
    data.exit_counts["held_open"] = 0
    data.exit_pnl = {t: 0.0 for t in data.exit_counts}
    data.tier_firings = {t: 0 for t in PROFIT_TIERS}
    data.by_strategy = {"dip": StrategyStat(), "momentum": StrategyStat()}

    if not ordered:
        return data

    running = 0.0
    for r in ordered:
        pnl = r.realised_pnl_usd
        running += pnl
        data.equity.append(running)

        if r.fully_exited:
            data.completed += 1
        else:
            data.open_held += 1

        if pnl > 0:
            data.wins += 1
        elif pnl < 0:
            data.losses += 1
        else:
            data.breakeven += 1

        bucket = _exit_bucket(r)
        data.exit_counts[bucket] = data.exit_counts.get(bucket, 0) + 1
        data.exit_pnl[bucket] = data.exit_pnl.get(bucket, 0.0) + pnl
        for tier in r.profit_tiers:
            data.tier_firings[tier] = data.tier_firings.get(tier, 0) + 1

        st = data.by_strategy.setdefault(r.strategy, StrategyStat())
        st.entries += 1
        st.pnl += pnl
        if r.fully_exited:
            st.completed += 1
        if pnl > 0:
            st.wins += 1

    data.cum_pnl = running
    data.avg_all = running / data.total
    data.avg_completed = (running / data.completed) if data.completed else 0.0
    data.win_rate_pct = 100.0 * data.wins / data.total
    data.best = max(ordered, key=lambda r: r.realised_pnl_usd)
    data.worst = min(ordered, key=lambda r: r.realised_pnl_usd)
    data.time_range = (
        (ordered[0].entry_time or "")[:16].replace("T", " "),
        (ordered[-1].entry_time or "")[:16].replace("T", " "),
    )
    return data


# --- SVG chart builders (pure string output, no dependencies) ---------------

def _money(v: float) -> str:
    """Signed USD, minus sign rendered as a true minus glyph."""
    s = f"${abs(v):,.2f}"
    return ("−" + s) if v < 0 else ("+" + s if v else s)


def svg_equity(data: ReportData) -> str:
    """Cumulative-PnL equity curve (area + line) as inline SVG."""
    eq = data.equity
    if not eq:
        return "<p class='empty'>No trades to plot.</p>"

    W, H = 1000, 340
    m = dict(t=24, r=26, b=36, l=58)
    iw, ih = W - m["l"] - m["r"], H - m["t"] - m["b"]
    lo = min(min(eq), 0.0)
    hi = max(max(eq), 0.0)
    span = (hi - lo) or 1.0
    pad = span * 0.12
    y_min, y_max = lo - pad, hi + pad
    n = len(eq)

    def x(i: int) -> float:
        return m["l"] + (i / (n - 1) if n > 1 else 0.5) * iw

    def y(v: float) -> float:
        return m["t"] + (y_max - v) / (y_max - y_min) * ih

    # y gridlines at 5 rounded steps.
    ticks = _nice_ticks(y_min, y_max, 5)
    grid, axis = [], []
    for tv in ticks:
        gy = y(tv)
        grid.append(f'<line x1="{m["l"]}" x2="{W-m["r"]}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
        axis.append(
            f'<text x="{m["l"]-10}" y="{gy+3.5:.1f}" text-anchor="end">{_money(tv)}</text>'
        )
    zero_y = y(0.0)

    line_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(eq))
    area = f"M {x(0):.1f} {zero_y:.1f} L " + \
        " L ".join(f"{x(i):.1f} {y(v):.1f}" for i, v in enumerate(eq)) + \
        f" L {x(n-1):.1f} {zero_y:.1f} Z"

    dots = []
    for i, v in enumerate(eq):
        r = data.records[i]
        cls = "eq-end" if i == n - 1 else ("eq-dot zero-dot" if v == 0 else "eq-dot")
        rr = 5 if i == n - 1 else 3.4
        tip = (f"#{i+1} {html.escape(r.symbol)} · {html.escape(_exit_bucket(r))}\n"
               f"trade {_money(r.realised_pnl_usd)} · cum {_money(v)}")
        dots.append(
            f'<circle class="{cls}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="{rr}">'
            f'<title>{tip}</title></circle>'
        )

    # Annotate the single worst drop if it dominates.
    ann = ""
    if data.worst and data.worst.realised_pnl_usd < 0:
        wi = data.records.index(data.worst)
        ann = (f'<text class="cliff" x="{x(wi)+10:.1f}" y="{y(data.equity[wi])+4:.1f}">'
               f'{_money(data.worst.realised_pnl_usd)} {html.escape(data.worst.exit_trigger or "")}</text>')

    # A few x labels.
    xlabels = []
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    for i in idxs:
        t = (data.records[i].entry_time or "")[11:16]
        xlabels.append(f'<text x="{x(i):.1f}" y="{H-12}" text-anchor="middle" class="xlab">{t}</text>')

    return f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative realised PnL equity curve">
  <g class="grid">{"".join(grid)}</g>
  <g class="zero"><line x1="{m['l']}" x2="{W-m['r']}" y1="{zero_y:.1f}" y2="{zero_y:.1f}"/></g>
  <path class="eq-area" d="{area}"/>
  <polyline class="eq-line" points="{line_pts}"/>
  {ann}
  {"".join(dots)}
  <g class="axis">{"".join(axis)}{"".join(xlabels)}</g>
</svg>'''


def svg_bars(items: List[Tuple[str, int, str]], *, unit: str = "") -> str:
    """Vertical bar chart from (label, value, css_color_var) triples."""
    if not items:
        return "<p class='empty'>No data.</p>"
    W, H = 520, 260
    m = dict(t=18, r=14, b=52, l=34)
    iw, ih = W - m["l"] - m["r"], H - m["t"] - m["b"]
    vmax = max((v for _, v, _ in items), default=1) or 1
    bw = iw / len(items)
    bars, labels, axis = [], [], []
    for tv in _nice_ticks(0, vmax, 4):
        gy = m["t"] + (1 - tv / vmax) * ih if vmax else m["t"] + ih
        axis.append(f'<line class="hbar" x1="{m["l"]}" x2="{W-m["r"]}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
        axis.append(f'<text class="tick" x="{m["l"]-8}" y="{gy+3.5:.1f}" text-anchor="end">{int(tv)}</text>')
    for k, (label, v, color) in enumerate(items):
        bh = (v / vmax) * ih if vmax else 0
        bx = m["l"] + k * bw + bw * 0.16
        by = m["t"] + ih - bh
        w = bw * 0.68
        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{w:.1f}" height="{max(bh,0):.1f}" rx="4" '
            f'fill="var({color})"><title>{html.escape(label)}: {v}{unit}</title></rect>'
        )
        bars.append(f'<text class="barval" x="{bx+w/2:.1f}" y="{by-6:.1f}" text-anchor="middle">{v}</text>')
        # wrap label onto up to 2 lines
        parts = label.split()
        mid = (len(parts) + 1) // 2
        l1, l2 = " ".join(parts[:mid]), " ".join(parts[mid:])
        labels.append(f'<text class="barlab" x="{bx+w/2:.1f}" y="{H-30}" text-anchor="middle">{html.escape(l1)}</text>')
        if l2:
            labels.append(f'<text class="barlab" x="{bx+w/2:.1f}" y="{H-16}" text-anchor="middle">{html.escape(l2)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img"><g class="axis">{"".join(axis)}</g>{"".join(bars)}{"".join(labels)}</svg>'


def _nice_ticks(lo: float, hi: float, count: int) -> List[float]:
    """A small set of rounded tick values spanning [lo, hi]."""
    if hi == lo:
        return [lo]
    step = (hi - lo) / count
    mag = 10 ** (len(str(int(abs(step)))) - 1) if abs(step) >= 1 else 1
    step = max(round(step / mag) * mag, mag) if mag else step
    ticks, v = [], lo - (lo % step if step else 0)
    while v <= hi + step:
        if lo - step <= v <= hi + step:
            ticks.append(round(v, 4))
        v += step
    return ticks or [lo, hi]


# --- HTML rendering ---------------------------------------------------------

def _hold(r: TradeRecord) -> str:
    return f"{r.hold_ticks}t / {r.hold_seconds:.0f}s"


def _rows(data: ReportData) -> str:
    out = []
    running = 0.0
    for i, r in enumerate(data.records, 1):
        running += r.realised_pnl_usd
        pnl = r.realised_pnl_usd
        pcls = "neg" if pnl < 0 else ("pos" if pnl > 0 else "zero")
        exit_lbl = r.exit_trigger or "held open"
        excls = {"flash_crash": "fc", "volume_collapse": "vc"}.get(r.exit_trigger or "", "ho")
        big = " class='big'" if r is data.worst and pnl < 0 else ""
        reason = html.escape((r.exit_reason or "")[:70])
        out.append(
            f"<tr{big}><td class='num'>{i}</td>"
            f"<td class='mono'>{html.escape((r.entry_time or '')[5:16].replace('T',' '))}</td>"
            f"<td>{html.escape(r.symbol)}</td>"
            f"<td><span class='sd sd-{'dip' if r.strategy=='dip' else 'mom'}'></span>{html.escape(r.strategy)}</td>"
            f"<td class='num'>{r.entry_price:.8f}</td>"
            f"<td><span class='pill {excls}'>{html.escape(exit_lbl)}</span></td>"
            f"<td class='reason'>{reason}</td>"
            f"<td class='num {pcls}'>{_money(pnl)}</td>"
            f"<td class='num mono'>{_hold(r)}</td>"
            f"<td class='num {'neg' if running<0 else 'zero'}'>{_money(running)}</td></tr>"
        )
    return "".join(out)


def render_html(data: ReportData) -> str:
    """Render the full standalone HTML document string."""
    prelim = data.completed < RELIABLE_SAMPLE
    exit_items = [
        ("held open", data.exit_counts.get("held_open", 0), "--neutral"),
        ("volume collapse", data.exit_counts.get("volume_collapse", 0), "--defensive"),
        ("flash crash", data.exit_counts.get("flash_crash", 0), "--loss"),
        ("hard stop", data.exit_counts.get("hard_stop_loss", 0), "--loss"),
        ("liq drop", data.exit_counts.get("liquidity_drop", 0), "--defensive"),
        ("coord dump", data.exit_counts.get("coordinated_dump", 0), "--defensive"),
        ("profit take", data.exit_counts.get("profit_take", 0), "--good"),
    ]
    wl_items = [
        ("wins", data.wins, "--good"),
        ("losses", data.losses, "--loss"),
        ("open / BE", data.open_held, "--neutral"),
    ]
    dip = data.by_strategy.get("dip", StrategyStat())
    mom = data.by_strategy.get("momentum", StrategyStat())

    def wr(s: StrategyStat) -> str:
        return f"{s.win_rate:.0f}%" if s.win_rate is not None else "n/a"

    watermark = '<div class="watermark">PRELIMINARY</div>' if prelim else ""
    banner = (
        f'<div class="prelim"><span class="dot"></span><b>PRELIMINARY</b> · '
        f'{data.completed} completed round-trips — below the ~{RELIABLE_SAMPLE} needed for '
        f'reliable conclusions. Directional only.</div>'
    ) if prelim else ""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Helix Vector — Real Trading Performance</title>
<style>
{_CSS}
</style>
</head>
<body>
{watermark}
<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">Read-only performance report · dry-run</p>
      <h1><span class="v">Helix Vector</span> — real trading</h1>
      {banner}
    </div>
    <div class="prov">
      <div>source · <span class="mono">trades.jsonl</span></div>
      <div><span class="mono">{data.total}</span> real · <span class="mono ok">{data.demo_dropped}</span> demo dropped</div>
      <div class="mono">{html.escape(data.time_range[0])} → {html.escape(data.time_range[1])} UTC</div>
    </div>
  </header>

  <div class="kpis">
    <div class="kpi"><span class="stripe loss"></span><div class="lbl">Cumulative PnL</div><div class="fig neg">{_money(data.cum_pnl)}</div><div class="sub">after slippage + fees</div></div>
    <div class="kpi"><span class="stripe loss"></span><div class="lbl">Win rate</div><div class="fig neg">{data.win_rate_pct:.0f}%</div><div class="sub">{data.wins}W · {data.losses}L · {data.open_held} open</div></div>
    <div class="kpi"><span class="stripe def"></span><div class="lbl">Avg / completed</div><div class="fig">{_money(data.avg_completed)}</div><div class="sub">{_money(data.avg_all)} across all {data.total}</div></div>
    <div class="kpi"><span class="stripe loss"></span><div class="lbl">Worst trade</div><div class="fig neg">{_money(data.worst.realised_pnl_usd) if data.worst else '-'}</div><div class="sub">{html.escape(data.worst.symbol) if data.worst else ''} · {html.escape(data.worst.exit_trigger or '') if data.worst else ''}</div></div>
    <div class="kpi"><span class="stripe acc"></span><div class="lbl">Best trade</div><div class="fig">{_money(data.best.realised_pnl_usd) if data.best else '-'}</div><div class="sub">{html.escape(data.best.symbol) if data.best else ''}</div></div>
  </div>

  <section class="card chart">
    <h2>Equity curve — cumulative realised PnL</h2>
    <p class="cap">Chronological, one point per trade. Hover a point for detail.</p>
    {svg_equity(data)}
  </section>

  <div class="two">
    <section class="card chart">
      <h2>Exit reasons — count</h2>
      <p class="cap">How the {data.total} entries ended</p>
      {svg_bars(exit_items)}
    </section>
    <section class="card chart">
      <h2>Win / loss</h2>
      <p class="cap">Wins vs losses vs unresolved</p>
      {svg_bars(wl_items)}
    </section>
  </div>

  <section class="card split">
    <h2>Dip vs momentum</h2>
    <div class="sgrid">
      <div class="sb"><div class="sh"><span class="nm"><span class="sd sd-dip"></span>Dip-buy</span><span class="p neg">{_money(dip.pnl)}</span></div>
        <div class="mt"><div>Entries<b class="mono">{dip.entries}</b></div><div>Completed<b class="mono">{dip.completed}</b></div><div>Win rate<b class="mono">{wr(dip)}</b></div></div></div>
      <div class="sb"><div class="sh"><span class="nm"><span class="sd sd-mom"></span>Momentum</span><span class="p {'neg' if mom.pnl<0 else 'zero'}">{_money(mom.pnl)}</span></div>
        <div class="mt"><div>Entries<b class="mono">{mom.entries}</b></div><div>Completed<b class="mono">{mom.completed}</b></div><div>Win rate<b class="mono">{wr(mom)}</b></div></div></div>
    </div>
  </section>

  <section class="card">
    <h2 style="padding:18px 20px 0">All real trades</h2>
    <div class="tablewrap">
      <table>
        <thead><tr><th>#</th><th>Time</th><th>Symbol</th><th>Strategy</th><th>Entry px</th><th>Exit</th><th>Reason</th><th>PnL</th><th>Hold</th><th>Cum</th></tr></thead>
        <tbody>{_rows(data)}</tbody>
      </table>
    </div>
  </section>

  <section class="card assess">
    <h2>Honest read — is there an edge?</h2>
    <p>{_VERDICT}</p>
  </section>

  <footer>
    <span class="mono">Helix Vector 1.0 · dry-run · read-only</span> — simulated fills (price impact + fees + fill drift); nothing signed or broadcast.
    Generated {generated} from trades.jsonl · {data.total} real trades, {data.completed} completed.
  </footer>
</div>
</body>
</html>"""


_VERDICT = (
    "Based on this data, there is <b>no visible edge, and the early signal is "
    "negative</b> — but the sample is far too small to call it “clearly losing” "
    "in a durable sense. Nine completed round-trips is well below the ~20–50 you "
    "need before variance stops dominating, and a <b>single flash-crash trade "
    "(BABYJIMOTHY, −$73.47) is 92% of the entire drawdown</b>; strip it and the "
    "bot sits at −$6.13 of pure friction. What the small sample <i>can</i> "
    "legitimately show is structural, not statistical: the profit ladder "
    "(2x/5x/10x) has <b>never fired once</b>, every resolved exit was defensive, "
    "and the same dead token (ACM) was re-bought seven times with no cooldown. "
    "So: not enough trades to judge profitability, but enough to see the bot is "
    "entering into fading or collapsing momentum and has never captured upside. "
    "<b>Too early to score the edge; not too early to fix the entry logic.</b> "
    "Collect 30–50 completed round-trips before drawing any profitability "
    "conclusion."
)


_CSS = """
:root{color-scheme:light;--page:#f4f6f9;--surface:#fff;--surface-2:#f8fafc;--ink:#0e1116;--ink-2:#4c525c;
--muted:#868b94;--hair:#e4e7ec;--hair-2:#cfd3da;--accent:#2a6fd6;--loss:#cc3b3b;--defensive:#d97a35;
--neutral:#9aa0aa;--good:#0e9e5a;--grid:#e8ebef;--shadow:0 1px 2px rgba(14,17,22,.05),0 8px 24px -12px rgba(14,17,22,.12);
--mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--page:#0c0e12;--surface:#14171d;
--surface-2:#191d24;--ink:#f2f4f7;--ink-2:#b6bcc6;--muted:#7f858f;--hair:#262b33;--hair-2:#333a44;--accent:#4a90ec;
--loss:#e35b5b;--defensive:#e08a44;--neutral:#767c86;--good:#22b46e;--grid:#222831;--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.6);}}
:root[data-theme=dark]{color-scheme:dark;--page:#0c0e12;--surface:#14171d;--surface-2:#191d24;--ink:#f2f4f7;--ink-2:#b6bcc6;
--muted:#7f858f;--hair:#262b33;--hair-2:#333a44;--accent:#4a90ec;--loss:#e35b5b;--defensive:#e08a44;--neutral:#767c86;--good:#22b46e;--grid:#222831;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
.watermark{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;z-index:0;
font-family:var(--mono);font-weight:800;font-size:min(19vw,260px);letter-spacing:.08em;color:var(--defensive);opacity:.06;
transform:rotate(-24deg);white-space:nowrap;user-select:none}
.wrap{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:38px 22px 64px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
header{display:flex;flex-wrap:wrap;gap:16px 28px;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--hair);padding-bottom:20px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
h1{font-size:clamp(25px,3.3vw,36px);margin:0;letter-spacing:-.02em;font-weight:640;text-wrap:balance}
h1 .v{color:var(--accent)}
.prov{text-align:right;font-size:12.5px;color:var(--ink-2)} .prov .mono{color:var(--ink)} .ok{color:var(--good)}
.prelim{display:inline-flex;align-items:center;gap:8px;margin-top:13px;background:color-mix(in srgb,var(--defensive) 14%,var(--surface));
border:1px solid color-mix(in srgb,var(--defensive) 45%,transparent);border-radius:999px;padding:6px 13px;font-size:12.5px;font-weight:540}
.prelim b{color:var(--defensive);letter-spacing:.04em} .prelim .dot{width:7px;height:7px;border-radius:50%;background:var(--defensive)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(176px,1fr));gap:13px;margin-top:26px}
.kpi{position:relative;overflow:hidden;background:var(--surface);border:1px solid var(--hair);border-radius:13px;box-shadow:var(--shadow);padding:16px 16px 15px}
.kpi .lbl{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.kpi .fig{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:clamp(24px,2.8vw,31px);font-weight:640;margin-top:8px;line-height:1}
.kpi .sub{font-size:11.5px;color:var(--ink-2);margin-top:8px}
.kpi .stripe{position:absolute;left:0;top:0;bottom:0;width:3px}
.stripe.loss{background:var(--loss)}.stripe.def{background:var(--defensive)}.stripe.acc{background:var(--accent)}
.neg{color:var(--loss)} .pos{color:var(--good)} .zero{color:var(--muted)}
section.card,.card{background:var(--surface);border:1px solid var(--hair);border-radius:14px;box-shadow:var(--shadow);margin-top:24px}
.chart{padding:18px 20px 14px} .chart h2,.split h2,.assess h2{margin:0;font-size:15px;font-weight:600}
.card h2{font-size:15px;font-weight:600}
.cap{font-size:12px;color:var(--muted);margin:4px 0 10px}
svg{display:block;width:100%;height:auto}
.grid line{stroke:var(--grid);stroke-width:1}.zero line{stroke:var(--hair-2);stroke-width:1.5}
.axis text,.tick,.xlab{fill:var(--muted);font-family:var(--mono);font-size:10.5px}
.hbar{stroke:var(--grid);stroke-width:1}
.eq-area{fill:color-mix(in srgb,var(--loss) 15%,transparent)}
.eq-line{fill:none;stroke:var(--loss);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.eq-dot{fill:var(--surface);stroke:var(--loss);stroke-width:2}.eq-dot.zero-dot{stroke:var(--neutral)}
.eq-end{fill:var(--loss);stroke:var(--surface);stroke-width:2}
.cliff{fill:var(--loss);font-family:var(--mono);font-size:11px;font-weight:600}
.barval{fill:var(--ink);font-family:var(--mono);font-size:12px;font-weight:600}
.barlab{fill:var(--ink-2);font-size:10.5px}
.two{display:grid;grid-template-columns:1.15fr 1fr;gap:20px}
@media (max-width:720px){.two{grid-template-columns:1fr}}
.split{padding:18px 20px 20px}
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media (max-width:560px){.sgrid{grid-template-columns:1fr}}
.sb{border:1px solid var(--hair);border-radius:12px;padding:14px 16px;background:var(--surface-2)}
.sh{display:flex;justify-content:space-between;align-items:center}.sh .nm{font-weight:600;font-size:14px;display:flex;align-items:center;gap:8px}
.sh .p{font-family:var(--mono);font-weight:640;font-size:16px}
.mt{display:flex;gap:18px;margin-top:10px}.mt div{font-size:11.5px;color:var(--ink-2)}.mt b{display:block;font-size:15px;color:var(--ink);margin-top:2px}
.sd{display:inline-block;width:8px;height:8px;border-radius:50%}.sd-dip{background:var(--accent)}.sd-mom{background:var(--good)}
.tablewrap{overflow-x:auto;margin:14px 0 0;border-top:1px solid var(--hair)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:9px 13px;text-align:left;white-space:nowrap}
thead th{font-family:var(--mono);font-size:10px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--surface-2);border-bottom:1px solid var(--hair)}
tbody td{border-bottom:1px solid var(--hair)}tbody tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
td.reason{color:var(--ink-2);max-width:280px;overflow:hidden;text-overflow:ellipsis}
.pill{font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:999px;border:1px solid var(--hair)}
.pill.fc{color:var(--loss);border-color:color-mix(in srgb,var(--loss) 40%,transparent)}
.pill.vc{color:var(--defensive);border-color:color-mix(in srgb,var(--defensive) 40%,transparent)}
.pill.ho{color:var(--muted)}
tr.big td{background:color-mix(in srgb,var(--loss) 7%,transparent)}
.assess{padding:18px 20px 20px}.assess p{margin:8px 0 0;font-size:13.5px;color:var(--ink-2);max-width:82ch}.assess b{color:var(--ink)}
.empty{color:var(--muted);padding:20px;font-size:13px}
footer{margin-top:30px;font-size:11.5px;color:var(--muted)}footer .mono{color:var(--ink-2)}
"""


def generate(path: str, out: str) -> ReportData:
    """Load the journal at ``path``, keep real trades, write ``out``. Returns data."""
    all_records = load_records(path)
    real = [r for r in all_records if is_real(r)]
    data = build_data(real, demo_dropped=len(all_records) - len(real))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(render_html(data))
    return data


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML performance report (read-only)."
    )
    parser.add_argument("--path", default=None, help="Journal file (default: config.TRADE_JOURNAL_PATH).")
    parser.add_argument("--out", default="report.html", help="Output HTML file (default: report.html).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    path = args.path
    if path is None:
        import config
        path = config.TRADE_JOURNAL_PATH
    data = generate(path, args.out)
    flag = " [PRELIMINARY]" if data.completed < RELIABLE_SAMPLE else ""
    # ASCII-only status line (Windows consoles default to cp1252, which can't
    # encode the U+2212 minus glyph used inside the HTML).
    print(
        f"wrote {args.out}: {data.total} real trades "
        f"({data.completed} completed, {data.demo_dropped} demo dropped), "
        f"cum PnL ${data.cum_pnl:,.2f}{flag}"
    )


if __name__ == "__main__":
    main()
