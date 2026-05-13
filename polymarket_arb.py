#!/usr/bin/env python3
"""
Polymarket Election Arbitrage Scanner

Scans active Polymarket *election* events (e.g. "NC-03 House Election Winner")
with 2+ outcomes and finds cases where the sum of best ask prices across all
outcomes is below $1.00 — net of Polymarket's per-market protocol fee.

Strategy
--------
An election event like "NC-03 House Election Winner" has one market per
candidate, each traded as a YES/NO binary. Exactly one candidate wins, so
exactly one YES token pays out $1.00 and the rest pay $0.00.

If you buy YES on every candidate and the total cost (fees included) is
below $1.00, you are guaranteed a profit regardless of who wins:

    net_profit_per_share = 1 − sum(p_i) − sum(feeRate × p_i × (1 − p_i))

Filters
-------
- Election-only by default (override with --all-markets).
- Markets where new outcomes can be added over time (e.g. "When will the
  ceasefire happen?" with rolling monthly options) are excluded, because a
  new cheap outcome added after you buy in would destroy the arb.

Fees
----
Polymarket's CTFExchange charges:

    fee = C × feeRate × p × (1 − p)

where C is share count, p is the share price, and feeRate is a decimal.
The politics taker rate is 0.04 (4%). Per share that simplifies to
`feeRate × p × (1 − p)`. Note p·(1−p) peaks at p=0.5 and decays to 0 at
the price extremes, so fees bite hardest in the middle of the price range.
We sum the per-leg fees and subtract from gross profit, so the reported
figures are net of protocol fees.

Usage
-----
    uv run python polymarket_arb.py
    uv run python polymarket_arb.py --min-profit 0.01   # at least 1 cent NET
    uv run python polymarket_arb.py --fee-rate 0.04     # politics taker rate
    uv run python polymarket_arb.py --all-markets       # disable election filter
    uv run python polymarket_arb.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

try:
    import requests
    from tqdm import tqdm
except ImportError as _ie:
    sys.exit(f"Missing dependency — run: uv add requests tqdm\n({_ie})")


# ── Configuration ─────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

GAMMA_PAGE_SIZE = 500  # max page size the Gamma API allows
CLOB_BATCH_SIZE = 20  # tokens per POST /books request
REQUEST_DELAY = 0.05  # seconds between CLOB batches (well under rate limits)

# Polymarket politics market taker fee rate (decimal). Applied to every leg
# unless overridden at the CLI with --fee-rate.
POLITICS_TAKER_FEE_RATE = 0.04

# ── Trading rules ─────────────────────────────────────────────────────────────
# Buy `bottleneck − ENTRY_BOTTLENECK_OFFSET` shares on every leg so the order
# never tries to consume the very last shares at the displayed price (those
# can vanish between snapshot and submit).
ENTRY_BOTTLENECK_OFFSET = 2

# Sum of best asks across all outcomes must be ≤ this fraction (98%).
MAX_IMPLIED_ODDS_SUM = 0.98

# Hard cap on total USD across all legs of a single basket entry.
MAX_TOTAL_TRADE_USD = 1000.0

# Each individual leg's entry value (shares × ask) must STRICTLY exceed this.
MIN_LEG_VALUE_USD = 1.0

# Exit triggers when ``sum(best_bid per leg) - entry_total_per_share`` is at
# least EXIT_MIN_EDGE_USD. The exit fills against bids (you cross the spread
# to sell out of the basket), so this measures the realized surplus over
# the entry cost basis. EXIT_MIN_EDGE_PCT is retained for diagnostics only —
# ``evaluate_exit`` reports it in ``edge_improvement_pct`` but no longer
# uses it as a trigger.
EXIT_MIN_EDGE_USD = 0.01
EXIT_MIN_EDGE_PCT = 0.01

# Default JSON file used to persist open and closed positions across runs.
DEFAULT_POSITIONS_FILE = Path("polymarket_positions.json")

# Default JSON file used as the append-only audit log of paper-trading entries
# and exit fills. One record per entry, one per partial exit fill.
DEFAULT_PAPER_TRADES_LOG = Path("polymarket_paper_trades.json")

# Matches a temporal reference: full month name (jan/january/march/…), year, or "end".
# Must be kept as a plain string so it can be embedded into larger patterns.
_MONTH_PAT = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_TEMPORAL_PAT = r"(?:" + _MONTH_PAT + r"|20\d\d|end)"

# Event titles suggesting the set of outcomes grows over time.
# Also catches "Will X be out/fired/gone by [date]?" and "on or before [date]?" patterns.
_DYNAMIC_TITLE_RE = re.compile(
    r"\b(?:when will|what (?:date|month|week|day|quarter)|by when|first (?:day|week|month))\b"
    # "by / before / on or before / prior to [month|year|end]" anywhere in the title
    r"|\b(?:by|before|on\s+or\s+before|prior\s+to)\s+" + _TEMPORAL_PAT + r"\b"
    # "will … by/before/on or before [date]" — "Will X be fired by March 2026?"
    r"|\bwill\b.{1,120}?\b(?:by|before|on\s+or\s+before)\s+" + _TEMPORAL_PAT + r"\b",
    re.IGNORECASE,
)

# Outcome labels that are themselves a date or date range (rolling deadline buckets).
_DATE_OUTCOME_RE = re.compile(
    r"""
    \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b  # Jan 2025
  | \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b # June 30
  | \b20\d\d-(0[1-9]|1[0-2])\b                                               # 2025-03
  | \bq[1-4]\s+20\d\d\b                                                       # Q2 2025
  | \bweek\s+of\b                                                              # week of …
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Deadline/boundary phrases inside an outcome label that signal a non-exhaustive
# sequential market. "by June 30", "before December", "on or before March" all mean
# Polymarket can keep appending future dates, so buying all current outcomes does NOT
# guarantee a $1.00 payout.
_DEADLINE_PHRASE_RE = re.compile(
    r"\b(?:by|before|on\s+or\s+before|prior\s+to|scheduled\s+by|happen\s+by|occur\s+by)"
    r"\s+" + _TEMPORAL_PAT + r"\b",
    re.IGNORECASE,
)

# "June FOMC Meeting", "July Meeting" — Polymarket appends new months as meetings are announced.
_MEETING_OUTCOME_RE = re.compile(
    r"\b" + _MONTH_PAT + r".{0,20}?\bmeeting\b",
    re.IGNORECASE,
)

# Titles naming an open entity question: "Who will acquire X?", "Which company will Y?"
# Non-exhaustive unless a catch-all outcome ("Other", "None", …) is present.
_OPEN_ENTITY_TITLE_RE = re.compile(
    r"\bwho\s+will\s+(?:acquire|buy|purchase|merge\s+with|be\s+acquired|receive|take\s+over)\b"
    r"|\bwhich\s+(?:company|country|nation|party|candidate|bank|fund|firm)\s+will\b",
    re.IGNORECASE,
)

# Outcomes that serve as a residual catch-all, proving collective exhaustiveness.
_CATCHALL_OUTCOME_RE = re.compile(
    r"^(?:other|none|no\s+deal|no\s+one|someone\s+else|the\s+field|field)$",
    re.IGNORECASE,
)

# Titles suggesting multiple outcomes can simultaneously resolve YES (non-mutually-exclusive).
_MULTI_WINNER_TITLE_RE = re.compile(
    r"\b(?:any\s+of(?:\s+the\s+following)?|which\s+of\s+the\s+following|at\s+least\s+one\s+of)\b",
    re.IGNORECASE,
)

# "What price will Bitcoin hit?", "What level will the S&P 500 reach?" — no price is guaranteed.
# The asset may stay below (or above) every listed target, so zero outcomes would resolve YES.
_OPEN_PRICE_QUESTION_RE = re.compile(
    r"\bwhat\s+(?:price|level|value|ath|all.?time.?high)\s+will\b",
    re.IGNORECASE,
)

# "What will happen before X?" — none of the listed events may occur before the trigger.
_WHAT_BEFORE_RE = re.compile(
    r"\bwhat\s+will\s+(?:happen|occur)\b"
    r"|\bwhat\s+(?:happens|occurs)\s+first\b",
    re.IGNORECASE,
)

# An outcome label that is entirely a price target: "$100k", "$1,500,000", "100k", "1.5M".
# Used to detect multi-outcome price-hit markets (all outcomes = price points → non-exhaustive).
_PRICE_LEVEL_OUTCOME_RE = re.compile(
    r"^\s*\$[\d,]+(?:\.\d+)?[kmb]?\s*$"  # "$100k", "$1,500,000", "$200M"
    r"|^\s*[\d,]+(?:\.\d+)?\s*[km]\s*$",  # "100k", "1.5M" — suffix required to avoid year numbers
    re.IGNORECASE,
)

# Election-related title keywords. Matches direct election words plus common
# adjacent terms (senate/governor/mayor/parliament/etc.) and US Congressional
# district codes ("NC-03", "TX-32") via the separate _DISTRICT_CODE_RE below.
_ELECTION_TITLE_RE = re.compile(
    r"\belections?\b"
    r"|\bre-?elections?\b"
    r"|\bsenat(?:e|or|ors|orial)\b"
    r"|\bpresiden(?:t|ts|tial|cy)\b"
    r"|\bgovern(?:or|ors|orship)\b"
    r"|\bgubernatorial\b"
    r"|\bmayor(?:s|al)?\b"
    r"|\bcongress(?:ional|man|woman|person)?\b"
    r"|\bparliament(?:ary|arian)?\b"
    r"|\bchancellor(?:ship)?\b"
    r"|\bnominee\b|\bnomination\b"
    r"|\bcaucus(?:es)?\b"
    r"|\brun-?off\b"
    r"|\bprime\s+minister\b"
    r"|\bspeaker\s+of\s+the\s+(?:house|senate|chamber|assembly)\b"
    r"|\bprimary\s+(?:election|race|winner|debate|night)\b"
    r"|\b(?:house|senate)\s+(?:race|seat|district|winner|election)\b",
    re.IGNORECASE,
)

# US Congressional / state-legislative district codes — kept case-sensitive so
# we don't match arbitrary 2-letter prefixes like "uh-2" or "id-1".
_DISTRICT_CODE_RE = re.compile(r"\b[A-Z]{2}-\d{1,2}\b")

# Sports conference championship winners. NFL (AFC/NFC), NBA (Eastern/Western),
# and the major college football conferences — all have a finite, named set of
# member teams, so "who wins the conference" is mutually exclusive & exhaustive.
_SPORTS_CONFERENCE_RE = re.compile(
    # NFL conferences: "NFC Championship", "AFC Champion", "AFC Conference winner"
    r"\b(?:nfc|afc)(?:\s+conference)?\s+(?:champion(?:ship)?|winner|title)\b"
    # NBA conferences: "Eastern Conference Finals winner", "Western Conference champion"
    r"|\b(?:eastern|western)\s+conference\s+(?:champion(?:ship)?|winner|finals?|title)\b"
    # College football conferences: "Big Ten Championship", "SEC champion", "Pac-12 winner"
    r"|\b(?:big\s+ten|big\s+12|big\s+east|sec|acc|pac-?12|american\s+athletic)"
    r"\s+(?:conference\s+)?(?:champion(?:ship)?|winner|title)\b"
    # Generic phrasing as a fallback: "conference champion[ship]" / "conference winner"
    r"|\bconference\s+(?:champion(?:ship)?|winner|title)\b",
    re.IGNORECASE,
)

# World Cup group-stage winners — must reference both World Cup and a group.
# Group A–H is the standard FIFA bracket; both orderings (cup-then-group and
# group-then-cup) are covered.
_WORLD_CUP_GROUP_RE = re.compile(
    r"\bworld\s+cup\b.{0,40}?\bgroup\s+[a-h]\b"
    r"|\bgroup\s+[a-h]\b.{0,40}?\bworld\s+cup\b",
    re.IGNORECASE,
)


# Comprehensive blocklist of title patterns where buying YES on every listed outcome cannot
# guarantee exactly one $1.00 payout — either zero or multiple outcomes may resolve YES,
# or the outcome set is provably non-exhaustive.
# Exception applied at call site: pure binary Yes/No markets are always exhaustive.
_BLOCKLIST_TITLE_RE = re.compile(
    # Price / level targets — asset may not reach any listed target
    r"\bwhat\s+will\b.{0,60}?\bhit\b"  # "What will Palantir hit?"
    r"|\bprice\s+(?:hit|touch|reach)\b"  # "crude oil price hit on any day"
    r"|\brating\s+(?:high|low)\b"  # "Trump approval rating high/low"
    # Open entity questions — companies and countries are typically non-exhaustive
    r"|\bwhich\s+countr(?:y|ies)\b"  # "Which country/countries?"
    r"|\bwhich\s+compan(?:y|ies)\b"  # "Which company/companies?"
    # MVP / last-place — only top candidates listed, not all possible winners
    r"|\bmvp\b"  # "MLB MVP", "Super Bowl MVP"
    r"|\blast\s+place\b"  # "last place in EPL"
    # Named domain patterns that produce non-exhaustive markets on Polymarket
    r"|\bNobel\b"  # Nobel Prize markets
    r"|\bJames\s+Bond\b"  # Next James Bond speculation
    r"|\bEpstein\b"  # Epstein visitor markets
    r"|\bpurge\b"  # Political purge speculation
    r"|\blead\s+bank\b"  # "Lead bank in [IPO]"
    r"|\bleader\s+at\s+end\s+of\b"  # "Iran leader at end of [year]"
    r"|\bend\s+of\s+20\d\d\b"  # "end of 2026" open-state query
    r"|\bETF\s+approval\b"  # ETF approval speculation
    r"|\bpublic\s+sales\s+commitments?\b",  # open sales-target markets
    re.IGNORECASE,
)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    title: str
    token_id: str
    best_ask: float  # lowest available ask in USD, range 0.0–1.0
    best_ask_size: float  # shares available at that price
    fee_rate: float = 0.0  # decimal — Polymarket taker fee rate (0.04 = 4%)

    @property
    def fee_per_share(self) -> float:
        """Polymarket per-share protocol fee at best_ask.

        Polymarket's CTFExchange formula:
            fee = C × feeRate × p × (1 − p)
        where C is shares and p is the share price. Per share that's:
            fee_per_share = feeRate × p × (1 − p)
        Note p·(1−p) peaks at 0.25 (when p=0.5) and decays to 0 at the price
        extremes, so fees bite hardest mid-book.
        """
        p = self.best_ask
        return self.fee_rate * p * (1 - p)

    @property
    def fee_total_usd(self) -> float:
        """Total USD fee for buying best_ask_size shares of this leg."""
        return self.fee_per_share * self.best_ask_size


@dataclass
class Opportunity:
    event_title: str
    event_url: str
    outcomes: list[Outcome]
    total_cost: float  # sum of best asks across all outcomes (no fees)
    event_slug: str = ""  # Gamma slug — stable identifier for position tracking

    @property
    def gross_profit(self) -> float:
        """Per-share profit before Polymarket fees: $1 payout − sum(asks)."""
        return 1.0 - self.total_cost

    @property
    def fees_per_share(self) -> float:
        """Sum of Polymarket protocol fees across every leg, per share."""
        return sum(o.fee_per_share for o in self.outcomes)

    @property
    def total_fees_usd(self) -> float:
        """Total USD fees if you fill max_shares on every leg."""
        return self.fees_per_share * self.max_shares

    @property
    def profit(self) -> float:
        """Net profit per share AFTER Polymarket fees."""
        return self.gross_profit - self.fees_per_share

    @property
    def profit_pct(self) -> float:
        return self.profit * 100

    @property
    def max_shares(self) -> float:
        """Max shares you can trade at best-ask prices across all outcomes.
        Bottlenecked by the thinnest leg — you need one share of every outcome."""
        return min(o.best_ask_size for o in self.outcomes)

    @property
    def max_profit_usd(self) -> float:
        """Total USD profit if you fill max_shares on every outcome (net of fees)."""
        return self.max_shares * self.profit

    # ── Entry rules (see ENTRY_BOTTLENECK_OFFSET / MAX_*) ─────────────────────

    @property
    def entry_shares(self) -> float:
        """Shares to buy per leg under the safety margin: ``max_shares − 2``.

        Clamped at 0 — if the bottleneck leg has fewer than ``OFFSET + 1``
        shares available we cannot enter at all.
        """
        return max(0.0, self.max_shares - ENTRY_BOTTLENECK_OFFSET)

    @property
    def entry_per_leg_usd(self) -> list[float]:
        """USD value of each individual leg at entry: ``entry_shares × best_ask``."""
        return [self.entry_shares * o.best_ask for o in self.outcomes]

    @property
    def entry_total_usd(self) -> float:
        """Total USD spend across all legs at entry (no fees)."""
        return self.entry_shares * self.total_cost

    def entry_check(self) -> tuple[bool, str]:
        """Return ``(passes, reason)`` for whether this opportunity is enterable.

        Rules (all must hold):
        - Bottleneck leg has at least ``ENTRY_BOTTLENECK_OFFSET + 1`` shares
          so we can buy ``max_shares − OFFSET > 0`` on every leg.
        - Sum of best asks ≤ ``MAX_IMPLIED_ODDS_SUM`` (default 0.98).
        - Each individual leg cost STRICTLY exceeds ``MIN_LEG_VALUE_USD``.
        - Combined entry value ≤ ``MAX_TOTAL_TRADE_USD``.
        """
        if self.entry_shares <= 0:
            return (
                False,
                f"bottleneck too thin (max_shares={self.max_shares:.2f}, "
                f"need > {ENTRY_BOTTLENECK_OFFSET})",
            )
        if self.total_cost > MAX_IMPLIED_ODDS_SUM:
            return (
                False,
                f"sum of asks {self.total_cost:.4f} > {MAX_IMPLIED_ODDS_SUM:.2f}",
            )
        smallest_leg = min(self.entry_per_leg_usd)
        if smallest_leg <= MIN_LEG_VALUE_USD:
            return (
                False,
                f"smallest leg ${smallest_leg:.2f} ≤ ${MIN_LEG_VALUE_USD:.2f}",
            )
        if self.entry_total_usd > MAX_TOTAL_TRADE_USD:
            return (
                False,
                f"total entry ${self.entry_total_usd:.2f} > ${MAX_TOTAL_TRADE_USD:.2f}",
            )
        return True, ""

    def to_dict(self) -> dict:
        passes, reason = self.entry_check()
        return {
            "event_title": self.event_title,
            "event_url": self.event_url,
            "total_cost_per_share": round(self.total_cost, 6),
            "gross_profit_per_share": round(self.gross_profit, 6),
            "fees_per_share": round(self.fees_per_share, 6),
            "total_fees_usd": round(self.total_fees_usd, 4),
            "profit_per_share": round(self.profit, 6),
            "profit_pct": round(self.profit_pct, 4),
            "max_shares": round(self.max_shares, 2),
            "max_profit_usd": round(self.max_profit_usd, 4),
            "entry": {
                "passes": passes,
                "reason": reason,
                "shares": round(self.entry_shares, 2),
                "total_usd": round(self.entry_total_usd, 4),
                "per_leg_usd": [round(v, 4) for v in self.entry_per_leg_usd],
            },
            "outcomes": [
                {
                    "title": o.title,
                    "token_id": o.token_id,
                    "best_ask": round(o.best_ask, 6),
                    "available_shares": round(o.best_ask_size, 2),
                    "fee_rate": o.fee_rate,
                    "fee_per_share": round(o.fee_per_share, 6),
                    "fee_total_usd": round(o.fee_total_usd, 4),
                }
                for o in self.outcomes
            ],
        }


# ── Positions (persistent state for the entry/exit workflow) ──────────────────


@dataclass
class PositionLeg:
    """A single leg of a held basket position.

    ``entry_price`` is the best ask we filled at when opening the position.
    ``fee_rate`` is captured at entry so we can reconstruct expected fees on
    exit even if Polymarket later changes the per-market rate.
    """

    title: str
    token_id: str
    entry_price: float
    fee_rate: float = 0.0


@dataclass
class Position:
    """One open or closed basket position written to ``polymarket_positions.json``.

    Identified by ``event_slug`` for cheap lookup in CLI commands like
    ``--exit <slug>``. ``status`` is ``"open"`` until ``close`` is called,
    after which ``exit_*`` fields are populated and status flips to ``"closed"``.
    """

    event_slug: str
    event_title: str
    event_url: str
    opened_at: str  # ISO-8601 UTC timestamp
    shares: float  # bottleneck − OFFSET at the moment of entry, same on every leg
    legs: list[PositionLeg]
    entry_total_per_share: float  # sum of leg entry asks (≤ MAX_IMPLIED_ODDS_SUM)
    entry_total_usd: float  # shares × entry_total_per_share
    status: str = "open"
    closed_at: Optional[str] = None
    exit_shares: Optional[float] = None
    exit_total_per_share: Optional[float] = None
    exit_total_usd: Optional[float] = None
    exit_legs: Optional[list[dict]] = None
    edge_improvement_per_share: Optional[float] = None
    edge_improvement_pct: Optional[float] = None

    @classmethod
    def from_opportunity(cls, opp: "Opportunity") -> "Position":
        shares = opp.entry_shares
        return cls(
            event_slug=opp.event_slug or opp.event_url,
            event_title=opp.event_title,
            event_url=opp.event_url,
            opened_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            shares=shares,
            legs=[
                PositionLeg(
                    title=o.title,
                    token_id=o.token_id,
                    entry_price=o.best_ask,
                    fee_rate=o.fee_rate,
                )
                for o in opp.outcomes
            ],
            entry_total_per_share=opp.total_cost,
            entry_total_usd=shares * opp.total_cost,
        )


def _position_from_dict(data: dict) -> Position:
    """Rehydrate a Position (and its legs) from a JSON-decoded dict."""
    legs_raw = data.get("legs") or []
    legs = [PositionLeg(**leg) for leg in legs_raw]
    return Position(**{**data, "legs": legs})


def load_positions(path: Path = DEFAULT_POSITIONS_FILE) -> list[Position]:
    """Read all positions from ``path``. Missing file → empty list."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_position_from_dict(p) for p in raw]


def save_positions(
    positions: list[Position], path: Path = DEFAULT_POSITIONS_FILE
) -> None:
    """Atomically write ``positions`` to ``path`` as a JSON list."""
    path.write_text(
        json.dumps([asdict(p) for p in positions], indent=2),
        encoding="utf-8",
    )


def append_paper_trade_event(
    event: dict, path: Path = DEFAULT_PAPER_TRADES_LOG
) -> None:
    """Append one entry/exit record to the paper-trading audit log.

    The log is stored as a JSON list (indented for human readability) so it
    can be opened in any text editor and re-parsed by other tools. Each call
    reads the existing list, appends ``event``, and rewrites the file.

    Records are flat: a ``type`` field discriminates between ``"entry"`` and
    ``"exit"``. Common fields (``trade_id``, ``event_slug``, ``timestamp``)
    appear on both; entry- and exit-specific fields differ.
    """
    if path.exists():
        try:
            existing_raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_raw = []
        existing = existing_raw if isinstance(existing_raw, list) else []
    else:
        existing = []
    existing.append(event)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def evaluate_exit(
    position: Position,
    current_books: Mapping[str, Optional[tuple[float, float]]],
) -> tuple[bool, str, dict]:
    """Decide whether ``position`` meets the exit conditions right now.

    Inputs:
        position       — an open Position
        current_books  — token_id → (best_bid, bid_size) or None for empty bid side

    A taker selling out of a basket position crosses the spread and hits
    bids, so the exit must be evaluated against bid prices and bid sizes
    (not asks). The trigger is purely on the SUM of best bids — there is
    no per-leg gate. A single leg's bid rising sharply is enough to fire
    the exit as long as the basket's total bid surplus crosses
    ``EXIT_MIN_EDGE_USD``.

    Exit conditions (all must hold):
    1. Every leg has a non-empty bid side (someone is willing to buy at
       any price).
    2. The bottleneck of bid sizes allows at least
       ``ENTRY_BOTTLENECK_OFFSET + 1`` shares to be sold.
    3. ``sum(best_bid per leg) ≥ entry_total_per_share + EXIT_MIN_EDGE_USD``
       — i.e. the basket can be unwound for at least 1¢ more per share
       than it was opened at.

    Returns ``(should_exit, reason, details)``. When ``should_exit`` is True
    the ``details`` dict carries the exit_shares, current per-leg bids,
    total proceeds, and edge-improvement metrics needed to actually close.
    """
    current_bids: list[float] = []
    current_sizes: list[float] = []
    for leg in position.legs:
        info = current_books.get(leg.token_id)
        if info is None:
            return False, f"no bids on leg {leg.title!r}", {}
        bid, size = info
        current_bids.append(bid)
        current_sizes.append(size)

    exit_shares = max(0.0, min(current_sizes) - ENTRY_BOTTLENECK_OFFSET)
    if exit_shares <= 0:
        return (
            False,
            f"bid bottleneck too thin to exit (min size {min(current_sizes):.2f})",
            {},
        )
    # Cap exit at the size of the held position — never sell more than we own.
    actual_exit_shares = min(exit_shares, position.shares)

    current_total = sum(current_bids)
    edge_improvement = current_total - position.entry_total_per_share
    edge_pct = (
        edge_improvement / position.entry_total_per_share
        if position.entry_total_per_share > 0
        else 0.0
    )
    if edge_improvement < EXIT_MIN_EDGE_USD:
        return (
            False,
            f"bid surplus ${edge_improvement:.4f} per share "
            f"below threshold (≥ ${EXIT_MIN_EDGE_USD:.2f})",
            {},
        )

    return (
        True,
        f"bid surplus ${edge_improvement:.4f} per share ({edge_pct * 100:.2f}%)",
        {
            "exit_shares": actual_exit_shares,
            "current_bids": current_bids,
            "current_total_per_share": current_total,
            "exit_total_usd": actual_exit_shares * current_total,
            "edge_improvement_per_share": edge_improvement,
            "edge_improvement_pct": edge_pct,
        },
    )


def close_position(position: Position, details: dict) -> Position:
    """Mark ``position`` as closed using the metrics produced by ``evaluate_exit``.

    ``exit_price`` on each leg is the best bid we sold into.
    """
    position.status = "closed"
    position.closed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    position.exit_shares = float(details["exit_shares"])
    position.exit_total_per_share = float(details["current_total_per_share"])
    position.exit_total_usd = float(details["exit_total_usd"])
    position.exit_legs = [
        {"title": leg.title, "token_id": leg.token_id, "exit_price": bid}
        for leg, bid in zip(position.legs, details["current_bids"])
    ]
    position.edge_improvement_per_share = float(details["edge_improvement_per_share"])
    position.edge_improvement_pct = float(details["edge_improvement_pct"])
    return position


# ── HTTP session ──────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({"User-Agent": "polymarket-arb-scanner/1.0"})


def _get(url: str, params: dict | None = None) -> Optional[list | dict]:
    for attempt in range(3):
        try:
            r = _session.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code == 429:
                time.sleep(2**attempt)
            else:
                print(f"  HTTP {code} for {url}")
                return None
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  Request failed: {exc}")
                return None
            time.sleep(1)
    return None


def _post(url: str, body: list | dict) -> Optional[list | dict]:
    for attempt in range(3):
        try:
            r = _session.post(url, json=body, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code == 429:
                time.sleep(2**attempt)
            else:
                print(f"  HTTP {code} for {url}")
                return None
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  Request failed: {exc}")
                return None
            time.sleep(1)
    return None


# ── Gamma API ─────────────────────────────────────────────────────────────────


def fetch_all_events() -> list[dict]:
    """Page through every active, non-closed Polymarket event."""
    events: list[dict] = []
    offset = 0
    with tqdm(desc="Fetching events", unit=" events", dynamic_ncols=True) as pbar:
        while True:
            batch = _get(
                f"{GAMMA_API}/events",
                {
                    "active": "true",
                    "closed": "false",
                    "limit": GAMMA_PAGE_SIZE,
                    "offset": offset,
                },
            )
            if not batch:
                break
            events.extend(batch)
            pbar.update(len(batch))
            if len(batch) < GAMMA_PAGE_SIZE:
                break
            offset += GAMMA_PAGE_SIZE
            time.sleep(0.3)
    return events


# ── Market filtering ──────────────────────────────────────────────────────────


def parse_clob_ids(raw: str | list | None) -> list[str]:
    """
    Return the CLOB token IDs for a market.
    The Gamma API returns them as a JSON-encoded string, e.g. '["111","222"]'.
    Index 0 = YES token, index 1 = NO token.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        return [str(t) for t in json.loads(raw)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def parse_fee_rate(market: dict, default: float = 0.0) -> float:
    """Read Polymarket's per-market fee rate as a decimal (0.04 = 4%).

    The Gamma API returns it as a string ("0", "100", …) in basis points;
    we divide by 10_000. Returns ``default`` for any missing or unparseable
    value so we never silently overstate profit.
    """
    raw = market.get("feeRateBps")
    if raw is None:
        return default
    try:
        return int(raw) / 10_000
    except (TypeError, ValueError):
        return default


def is_eligible_event(event: dict) -> bool:
    """Return True if the event belongs to one of the in-scope market families.

    The scope is intentionally narrow — we only scan markets where the outcome
    set is a finite, named list of mutually exclusive alternatives:

    - Political elections (seat / mayoral / presidential / parliamentary / etc.)
    - Sports conference winners (AFC, NFC, Eastern/Western, Big Ten, SEC, …)
    - World Cup group winners (FIFA group A–H)

    Anything else is excluded by default; pass ``--all-markets`` to override.
    """
    title = event.get("title") or ""
    if _ELECTION_TITLE_RE.search(title):
        return True
    if _DISTRICT_CODE_RE.search(title):
        return True
    if _SPORTS_CONFERENCE_RE.search(title):
        return True
    if _WORLD_CUP_GROUP_RE.search(title):
        return True
    return False


def outcome_label(market: dict) -> str:
    return (
        market.get("groupItemTitle")
        or market.get("outcomeName")
        or market.get("question")
        or "Unknown"
    )


def _parse_range(label: str) -> Optional[tuple[int, float]]:
    """Parse a pure numeric outcome label into (min, max) inclusive range.

    Returns None if the label is not a recognisable numeric form.
    Examples: "3" → (3,3), "3-5" → (3,5), "6+" → (6,inf), "3–10" → (3,10).
    Uses fullmatch so only pure numeric labels are accepted.
    """
    s = label.strip()
    m = re.fullmatch(r"(\d+)\s*\+", s)
    if m:
        return (int(m.group(1)), float("inf"))
    m = re.fullmatch(r"(\d+)\s+or\s+more", s, re.IGNORECASE)
    if m:
        return (int(m.group(1)), float("inf"))
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", s)  # hyphen or en-dash
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        return (n, n)
    return None


def is_future_expandable(event: dict, markets: list[dict]) -> tuple[bool, str]:
    """Return (True, reason) if Polymarket can append new outcomes to this market later.

    True means the market IS expandable — reject it.
    """
    labels = [outcome_label(m) for m in markets]
    title = event.get("title") or ""

    if _DYNAMIC_TITLE_RE.search(title):
        return True, "event title suggests rolling or date-gated outcomes"

    for label in labels:
        if _DEADLINE_PHRASE_RE.search(label):
            return True, f"contains rolling by-date outcomes — '{label[:60]}'"

    for label in labels:
        if _DATE_OUTCOME_RE.search(label):
            return (
                True,
                f"non-exhaustive sequential deadlines — outcome label is a date: '{label[:60]}'",
            )

    years = sorted({int(y) for lbl in labels for y in re.findall(r"\b20\d\d\b", lbl)})
    if len(years) >= 2 and years[-1] - years[0] < len(years) * 3:
        return (
            True,
            "expandable future date market — sequential year progression across outcomes",
        )

    for label in labels:
        if _MEETING_OUTCOME_RE.search(label):
            return True, f"non-exhaustive by-meeting deadline market — '{label[:60]}'"

    return False, ""


def can_multiple_outcomes_resolve_yes(
    event: dict, markets: list[dict]
) -> tuple[bool, str]:
    """Return (True, reason) if more than one outcome can simultaneously resolve YES.

    True means outcomes are NOT mutually exclusive — reject the market.
    """
    title = event.get("title") or ""
    if _MULTI_WINNER_TITLE_RE.search(title):
        return (
            True,
            f"non-mutually-exclusive outcomes — title suggests multi-select: '{title[:60]}'",
        )
    return False, ""


_RATE_CHANGE_TERMS: frozenset[str] = frozenset(
    {
        "increase",
        "hike",
        "raise",
        "decrease",
        "cut",
        "lower",
        "no change",
        "hold",
        "pause",
        "unchanged",
    }
)

_POLITICAL_PARTY_TERMS: frozenset[str] = frozenset(
    {
        "democrat",
        "republican",
        "democrats",
        "republicans",
        "democratic",
        "democratic party",
        "republican party",
        "labour",
        "conservative",
        "liberal",
        "liberal democrat",
        "green party",
        "libertarian party",
    }
)


def _is_known_exhaustive_outcome_set(labels: list[str]) -> bool:
    """Return True if outcome labels form a provably complete partition without a catch-all.

    Covers two cases:
    - Rate-change markets: all labels are rate-direction terms (Increase / Decrease / No change).
    - Two-party elections: exactly two labels, both recognisable political-party names.
    """
    normalized = {lbl.strip().lower() for lbl in labels}
    if len(normalized) >= 2 and normalized.issubset(_RATE_CHANGE_TERMS):
        return True
    if len(normalized) == 2 and normalized.issubset(_POLITICAL_PARTY_TERMS):
        return True
    return False


def can_zero_outcomes_resolve_yes(event: dict, markets: list[dict]) -> tuple[bool, str]:
    """Return (True, reason) if the actual result might match none of the listed outcomes.

    True means the outcome set may be non-exhaustive — reject the market.
    Checks (in order):
    0. Binary Yes/No — always exhaustive; return immediately.
    1. Known-exhaustive allowlist (rate-change triads, two-party elections) — return immediately.
    2. Comprehensive blocklist of non-exhaustive market patterns.
    3. Open entity/acquirer title without a catch-all outcome.
    4. Open price question ("What price will X hit?") — asset may not hit any listed target.
    5. "What will happen before X?" — none of the events may occur.
    6. All outcome labels are specific price points — non-exhaustive partition.
    """
    labels = [outcome_label(m) for m in markets]
    title = event.get("title") or ""

    # Pure binary Yes/No markets are always collectively exhaustive — one MUST resolve YES.
    # Skip all zero-resolves checks: no title pattern can make a binary market non-exhaustive.
    if len(labels) == 2 and {lbl.strip().lower() for lbl in labels} == {"yes", "no"}:
        return False, ""

    # Known-exhaustive outcome sets: rate-change triads and two-party elections.
    # Fire before the blocklist and entity checks so these are never incorrectly rejected.
    if _is_known_exhaustive_outcome_set(labels):
        return False, ""

    # 0. Blocklist: always reject these market types (binary Y/N already returned above).
    if _BLOCKLIST_TITLE_RE.search(title):
        return (
            True,
            f"blocklisted market type — no exhaustive outcome set guaranteed: '{title[:60]}'",
        )

    # 1. "Who will acquire X?" / "Which company will Y?" without Other/None
    if _OPEN_ENTITY_TITLE_RE.search(title):
        if not any(_CATCHALL_OUTCOME_RE.search(lbl) for lbl in labels):
            return True, "non-exhaustive acquirer/entity market — no catch-all outcome"

    # 2. "What price/level will X hit/reach?" — the asset may stay outside all listed targets
    if _OPEN_PRICE_QUESTION_RE.search(title):
        return (
            True,
            f"open price question — asset may not hit any listed target: '{title[:60]}'",
        )

    # 3. "What will happen before X?" — none of the events may occur in time
    if _WHAT_BEFORE_RE.search(title):
        return (
            True,
            f"open 'what happens before' question — none of the outcomes may occur: '{title[:60]}'",
        )

    # 4. Every outcome label is a bare price target ($100k, 1.5M) — non-exhaustive partition
    if len(labels) >= 2 and all(_PRICE_LEVEL_OUTCOME_RE.search(lbl) for lbl in labels):
        return (
            True,
            "non-exhaustive price-target market — outcomes are price points, not a complete partition",
        )

    return False, ""


def is_range_incomplete(markets: list[dict]) -> tuple[bool, str]:
    """Return (True, reason) if numeric outcome labels form a range with gaps or a missing zero.

    True means at least one integer count is uncovered — reject the market.
    Only activates when ALL outcome labels are pure numeric forms.
    """
    labels = [outcome_label(m) for m in markets]
    parsed = [_parse_range(lbl) for lbl in labels]
    numeric = [r for r in parsed if r is not None]

    if len(numeric) < 2 or len(numeric) != len(labels):
        return False, ""

    sorted_ranges = sorted(numeric, key=lambda r: r[0])
    lowest = sorted_ranges[0][0]
    if lowest > 0:
        return (
            True,
            f"incomplete numeric range — outcomes start at {lowest}, missing 0 to {lowest - 1}",
        )

    current_max: float = sorted_ranges[0][1]
    for lo, hi in sorted_ranges[1:]:
        if lo > current_max + 1:
            return (
                True,
                f"incomplete numeric range — gap between {int(current_max)} and {lo}",
            )
        current_max = max(current_max, hi)

    return False, ""


def is_closed_outcome_set(event: dict, markets: list[dict]) -> tuple[bool, str]:
    """Return (True, "") only if the outcome set is provably closed and collectively exhaustive.

    Thin wrapper: calls all four validation functions in order and returns the first rejection.
    Conservative — if any check fires, the market is rejected.
    """
    expandable, reason = is_future_expandable(event, markets)
    if expandable:
        return False, reason

    multi, reason = can_multiple_outcomes_resolve_yes(event, markets)
    if multi:
        return False, reason

    zero, reason = can_zero_outcomes_resolve_yes(event, markets)
    if zero:
        return False, reason

    incomplete, reason = is_range_incomplete(markets)
    if incomplete:
        return False, reason

    return True, ""


def is_valid_yes_basket_arb(event: dict, markets: list[dict]) -> tuple[bool, str]:
    """Return (True, '') only if buying YES on every outcome guarantees exactly one $1.00 payout.

    Hard rule: the outcome set must be mutually exclusive AND collectively exhaustive so that
    exactly one outcome MUST resolve YES no matter what happens in the real world.
    Returns (False, 'rejected: no guaranteed single YES outcome — <reason>') for any rejection.
    """
    valid, reason = is_closed_outcome_set(event, markets)
    if not valid:
        return False, f"rejected: no guaranteed single YES outcome — {reason}"
    return True, ""


def extract_candidates(
    events: list[dict],
    verbose: bool = False,
    eligible_only: bool = True,
) -> list[tuple[dict, list[dict]]]:
    """
    Filter the full event list down to events that qualify for scanning:
      - (optional) Title is in one of the eligible market families (elections,
        sports-conference winners, World Cup group winners) — controlled by
        ``eligible_only`` (default True). Disable with --all-markets.
      - 2 or more active, order-book-enabled markets with valid CLOB token IDs
      - Outcome set is provably closed and exhaustive (is_valid_yes_basket_arb)

    With verbose=True, logs each rejected market and prints a breakdown by
    rejection_reason category at the end.
    """
    candidates: list[tuple[dict, list[dict]]] = []
    tally: dict[str, int] = {}

    for event in events:
        if eligible_only and not is_eligible_event(event):
            tally["out-of-scope market"] = tally.get("out-of-scope market", 0) + 1
            if verbose:
                title_short = (event.get("title") or "")[:55]
                tqdm.write(f"  REJECTED  {title_short!r:57}  out of eligible scope")
            continue

        raw_markets = event.get("markets") or []
        active = [
            m
            for m in raw_markets
            if m.get("active")
            and not m.get("closed")
            and m.get("enableOrderBook")
            and len(parse_clob_ids(m.get("clobTokenIds"))) >= 2
        ]
        if len(active) < 2:
            continue

        valid, rejection_reason = is_valid_yes_basket_arb(event, active)
        if not valid:
            # Bucket by the leading category phrase for the summary tally
            category = rejection_reason.split(" — ")[0].split(" - ")[0]
            tally[category] = tally.get(category, 0) + 1
            if verbose:
                title_short = (event.get("title") or "")[:55]
                tqdm.write(f"  REJECTED  {title_short!r:57}  {rejection_reason}")
            continue

        candidates.append((event, active))

    if verbose and tally:
        tqdm.write("\nRejection breakdown:")
        for reason, count in sorted(tally.items(), key=lambda x: -x[1]):
            tqdm.write(f"  {count:>5}  {reason}")

    return candidates


# ── CLOB order book queries ───────────────────────────────────────────────────


def fetch_best_bids(
    token_ids: list[str],
) -> dict[str, Optional[tuple[float, float]]]:
    """
    Batch-query the CLOB POST /books endpoint for the best bid per YES token.
    Returns {token_id -> (best_bid_price, bid_size)} or None if no bid side.

    Bids are returned sorted descending (highest price first), so the
    best bid is the maximum. We confirm with ``max(..., key=price)``
    defensively in case of API ordering inconsistencies.
    """
    results: dict[str, Optional[tuple[float, float]]] = {}
    total_batches = (len(token_ids) + CLOB_BATCH_SIZE - 1) // CLOB_BATCH_SIZE

    with tqdm(
        total=len(token_ids),
        desc="Querying bids",
        unit=" tokens",
        dynamic_ncols=True,
    ) as pbar:
        for i in range(0, len(token_ids), CLOB_BATCH_SIZE):
            chunk = token_ids[i : i + CLOB_BATCH_SIZE]
            batch_num = i // CLOB_BATCH_SIZE + 1
            pbar.set_postfix(batch=f"{batch_num}/{total_batches}", refresh=False)

            data = _post(f"{CLOB_API}/books", [{"token_id": t} for t in chunk])

            if not data:
                for t in chunk:
                    results[t] = None
                pbar.update(len(chunk))
                continue

            books = data if isinstance(data, list) else [data]
            for book in books:
                tid = str(book.get("asset_id") or book.get("token_id") or "")
                bids = book.get("bids") or []
                if bids:
                    best = max(bids, key=lambda b: float(b["price"]))
                    results[tid] = (float(best["price"]), float(best["size"]))
                else:
                    results[tid] = None

            pbar.update(len(chunk))
            time.sleep(REQUEST_DELAY)

    return results


def fetch_best_asks(
    token_ids: list[str],
) -> dict[str, Optional[tuple[float, float]]]:
    """
    Batch-query the CLOB POST /books endpoint for a list of YES token IDs.
    Returns {token_id -> (best_ask_price, available_shares)} or None if no liquidity.

    The /books endpoint accepts up to CLOB_BATCH_SIZE tokens per request.
    Asks are returned sorted ascending (lowest price first), so asks[0]
    is the best (cheapest) price at which someone will sell YES to you.
    """
    results: dict[str, Optional[tuple[float, float]]] = {}
    total_batches = (len(token_ids) + CLOB_BATCH_SIZE - 1) // CLOB_BATCH_SIZE

    with tqdm(
        total=len(token_ids),
        desc="Querying order books",
        unit=" tokens",
        dynamic_ncols=True,
    ) as pbar:
        for i in range(0, len(token_ids), CLOB_BATCH_SIZE):
            chunk = token_ids[i : i + CLOB_BATCH_SIZE]
            batch_num = i // CLOB_BATCH_SIZE + 1
            pbar.set_postfix(batch=f"{batch_num}/{total_batches}", refresh=False)

            data = _post(f"{CLOB_API}/books", [{"token_id": t} for t in chunk])

            if not data:
                for t in chunk:
                    results[t] = None
                pbar.update(len(chunk))
                continue

            books = data if isinstance(data, list) else [data]
            for book in books:
                tid = str(book.get("asset_id") or book.get("token_id") or "")
                asks = book.get("asks") or []
                if asks:
                    # Guard: REST API sorts ascending — confirm defensively
                    best = min(asks, key=lambda a: float(a["price"]))
                    results[tid] = (float(best["price"]), float(best["size"]))
                else:
                    results[tid] = None

            pbar.update(len(chunk))
            time.sleep(REQUEST_DELAY)

    return results


# ── Main scan ─────────────────────────────────────────────────────────────────


def scan(
    min_profit: float = 0.0,
    verbose: bool = False,
    eligible_only: bool = True,
    fee_rate: float = POLITICS_TAKER_FEE_RATE,
    strict_entry: bool = True,
) -> list[Opportunity]:
    """
    Full scan pipeline:
    1. Fetch all active events from the Gamma API.
    2. (Optional) Filter to in-scope markets (elections, sports conference
       winners, World Cup group winners) when ``eligible_only`` is True.
    3. Filter to closed-set exhaustive multi-outcome events.
    4. Batch-query CLOB order books for every YES token in one pass.
    5. Apply Polymarket's fee formula (C × feeRate × p × (1 − p)) using
       `fee_rate` (default = POLITICS_TAKER_FEE_RATE = 0.04).
    6. Return Opportunity objects whose NET profit per share (after
       Polymarket fees) is positive and >= min_profit. When ``strict_entry``
       is True (default), only opportunities that also satisfy the entry
       rules (see ``Opportunity.entry_check``) are returned.
    """
    tqdm.write("Fetching all active events from Polymarket Gamma API…")
    events = fetch_all_events()

    candidates = extract_candidates(
        events, verbose=verbose, eligible_only=eligible_only
    )
    skipped = len(events) - len(candidates)
    scope = "in-scope" if eligible_only else "all-market"
    tqdm.write(
        f"{len(candidates)} {scope} multi-outcome events qualify "
        f"({skipped} skipped: out-of-scope, single-outcome, dynamic, or no order book)."
    )

    if not candidates:
        return []

    # Collect all YES tokens from all candidate events
    token_to_event: list[tuple[dict, list[dict], list[str]]] = []
    all_yes_tokens: list[str] = []

    for event, markets in candidates:
        yes_ids = [parse_clob_ids(m.get("clobTokenIds"))[0] for m in markets]
        all_yes_tokens.extend(yes_ids)
        token_to_event.append((event, markets, yes_ids))

    total_batches = (len(all_yes_tokens) + CLOB_BATCH_SIZE - 1) // CLOB_BATCH_SIZE
    tqdm.write(
        f"Querying CLOB order books for {len(all_yes_tokens)} outcome tokens "
        f"({total_batches} batches)…"
    )
    best_asks = fetch_best_asks(all_yes_tokens)

    opportunities: list[Opportunity] = []

    for event, markets, yes_ids in token_to_event:
        outcomes: list[Outcome] = []
        skip = False

        for market, token_id in zip(markets, yes_ids):
            ask_info = best_asks.get(token_id)
            if ask_info is None:
                # No active sell orders for this outcome — can't execute the arb
                skip = True
                break
            price, size = ask_info
            outcomes.append(
                Outcome(
                    title=outcome_label(market),
                    token_id=token_id,
                    best_ask=price,
                    best_ask_size=size,
                    # Apply the user-specified taker rate. If the Gamma API
                    # advertises a higher per-market rate (rare for politics
                    # right now, but possible), we honour the larger of the
                    # two so net profit is never overstated.
                    fee_rate=max(fee_rate, parse_fee_rate(market)),
                )
            )

        if skip or not outcomes:
            continue

        total = sum(o.best_ask for o in outcomes)
        slug = event.get("slug") or ""
        opp = Opportunity(
            event_title=event.get("title") or "Unknown",
            event_url=f"https://polymarket.com/event/{slug}" if slug else "N/A",
            outcomes=outcomes,
            total_cost=total,
            event_slug=slug,
        )
        # Use NET profit (after Polymarket fees) for the threshold check.
        if opp.profit > 0 and opp.profit >= min_profit:
            opportunities.append(opp)

    if strict_entry:
        before = len(opportunities)
        opportunities = [o for o in opportunities if o.entry_check()[0]]
        dropped = before - len(opportunities)
        if dropped:
            tqdm.write(
                f"Entry-rule filter dropped {dropped} opportunit"
                f"{'y' if dropped == 1 else 'ies'} "
                f"(sum-of-asks > {MAX_IMPLIED_ODDS_SUM:.2f}, "
                f"leg < ${MIN_LEG_VALUE_USD:.2f}, total > ${MAX_TOTAL_TRADE_USD:.0f}, "
                f"or bottleneck ≤ {ENTRY_BOTTLENECK_OFFSET})."
            )

    # Sort by max extractable USD profit: combines spread size AND available liquidity
    return sorted(opportunities, key=lambda o: o.max_profit_usd, reverse=True)


# ── Output ────────────────────────────────────────────────────────────────────


def print_report(opps: list[Opportunity]) -> None:
    bar = "═" * 66
    print(f"\n{bar}")

    if not opps:
        print("  No arbitrage opportunities found.")
        print(bar)
        return

    print(f"  ARBITRAGE OPPORTUNITIES FOUND: {len(opps)}")
    print(bar)

    for rank, opp in enumerate(opps, 1):
        print(f"\n  #{rank}  {opp.event_title}")
        print(f"  URL          : {opp.event_url}")
        if opp.event_slug:
            print(f"  Slug         : {opp.event_slug}")
        print()
        print(f"  Sum of asks  : ${opp.total_cost:.4f}/share")
        print(f"  Gross profit : ${opp.gross_profit:.4f}/share")
        passes, reason = opp.entry_check()
        status = "PASS" if passes else f"FAIL — {reason}"
        print(
            f"  Entry rules  : {status}  "
            f"(buy {opp.entry_shares:,.0f} sh × {len(opp.outcomes)} legs "
            f"= ${opp.entry_total_usd:,.2f})"
        )

        # ── Dedicated fee section: formula + per-leg breakdown + totals ──
        print()
        print("  ── Fees (formula: C × feeRate × p × (1 − p)) ──")
        for o in opp.outcomes:
            label = o.title[:30]
            print(
                f"    {label:<30}  rate={o.fee_rate:.4f}  "
                f"p={o.best_ask:.4f}  fee/share=${o.fee_per_share:.6f}  "
                f"× {o.best_ask_size:,.0f} sh = ${o.fee_total_usd:,.4f}"
            )
        print(f"    {'TOTAL FEES PER SHARE':<30}  ${opp.fees_per_share:.6f}")
        print(
            f"    {'TOTAL FEES (USD)':<30}  ${opp.total_fees_usd:,.4f}  "
            f"(at {opp.max_shares:,.0f} shares × every leg)"
        )

        # ── Net result ──
        print()
        print(f"  NET profit   : ${opp.profit:.4f}/share  ({opp.profit_pct:.2f}%)")
        print(
            f"  Liquidity    : {opp.max_shares:,.0f} shares  "
            f"→  max NET profit ${opp.max_profit_usd:,.2f}"
        )

        # ── Per-leg buy table ──
        hdr = "─ Buy YES on each outcome (bottleneck leg sets max shares) "
        print(f"  ┌{hdr:─<76}┐")
        for o in opp.outcomes:
            label = o.title[:36]
            size_str = f"{o.best_ask_size:,.0f} sh"
            marker = " ◄ bottleneck" if o.best_ask_size == opp.max_shares else ""
            print(f"  │  {label:<36}  @ ${o.best_ask:.4f}  {size_str:>10}{marker:<14}│")
        print(f"  └{'─' * 78}┘")

    # ── Aggregate fee summary across ALL opportunities ──
    total_gross_usd = sum(o.gross_profit * o.max_shares for o in opps)
    total_fees_usd = sum(o.total_fees_usd for o in opps)
    total_net_usd = sum(o.max_profit_usd for o in opps)
    total_shares = sum(o.max_shares for o in opps)
    print(f"\n{bar}")
    print(f"  TOTAL FEES SUMMARY  (across all {len(opps)} opportunities)")
    print(bar)
    print(f"  Aggregate executable shares    : {total_shares:,.0f}")
    print(f"  Aggregate gross profit (USD)   : ${total_gross_usd:,.2f}")
    print(f"  Aggregate fees         (USD)   : ${total_fees_usd:,.2f}")
    print(f"  Aggregate NET profit   (USD)   : ${total_net_usd:,.2f}")
    print(bar)

    print(f"\n{bar}")
    best = opps[0]
    print(f"  #1 (most extractable): {best.event_title}")
    print(
        f"     ${best.profit:.4f}/share NET ({best.profit_pct:.2f}%)  ×  "
        f"{best.max_shares:,.0f} shares  =  ${best.max_profit_usd:,.2f} max NET profit"
    )
    print(bar)


def save_json(opps: list[Opportunity], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([o.to_dict() for o in opps], f, indent=2)
    print(f"\nResults saved → {path}")


# ── Position commands (enter / list / check-exits / exit) ─────────────────────


def print_positions(positions: list[Position]) -> None:
    """Pretty-print all known positions, open first then closed."""
    if not positions:
        print("\n  No positions on file.")
        return

    bar = "─" * 66
    open_pos = [p for p in positions if p.status == "open"]
    closed_pos = [p for p in positions if p.status != "open"]

    print(f"\n  OPEN POSITIONS: {len(open_pos)}")
    print(bar)
    for p in open_pos:
        print(f"  • {p.event_title}")
        print(
            f"      slug={p.event_slug}  opened={p.opened_at}  "
            f"shares={p.shares:,.0f}  entry=${p.entry_total_per_share:.4f}/sh  "
            f"total=${p.entry_total_usd:,.2f}"
        )
        for leg in p.legs:
            print(
                f"        - {leg.title:<30}  entry_ask=${leg.entry_price:.4f}  "
                f"fee_rate={leg.fee_rate:.4f}"
            )

    if closed_pos:
        print(f"\n  CLOSED POSITIONS: {len(closed_pos)}")
        print(bar)
        for p in closed_pos:
            edge_usd = p.edge_improvement_per_share or 0.0
            edge_pct = (p.edge_improvement_pct or 0.0) * 100
            print(
                f"  • {p.event_title}\n"
                f"      slug={p.event_slug}  closed={p.closed_at}  "
                f"shares={p.exit_shares or 0:,.0f}  "
                f"edge_improvement=${edge_usd:.4f}/sh ({edge_pct:.2f}%)"
            )


def cmd_enter(
    slug: str,
    fee_rate: float,
    eligible_only: bool,
    positions_path: Path,
) -> int:
    """Find a current opportunity matching ``slug`` and open a position.

    Re-runs the scan with strict_entry=True so the resulting Position is
    guaranteed to satisfy every rule (≤ 98% odds, > $1 per leg, ≤ $1000 total).
    Returns a shell exit code (0 = position opened, non-zero = no match).
    """
    opps = scan(
        min_profit=0.0,
        verbose=False,
        eligible_only=eligible_only,
        fee_rate=fee_rate,
        strict_entry=True,
    )
    match = next((o for o in opps if o.event_slug == slug), None)
    if match is None:
        # Fall back to substring match on the slug for ergonomic CLI use.
        match = next((o for o in opps if slug.lower() in o.event_slug.lower()), None)

    if match is None:
        print(f"\nNo enterable opportunity found for slug {slug!r}.")
        print("Re-run without --enter to see what currently passes entry rules.")
        return 1

    positions = load_positions(positions_path)
    if any(p.event_slug == match.event_slug and p.status == "open" for p in positions):
        print(
            f"\nA position is already open for {match.event_slug!r}. "
            "Use --exit first if you want to re-enter."
        )
        return 1

    pos = Position.from_opportunity(match)
    positions.append(pos)
    save_positions(positions, positions_path)
    print(
        f"\nOpened position on {match.event_title!r}: "
        f"{pos.shares:,.0f} sh × {len(pos.legs)} legs at "
        f"${pos.entry_total_per_share:.4f}/sh (${pos.entry_total_usd:,.2f} total)."
    )
    print(f"Position file: {positions_path}")
    return 0


def cmd_check_exits(positions_path: Path) -> int:
    """For every open position, fetch current asks and print exit verdict.

    Does NOT modify the positions file — pure diagnostic. Use ``--exit <slug>``
    to actually close a position.
    """
    positions = load_positions(positions_path)
    open_pos = [p for p in positions if p.status == "open"]
    if not open_pos:
        print("\n  No open positions to evaluate.")
        return 0

    all_tokens: list[str] = [leg.token_id for p in open_pos for leg in p.legs]
    print(
        f"\nChecking exit conditions for {len(open_pos)} open position(s) "
        f"({len(all_tokens)} legs)…"
    )
    books = fetch_best_bids(all_tokens)

    bar = "─" * 66
    print(bar)
    for p in open_pos:
        should_exit, reason, details = evaluate_exit(p, books)
        verdict = "EXIT" if should_exit else "HOLD"
        print(f"  [{verdict}] {p.event_title}  (slug={p.event_slug})")
        print(f"      {reason}")
        if should_exit:
            print(
                f"      exit_shares={details['exit_shares']:,.0f}  "
                f"current_total=${details['current_total_per_share']:.4f}/sh  "
                f"proceeds=${details['exit_total_usd']:,.2f}"
            )
    print(bar)
    return 0


def cmd_exit(slug: str, positions_path: Path) -> int:
    """Close the open position matching ``slug`` if exit conditions are met."""
    positions = load_positions(positions_path)
    target = next(
        (p for p in positions if p.event_slug == slug and p.status == "open"), None
    )
    if target is None:
        print(f"\nNo open position with slug {slug!r}.")
        return 1

    token_ids = [leg.token_id for leg in target.legs]
    books = fetch_best_bids(token_ids)
    should_exit, reason, details = evaluate_exit(target, books)

    if not should_exit:
        print(f"\nExit rejected for {slug!r}: {reason}")
        return 1

    close_position(target, details)
    save_positions(positions, positions_path)
    print(
        f"\nClosed position on {target.event_title!r}: "
        f"{target.exit_shares:,.0f} sh × {len(target.legs)} legs at "
        f"${target.exit_total_per_share:.4f}/sh "
        f"(${target.exit_total_usd:,.2f} proceeds).  "
        f"Edge improvement: ${target.edge_improvement_per_share:.4f}/sh "
        f"({(target.edge_improvement_pct or 0.0) * 100:.2f}%)."
    )
    return 0


# ── Paper trading (live simulator using the SAME entry/exit rules) ────────────
#
# The paper trader opens simulated positions on the highest-ranked opportunities
# from a scan and then polls the CLOB order book on a fixed cadence, printing
# the TOP 2 bids and asks per leg so the operator can see the live connection
# is healthy and prices are moving. Partial exits are applied whenever
# ``evaluate_exit`` fires — each tick can sell ``bottleneck − 2`` more shares
# until ``remaining_shares`` reaches zero.
#
# Entry/exit rules are NOT redefined here — they are reused verbatim:
#   * Entry  → ``Opportunity.entry_check`` and ``Opportunity.entry_shares``
#              (≤ 98% sum-of-asks, leg > $1, total ≤ $1000, bottleneck − 2 sh).
#   * Exit   → ``evaluate_exit`` (sum of best bids across legs ≥ entry
#              sum + $0.01/sh, bid bottleneck − 2 sh).

DEFAULT_PAPER_TOP_N = 1  # rank window: only the top-ranked opportunity (rank #1)
DEFAULT_PAPER_POLL_INTERVAL = 2.0  # seconds between order-book polls
DEFAULT_PAPER_DEPTH = 2  # top-N price levels per side to print


@dataclass
class BookQuote:
    """Full bid/ask ladder for a single token, sorted with best level first."""

    bids: list[tuple[float, float]]  # (price, size), descending by price
    asks: list[tuple[float, float]]  # (price, size), ascending by price

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0][1] if self.asks else None


@dataclass
class PaperFill:
    """A single partial-exit fill against a PaperTrade.

    ``current_bids`` are the best bid prices we sold into on each leg at
    the moment of the fill; ``sum_bids`` is their total per share.
    """

    timestamp: float
    exit_shares: float
    current_bids: list[float]
    sum_bids: float
    edge_improvement_per_share: float
    edge_improvement_pct: float


@dataclass
class PaperTrade:
    """A simulated basket position used by the paper-trading loop.

    Wraps an immutable ``Position`` (the entry state) with mutable bookkeeping
    for partial exits: ``remaining_shares`` is reduced on every fill, and the
    trade is considered closed when it hits zero.
    """

    trade_id: int
    rank: int
    position: Position
    remaining_shares: float
    fills: list[PaperFill]

    @property
    def closed(self) -> bool:
        return self.remaining_shares <= 1e-9


def fetch_book_quotes(token_ids: list[str]) -> dict[str, BookQuote]:
    """Batch-fetch full bid/ask ladders for ``token_ids`` from POST /books.

    Mirrors ``fetch_best_asks`` but keeps every price level on both sides so
    the live monitor can display top-N bids and asks. Unknown / unreturned
    tokens map to empty BookQuote (no liquidity).
    """
    results: dict[str, BookQuote] = {}
    for i in range(0, len(token_ids), CLOB_BATCH_SIZE):
        chunk = token_ids[i : i + CLOB_BATCH_SIZE]
        data = _post(f"{CLOB_API}/books", [{"token_id": t} for t in chunk])
        if not data:
            for t in chunk:
                results[t] = BookQuote(bids=[], asks=[])
            continue

        books = data if isinstance(data, list) else [data]
        seen: set[str] = set()
        for book in books:
            tid = str(book.get("asset_id") or book.get("token_id") or "")
            seen.add(tid)
            bids = sorted(
                (
                    (float(b["price"]), float(b["size"]))
                    for b in (book.get("bids") or [])
                ),
                key=lambda kv: -kv[0],
            )
            asks = sorted(
                (
                    (float(a["price"]), float(a["size"]))
                    for a in (book.get("asks") or [])
                ),
                key=lambda kv: kv[0],
            )
            results[tid] = BookQuote(bids=bids, asks=asks)

        for t in chunk:
            if t not in seen:
                results[t] = BookQuote(bids=[], asks=[])
        time.sleep(REQUEST_DELAY)
    return results


def select_top_opportunities(
    opportunities: list[Opportunity],
    top_n: int = DEFAULT_PAPER_TOP_N,
    skip_band: tuple[int, int] = (7, 17),
) -> list[tuple[int, Opportunity]]:
    """Return ``(rank, opp)`` pairs for the top-N, explicitly skipping a band.

    Ranking matches the scan output (``max_profit_usd`` descending). The
    default config — top_n=6, skip_band=(7, 17) — yields ranks 1–6. The
    skip band is honoured exactly as requested in the spec so changing
    ``top_n`` later still skips ranks #7–#17.
    """
    lo, hi = skip_band
    return [
        (rank, opp)
        for rank, opp in enumerate(opportunities, start=1)
        if rank <= top_n and not (lo <= rank <= hi)
    ]


def open_paper_trade(rank: int, trade_id: int, opp: Opportunity) -> PaperTrade:
    """Create a PaperTrade for ``opp``. Caller must have verified entry rules."""
    position = Position.from_opportunity(opp)
    return PaperTrade(
        trade_id=trade_id,
        rank=rank,
        position=position,
        remaining_shares=position.shares,
        fills=[],
    )


def _books_to_bid_map(
    legs: list[PositionLeg], books: Mapping[str, BookQuote]
) -> dict[str, Optional[tuple[float, float]]]:
    """Project full ladders down to the (best_bid, best_bid_size) form that
    ``evaluate_exit`` expects. A leg with no bid side maps to None."""
    bid_map: dict[str, Optional[tuple[float, float]]] = {}
    for leg in legs:
        quote = books.get(leg.token_id)
        if quote is None or quote.best_bid is None or not quote.bids:
            bid_map[leg.token_id] = None
        else:
            bid_map[leg.token_id] = (quote.best_bid, quote.bids[0][1])
    return bid_map


def evaluate_paper_exit(
    trade: PaperTrade, books: Mapping[str, BookQuote]
) -> tuple[bool, str, dict]:
    """Reuse ``evaluate_exit`` against ``trade``'s remaining shares.

    A scratch Position is built with ``shares = remaining_shares`` so the
    bottleneck cap inside ``evaluate_exit`` clamps to the still-held size.
    """
    if trade.closed:
        return False, "trade already fully exited", {}
    bid_map = _books_to_bid_map(trade.position.legs, books)
    scratch = Position(
        event_slug=trade.position.event_slug,
        event_title=trade.position.event_title,
        event_url=trade.position.event_url,
        opened_at=trade.position.opened_at,
        shares=trade.remaining_shares,
        legs=trade.position.legs,
        entry_total_per_share=trade.position.entry_total_per_share,
        entry_total_usd=trade.position.entry_total_usd,
    )
    return evaluate_exit(scratch, bid_map)


def apply_paper_exit(trade: PaperTrade, details: dict) -> PaperFill:
    """Record a partial fill on ``trade`` and decrement remaining shares."""
    fill = PaperFill(
        timestamp=time.time(),
        exit_shares=float(details["exit_shares"]),
        current_bids=list(details["current_bids"]),
        sum_bids=float(details["current_total_per_share"]),
        edge_improvement_per_share=float(details["edge_improvement_per_share"]),
        edge_improvement_pct=float(details["edge_improvement_pct"]),
    )
    trade.fills.append(fill)
    trade.remaining_shares = max(0.0, trade.remaining_shares - fill.exit_shares)
    return fill


def _fmt_ladder(levels: list[tuple[float, float]], depth: int) -> str:
    """Render ``depth`` (price, size) pairs as 'price×size' columns, padded."""
    parts: list[str] = []
    for price, size in levels[:depth]:
        parts.append(f"${price:.4f}×{size:>6,.0f}")
    while len(parts) < depth:
        parts.append("  —.————×     —")
    return "  ".join(parts)


def print_paper_orderbook(
    trade: PaperTrade,
    books: Mapping[str, BookQuote],
    depth: int = DEFAULT_PAPER_DEPTH,
) -> None:
    """Print top-``depth`` bids and asks for every leg of ``trade``.

    The output proves we are currently connected to Polymarket (the prices
    move tick-to-tick) and surfaces enough of the book to see whether the
    bottleneck has enough size for another partial exit.
    """
    status = "CLOSED" if trade.closed else "OPEN"
    print(
        f"  [trade #{trade.trade_id}] {status}  rank={trade.rank}  "
        f"{trade.position.event_title[:60]}"
    )
    print(
        f"    entry_total=${trade.position.entry_total_per_share:.4f}/sh  "
        f"entry_size={trade.position.shares:,.0f}sh  "
        f"remaining={trade.remaining_shares:,.0f}sh  "
        f"fills={len(trade.fills)}"
    )
    label_w = 20
    for leg in trade.position.legs:
        quote = books.get(leg.token_id) or BookQuote(bids=[], asks=[])
        asks_str = _fmt_ladder(quote.asks, depth)
        bids_str = _fmt_ladder(quote.bids, depth)
        gate_bid = (
            "↑"
            if (quote.best_bid is not None and quote.best_bid > leg.entry_price)
            else "·"
        )
        print(
            f"    {leg.title[:label_w]:<{label_w}}  "
            f"entry=${leg.entry_price:.4f} {gate_bid}  "
            f"ASKS [{asks_str}]  BIDS [{bids_str}]"
        )


def _entry_event(trade: PaperTrade) -> dict:
    """Build the JSON record written when a paper trade is opened."""
    return {
        "type": "entry",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trade_id": trade.trade_id,
        "rank": trade.rank,
        "event_slug": trade.position.event_slug,
        "event_title": trade.position.event_title,
        "event_url": trade.position.event_url,
        "shares": trade.position.shares,
        "entry_total_per_share": trade.position.entry_total_per_share,
        "entry_total_usd": trade.position.entry_total_usd,
        "legs": [asdict(leg) for leg in trade.position.legs],
    }


def _exit_event(trade: PaperTrade, fill: PaperFill) -> dict:
    """Build the JSON record written when a partial exit fill is applied."""
    return {
        "type": "exit",
        "timestamp": datetime.fromtimestamp(fill.timestamp, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        "trade_id": trade.trade_id,
        "rank": trade.rank,
        "event_slug": trade.position.event_slug,
        "event_title": trade.position.event_title,
        "exit_shares": fill.exit_shares,
        "current_bids": list(fill.current_bids),
        "sum_bids": fill.sum_bids,
        "edge_improvement_per_share": fill.edge_improvement_per_share,
        "edge_improvement_pct": fill.edge_improvement_pct,
        "remaining_shares": trade.remaining_shares,
        "fully_closed": trade.closed,
    }


def run_paper_trading(
    opportunities: list[Opportunity],
    top_n: int = DEFAULT_PAPER_TOP_N,
    skip_band: tuple[int, int] = (7, 17),
    poll_interval: float = DEFAULT_PAPER_POLL_INTERVAL,
    display_depth: int = DEFAULT_PAPER_DEPTH,
    max_iterations: Optional[int] = None,
    trades_log_path: Optional[Path] = DEFAULT_PAPER_TRADES_LOG,
) -> list[PaperTrade]:
    """Open paper trades on the selected rank window, then poll until closed.

    On every tick the loop:
      1. fetches full bid/ask ladders for every leg still in play,
      2. prints the top ``display_depth`` levels per side so the user can
         confirm the connection is live and watch prices move,
      3. runs ``evaluate_paper_exit`` per open trade and applies a partial
         exit (``apply_paper_exit``) whenever the rules fire.

    Every entry and every partial-exit fill is appended to
    ``trades_log_path`` (pass ``None`` to disable logging). Each partial
    fill produces its own ``"exit"`` record; the final fill carries
    ``fully_closed: true``.

    Returns the full list of PaperTrade objects (open + closed) at end of run.
    """
    selected = select_top_opportunities(opportunities, top_n, skip_band)
    if not selected:
        print(
            f"[paper trading] no opportunities in scope "
            f"(top_n={top_n}, skip_band=#{skip_band[0]}–#{skip_band[1]}, "
            f"have {len(opportunities)} total)"
        )
        return []

    trades: list[PaperTrade] = []
    for trade_id, (rank, opp) in enumerate(selected, start=1):
        passes, reason = opp.entry_check()
        if not passes:
            print(
                f"[paper trading] skip rank #{rank} {opp.event_title[:50]!r} — "
                f"entry rules fail ({reason})"
            )
            continue
        trade = open_paper_trade(rank, trade_id, opp)
        trades.append(trade)
        if trades_log_path is not None:
            append_paper_trade_event(_entry_event(trade), trades_log_path)

    if not trades:
        print("[paper trading] no opportunities passed entry rules — nothing to do")
        return []

    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  PAPER TRADING — opened {len(trades)} positions (ranks #1–#{top_n})")
    print(bar)
    for t in trades:
        print(
            f"  trade #{t.trade_id}  rank={t.rank}  shares={t.remaining_shares:,.0f}  "
            f"entry=${t.position.entry_total_per_share:.4f}/sh  "
            f"({t.position.event_title[:42]})"
        )
    print(bar)

    iteration = 0
    while not all(t.closed for t in trades):
        if max_iterations is not None and iteration >= max_iterations:
            print(f"\n[paper trading] stopped after {iteration} ticks (max reached)")
            break
        iteration += 1

        token_ids = sorted(
            {leg.token_id for t in trades if not t.closed for leg in t.position.legs}
        )
        books = fetch_book_quotes(token_ids)

        print(
            f"\n── tick {iteration} @ {time.strftime('%H:%M:%S')} "
            f"(polling Polymarket CLOB, top {display_depth} bids/asks) " + "─" * 14
        )
        for trade in trades:
            if trade.closed:
                continue
            print_paper_orderbook(trade, books, depth=display_depth)

            should_exit, reason, details = evaluate_paper_exit(trade, books)
            if should_exit:
                fill = apply_paper_exit(trade, details)
                if trades_log_path is not None:
                    append_paper_trade_event(_exit_event(trade, fill), trades_log_path)
                print(
                    f"    >>> EXIT FILL: sold {fill.exit_shares:,.0f} sh "
                    f"@ sum_bids=${fill.sum_bids:.4f}  "
                    f"edge=+${fill.edge_improvement_per_share:.4f}/sh "
                    f"({fill.edge_improvement_pct * 100:.2f}%)  "
                    f"remaining={trade.remaining_shares:,.0f} sh"
                )
            else:
                print(f"    hold — {reason}")

        if not all(t.closed for t in trades):
            time.sleep(poll_interval)

    print(f"\n{bar}")
    print(f"  PAPER TRADING SUMMARY  ({iteration} ticks)")
    print(bar)
    for t in trades:
        status = "CLOSED" if t.closed else "OPEN"
        sold = t.position.shares - t.remaining_shares
        avg_exit = (
            sum(f.exit_shares * f.sum_bids for f in t.fills) / sold if sold > 0 else 0.0
        )
        edge = avg_exit - t.position.entry_total_per_share if sold > 0 else 0.0
        print(
            f"  [{status}] #{t.trade_id} rank={t.rank}  "
            f"sold {sold:,.0f}/{t.position.shares:,.0f} sh  "
            f"avg_exit=${avg_exit:.4f}/sh  "
            f"edge=+${edge:.4f}/sh  "
            f"({t.position.event_title[:38]})"
        )
    print(bar)
    return trades


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Polymarket eligible markets (elections, sports conference "
            "winners, World Cup group winners) for multi-outcome arbitrage "
            "opportunities (net of Polymarket protocol fees), and manage "
            "open basket positions."
        )
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.0,
        metavar="USD",
        help="Minimum NET profit per share (after fees) to report — default: any > $0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save scan results to a JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log each rejected market and show a rejection-reason breakdown",
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Disable the eligible-scope filter and scan every multi-outcome event",
    )
    parser.add_argument(
        "--no-strict-entry",
        action="store_true",
        help=(
            "Show all positive-profit opportunities even if they violate the "
            "entry rules (sum_asks ≤ 0.98, leg > $1, total ≤ $1000, bottleneck > 2)."
        ),
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=POLITICS_TAKER_FEE_RATE,
        metavar="RATE",
        help=(
            f"Polymarket taker fee rate as a decimal "
            f"(default {POLITICS_TAKER_FEE_RATE} for politics). "
            f"Per-share fee = rate * p * (1 - p)."
        ),
    )
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=DEFAULT_POSITIONS_FILE,
        metavar="PATH",
        help=f"Path to the positions JSON store (default: {DEFAULT_POSITIONS_FILE}).",
    )

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--enter",
        type=str,
        metavar="SLUG",
        help="Open a basket position for the event with the given slug.",
    )
    actions.add_argument(
        "--exit",
        dest="exit_slug",
        type=str,
        metavar="SLUG",
        help="Close the open position for the event with the given slug.",
    )
    actions.add_argument(
        "--check-exits",
        action="store_true",
        help="Evaluate exit conditions for every open position (read-only).",
    )
    actions.add_argument(
        "--positions",
        action="store_true",
        help="List all open and closed positions on file.",
    )
    actions.add_argument(
        "--paper-trade",
        action="store_true",
        help=(
            "Run the paper trader: scan, open a simulated position on the "
            f"top {DEFAULT_PAPER_TOP_N} rank (default: rank #1 only) and poll "
            "the order book, printing the top-2 bids/asks per leg every tick "
            "until the trade is fully exited."
        ),
    )
    parser.add_argument(
        "--paper-top-n",
        type=int,
        default=DEFAULT_PAPER_TOP_N,
        metavar="N",
        help=(
            f"How many of the top ranks to paper trade "
            f"(default: {DEFAULT_PAPER_TOP_N} — just rank #1). Ranks #7–#17 "
            f"are always skipped if N is raised above 6."
        ),
    )
    parser.add_argument(
        "--paper-poll-interval",
        type=float,
        default=DEFAULT_PAPER_POLL_INTERVAL,
        metavar="SECONDS",
        help=(
            f"Order-book polling cadence for paper trading "
            f"(default: {DEFAULT_PAPER_POLL_INTERVAL}s)."
        ),
    )
    parser.add_argument(
        "--paper-max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Stop the paper-trade loop after N ticks (default: run until all closed).",
    )
    parser.add_argument(
        "--paper-trades-file",
        type=Path,
        default=DEFAULT_PAPER_TRADES_LOG,
        metavar="PATH",
        help=(
            "Append-only JSON log of paper-trade entries and exit fills "
            f"(default: {DEFAULT_PAPER_TRADES_LOG}). Pass an empty string to "
            "disable logging."
        ),
    )

    args = parser.parse_args()

    if args.positions:
        print_positions(load_positions(args.positions_file))
        return

    if args.enter:
        sys.exit(
            cmd_enter(
                slug=args.enter,
                fee_rate=args.fee_rate,
                eligible_only=not args.all_markets,
                positions_path=args.positions_file,
            )
        )

    if args.exit_slug:
        sys.exit(cmd_exit(args.exit_slug, args.positions_file))

    if args.check_exits:
        sys.exit(cmd_check_exits(args.positions_file))

    if args.paper_trade:
        print("[paper trading] running scan to identify opportunities…")
        opps = scan(
            min_profit=args.min_profit,
            verbose=args.verbose,
            eligible_only=not args.all_markets,
            fee_rate=args.fee_rate,
            strict_entry=True,
        )
        if not opps:
            print("[paper trading] no enterable opportunities — exiting.")
            return
        try:
            run_paper_trading(
                opps,
                top_n=args.paper_top_n,
                poll_interval=args.paper_poll_interval,
                max_iterations=args.paper_max_iterations,
                trades_log_path=args.paper_trades_file or None,
            )
        except KeyboardInterrupt:
            print("\n[paper trading] interrupted by user")
        return

    # Default action: run a scan.
    t0 = time.perf_counter()
    opps = scan(
        min_profit=args.min_profit,
        verbose=args.verbose,
        eligible_only=not args.all_markets,
        fee_rate=args.fee_rate,
        strict_entry=not args.no_strict_entry,
    )
    elapsed = time.perf_counter() - t0

    print_report(opps)

    if args.output:
        save_json(opps, args.output)

    plural = "y" if len(opps) == 1 else "ies"
    print(
        f"\nScan completed in {elapsed:.1f}s  |  {len(opps)} opportunit{plural} found"
    )

    if opps:
        try:
            run_paper_trading(
                opps,
                top_n=1,
                poll_interval=DEFAULT_PAPER_POLL_INTERVAL,
                trades_log_path=args.paper_trades_file or None,
            )
        except KeyboardInterrupt:
            print("\n[paper trading] interrupted by user")


if __name__ == "__main__":
    main()
