"""Tests for the fixture board: analysis collection, persistence, and thesis rendering.

The board's contract: every leg of every fixture gets a row each cycle with its full
probability breakdown and a decision explaining what happened to it — so the dashboard
can show model-vs-Kalshi discrepancies even where nothing was bet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

from data import db
from dashboard.app import explain_analysis, render
from execution.portfolio import PortfolioState
from features import elo
from ingestion.api_football import Fixture
from schemas import LegAnalysis
from strategy import signal_gen


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-01", "2024-09-01"]),
            "home_team": ["Brazil", "Serbia"],
            "away_team": ["Serbia", "Brazil"],
            "fthg": [2, 0],
            "ftag": [0, 1],
            "ftr": ["H", "A"],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [False, False],
        }
    )


def _fixture() -> Fixture:
    return Fixture(
        fixture_id=999,
        kickoff_utc="2026-06-20T18:00:00+00:00",
        status="NS",
        home_team="Brazil",
        away_team="Serbia",
        round="Group G",
    )


def _markets(home: int, draw: int, away: int) -> list[dict]:
    def ask(cents: int) -> str:
        return f"{cents / 100:.4f}"

    return [
        {
            "ticker": f"KXWC26-BRA-{leg}",
            "title": "Brazil vs Serbia",
            "yes_sub_title": sub,
            "yes_ask_dollars": ask(price),
            "open_interest_fp": "10000.00",
        }
        for leg, sub, price in (
            ("H", "Brazil", home),
            ("D", "Draw", draw),
            ("A", "Serbia", away),
        )
    ]


_BOOK = {
    frozenset({"Brazil", "Serbia"}): {"Brazil": 0.60, "Draw": 0.25, "Serbia": 0.15}
}


def _generate(analysis: list[LegAnalysis] | None, **overrides) -> list:
    history = _history()
    kwargs: dict = dict(
        fixtures=[_fixture()],
        history=history,
        ratings=elo.final_ratings(history, use_tournament_k=True),
        bundle={},
        markets=_markets(48, 30, 22),
        bankroll_cents=20000,
        threshold=0.04,
        book_probs_by_pair=_BOOK,
        predictor=lambda _f, _feat: {"H": 0.55, "D": 0.27, "A": 0.18},
        analysis=analysis,
    )
    kwargs.update(overrides)
    return signal_gen.generate_signals(**kwargs)


def test_analysis_rows_cover_every_leg_with_decisions() -> None:
    rows: list[LegAnalysis] = []
    signals = _generate(rows)
    assert len(signals) == 1  # H is the book-vs-Kalshi divergence bet
    assert len(rows) == 3  # one row per leg, bet or not
    by_leg = {row["leg"]: row for row in rows}
    assert by_leg["H"]["decision"] == "signal"
    assert by_leg["D"]["decision"] == "below_threshold"
    assert by_leg["A"]["decision"] == "below_threshold"
    # The breakdown fields carry the pipeline stages for the traded leg.
    h = by_leg["H"]
    assert h["anchor_source"] == "book"
    assert h["raw_model_prob"] is not None
    assert h["tilted_prob"] is not None
    assert h["blended_prob"] is not None
    assert h["kalshi_price"] == 0.48
    assert h["edge"] is not None and h["edge"] > 0.04


def test_analysis_marks_held_and_no_market() -> None:
    rows: list[LegAnalysis] = []
    _generate(rows, held_tickers={"KXWC26-BRA-H"})
    by_leg = {row["leg"]: row for row in rows}
    assert by_leg["H"]["decision"] == "held"

    rows2: list[LegAnalysis] = []
    _generate(rows2, markets=[{"ticker": "X", "title": "France vs Spain"}])
    assert len(rows2) == 1
    assert rows2[0]["leg"] is None
    assert rows2[0]["decision"] == "no_market"


def test_analysis_marks_filtered_legs() -> None:
    # Underdog leg overpriced vs the line (blended/price > 1.6) -> ratio guard reason.
    rows: list[LegAnalysis] = []
    _generate(
        rows,
        markets=_markets(50, 40, 10),
        book_probs_by_pair={
            frozenset({"Brazil", "Serbia"}): {
                "Brazil": 0.60,
                "Draw": 0.22,
                "Serbia": 0.18,
            }
        },
    )
    by_leg = {row["leg"]: row for row in rows}
    assert by_leg["A"]["decision"] == "filtered:model_market_mismatch"


def test_analysis_none_skips_collection() -> None:
    signals = _generate(None)
    assert len(signals) == 1  # behaviour identical without a collector


def test_db_roundtrip_latest_cycle_and_ticker_lookup(tmp_path: Path) -> None:
    db_file = tmp_path / "test.sqlite"
    db.init_db(db_file)

    def row(cycle: datetime, ticker: str, decision: str) -> LegAnalysis:
        return LegAnalysis(
            cycle_ts=cycle,
            fixture_id="999",
            home_team="Brazil",
            away_team="Serbia",
            kickoff_utc="2026-06-20T18:00:00+00:00",
            leg="H",
            ticker=ticker,
            kalshi_price=0.48,
            raw_model_prob=0.55,
            tilted_prob=0.56,
            anchor_prob=0.60,
            anchor_source="book",
            blended_prob=0.585,
            edge=0.105,
            decision=decision,
        )

    t1 = datetime(2026, 7, 3, 3, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 7, 3, 6, 0, tzinfo=timezone.utc)
    with db.connect(db_file) as conn:
        db.log_fixture_analysis(conn, [row(t1, "TKT-H", "signal")])
        db.log_fixture_analysis(conn, [row(t2, "TKT-H", "held")])

        latest = db.latest_analysis(conn)
        assert len(latest) == 1
        assert latest[0]["decision"] == "held"  # only the newest cycle

        # Thesis lookup wants the cycle that PLACED the bet, not later held rows.
        placed = db.analysis_for_ticker(conn, "TKT-H")
        assert placed is not None
        assert placed["decision"] == "signal"
        assert db.analysis_for_ticker(conn, "UNKNOWN") is None


def test_explain_analysis_one_liner() -> None:
    row = {
        "leg": "H",
        "home_team": "Portugal",
        "away_team": "Ghana",
        "tilted_prob": 0.55,
        "anchor_prob": 0.58,
        "anchor_source": "book",
        "kalshi_price": 0.48,
        "edge": 0.095,
    }
    text = explain_analysis(row, bet_size_cents=120)
    assert "Portugal to beat Ghana" in text
    assert "model 55%" in text
    assert "books 58%" in text
    assert "Kalshi 48c" in text
    assert "+9.5% edge" in text
    assert "$1.20" in text
    # Kalshi-anchored fallback is called out so a missing book line is visible.
    row["anchor_source"] = "kalshi"
    assert "Kalshi-anchored" in explain_analysis(row)


def test_render_includes_fixture_board() -> None:
    state = PortfolioState(bankroll_cents=5000, peak_bankroll_cents=5000)
    board = [
        {
            "cycle_ts": "2026-07-03T06:00:00+00:00",
            "home_team": "Brazil",
            "away_team": "Serbia",
            "leg": "H",
            "tilted_prob": 0.56,
            "anchor_prob": 0.60,
            "anchor_source": "book",
            "blended_prob": 0.585,
            "kalshi_price": 0.48,
            "edge": 0.105,
            "decision": "signal",
        },
        {
            "cycle_ts": "2026-07-03T06:00:00+00:00",
            "home_team": "France",
            "away_team": "Spain",
            "leg": None,
            "tilted_prob": None,
            "anchor_prob": None,
            "anchor_source": None,
            "blended_prob": None,
            "kalshi_price": None,
            "edge": None,
            "decision": "no_market",
        },
    ]
    console = Console(record=True, width=200)
    render(state, [], board=board, console=console)
    text = console.export_text()
    assert "FIXTURE BOARD" in text
    assert "Brazil vs Serbia" in text
    assert "no Kalshi market" in text
    assert "signal" in text
