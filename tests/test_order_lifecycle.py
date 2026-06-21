"""Tests for order lifecycle helpers (execution/order_manager.py): build + await_fill."""

from __future__ import annotations

import asyncio

from execution import order_manager


def _request() -> order_manager.OrderRequest:
    return order_manager.OrderRequest(
        ticker="KXWC26-BRA", action="buy", side="yes", count=10, limit_price_cents=50
    )


def test_build_order_row() -> None:
    row = order_manager.build_order_row(
        order_id="ord1", signal_id=7, request=_request()
    )
    assert row["id"] == "ord1"
    assert row["signal_id"] == 7
    assert row["contracts"] == 10
    assert row["limit_price"] == 0.50
    assert row["status"] == "pending"
    assert row["pnl_cents"] is None


def test_await_fill_returns_filled_without_cancel() -> None:
    cancelled: list[str] = []

    async def status_fn(_: str) -> str:
        return "filled"

    async def cancel_fn(order_id: str) -> None:
        cancelled.append(order_id)

    async def sleeper(_: int) -> None:
        return None

    result = asyncio.run(
        order_manager.await_fill(
            "ord1", status_fn=status_fn, cancel_fn=cancel_fn, sleeper=sleeper
        )
    )
    assert result == "filled"
    assert cancelled == []


def test_await_fill_times_out_and_cancels() -> None:
    cancelled: list[str] = []

    async def status_fn(_: str) -> str:
        return "resting"

    async def cancel_fn(order_id: str) -> None:
        cancelled.append(order_id)

    async def sleeper(_: int) -> None:
        return None

    result = asyncio.run(
        order_manager.await_fill(
            "ord1",
            status_fn=status_fn,
            cancel_fn=cancel_fn,
            sleeper=sleeper,
            timeout_s=2,
            interval_s=1,
        )
    )
    assert result == "timeout"
    assert cancelled == ["ord1"]
