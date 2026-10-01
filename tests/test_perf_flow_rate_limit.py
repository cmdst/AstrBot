"""Performance regression tests for the rate-limit stage (contract items F1 / I3).

F1: the wait of a rate-limited event must happen outside the per-session lock,
so one limited event cannot block the rest of the session.
I3: the STALL / DISCARD semantics must stay exactly as before.
"""

import asyncio
import time
from datetime import timedelta

import pytest

from astrbot.core.config.astrbot_config import RateLimitStrategy
from astrbot.core.pipeline.rate_limit_check import stage as rate_limit_stage
from astrbot.core.pipeline.rate_limit_check.stage import RateLimitStage


class FakeEvent:
    """Minimal stand-in for AstrMessageEvent, exposing only what the stage uses."""

    def __init__(self, umo: str) -> None:
        self.unified_msg_origin = umo
        self.stopped = False

    def stop_event(self) -> "FakeEvent":
        """Mark the event as stopped and return it, mirroring the real event."""
        self.stopped = True
        return self


def make_stage(strategy: str, count: int, window_seconds: float) -> RateLimitStage:
    """Build a stage with an explicit limit configuration, bypassing initialize()."""
    stage = RateLimitStage()
    stage.rate_limit_count = count
    stage.rate_limit_time = timedelta(seconds=window_seconds)
    stage.rl_strategy = strategy
    return stage


def park_sleep(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Replace the stage's ``asyncio.sleep`` by a parked wait and report entry.

    Args:
        monkeypatch: Active pytest monkeypatch fixture.

    Returns:
        An event that is set as soon as a rate-limited event starts waiting.
    """
    real_sleep = rate_limit_stage.asyncio.sleep
    waiting = asyncio.Event()

    async def fake_sleep(seconds: float) -> None:  # noqa: ARG001
        waiting.set()
        await real_sleep(3600)

    monkeypatch.setattr(rate_limit_stage.asyncio, "sleep", fake_sleep)
    return waiting


@pytest.mark.asyncio
async def test_unlimited_session_records_nothing():
    """count <= 0 keeps the stage a no-op and never records a timestamp."""
    stage = make_stage(RateLimitStrategy.STALL.value, 0, 60)
    umo = "platform:GroupMessage:1"

    assert await stage.process(FakeEvent(umo)) is None
    assert len(stage.event_timestamps[umo]) == 0
    assert not stage.locks[umo].locked()


@pytest.mark.asyncio
async def test_stall_still_delays_and_admits_the_limited_event():
    """I3: STALL waits for the window and then admits the event (no dropping)."""
    stage = make_stage(RateLimitStrategy.STALL.value, 1, 0.1)
    umo = "platform:GroupMessage:2"

    await stage.process(FakeEvent(umo))

    event = FakeEvent(umo)
    started = time.perf_counter()
    assert await stage.process(event) is None
    elapsed = time.perf_counter() - started

    # The fixed window (0.1 s) plus the intentional 0.3 s grace of the stage.
    assert elapsed >= 0.25, f"rate-limited event was not delayed ({elapsed:.3f}s)"
    assert event.stopped is False, "STALL must not drop the event"
    assert len(stage.event_timestamps[umo]) == 1


@pytest.mark.asyncio
async def test_discard_still_drops_the_limited_event_without_waiting():
    """I3: DISCARD stops the event immediately and records nothing."""
    stage = make_stage(RateLimitStrategy.DISCARD.value, 1, 3600)
    umo = "platform:GroupMessage:3"

    await stage.process(FakeEvent(umo))

    event = FakeEvent(umo)
    started = time.perf_counter()
    await stage.process(event)
    elapsed = time.perf_counter() - started

    assert event.stopped is True, "DISCARD must stop the event"
    assert elapsed < 0.5, f"DISCARD must not wait ({elapsed:.3f}s)"
    assert len(stage.event_timestamps[umo]) == 1


@pytest.mark.asyncio
async def test_sleep_happens_outside_the_session_lock(monkeypatch):
    """F1: the session lock is already released when a limited event waits."""
    stage = make_stage(RateLimitStrategy.STALL.value, 1, 3600)
    umo = "platform:GroupMessage:4"
    await stage.process(FakeEvent(umo))  # fill the window

    waiting = park_sleep(monkeypatch)
    task = asyncio.create_task(stage.process(FakeEvent(umo)))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2.0)
        assert not stage.locks[umo].locked(), (
            "the session lock is held while a limited event waits"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_one_event_per_session_waits_at_a_time(monkeypatch):
    """F1: the waiting event owns only the stall gate, never the counting lock.

    The fixed window keeps its FIFO admission order: a second rate-limited event
    of the same session queues behind the first waiter instead of starting its
    own sleep. The counting lock stays free for decision-path work.
    """
    stage = make_stage(RateLimitStrategy.STALL.value, 1, 3600)
    umo = "platform:GroupMessage:5"
    await stage.process(FakeEvent(umo))  # fill the window

    real_sleep = rate_limit_stage.asyncio.sleep
    waiting: set[asyncio.Task] = set()
    entered_wait = asyncio.Event()

    async def fake_sleep(seconds: float) -> None:  # noqa: ARG001
        waiting.add(asyncio.current_task())
        entered_wait.set()
        await real_sleep(3600)

    monkeypatch.setattr(rate_limit_stage.asyncio, "sleep", fake_sleep)
    task_a = asyncio.create_task(stage.process(FakeEvent(umo)))
    await asyncio.wait_for(entered_wait.wait(), timeout=2.0)
    task_b = asyncio.create_task(stage.process(FakeEvent(umo)))
    try:
        # Give the second event every chance to enter a sleep of its own.
        for _ in range(20):
            await real_sleep(0.01)
        assert len(waiting) == 1, "two events of one session slept at the same time"
        assert not stage.locks[umo].locked(), "the counting lock is held while waiting"
    finally:
        task_a.cancel()
        task_b.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_window_still_admits_at_most_count_events(monkeypatch):
    """I3: the fixed window admits `count` events and stalls the next one."""
    stage = make_stage(RateLimitStrategy.STALL.value, 2, 3600)
    umo = "platform:GroupMessage:6"

    await stage.process(FakeEvent(umo))
    await stage.process(FakeEvent(umo))
    assert len(stage.event_timestamps[umo]) == 2

    waiting = park_sleep(monkeypatch)
    task = asyncio.create_task(stage.process(FakeEvent(umo)))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2.0)
        assert len(stage.event_timestamps[umo]) == 2, (
            "a third event was recorded inside a full window"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.undo()
