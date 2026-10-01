"""Performance regression tests for the pipeline stages (F3 / F4) and the
stage-level slow-call instrumentation (S2).

F3: path-mapping regexes are compiled once per stage instead of per mapping.
F4: ``waking_check`` must not bind ``default={}`` to ``get_extra(key)``.
S2: only stages above the slow threshold may log, i.e. +0 log lines per message.
"""

import ast
import asyncio
import pathlib
import re
from unittest.mock import patch

import pytest

from astrbot.core.pipeline import scheduler as scheduler_module
from astrbot.core.pipeline.preprocess_stage.stage import (
    DRIVE_SOURCE_MAPPING_PATTERN,
    DRIVE_TARGET_MAPPING_PATTERN,
)
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE = REPO_ROOT / "astrbot/core/pipeline/preprocess_stage/stage.py"
WAKING_CHECK_SOURCE = REPO_ROOT / "astrbot/core/pipeline/waking_check/stage.py"

PATH_MAPPINGS = [
    "C:\\data\\images:/var/lib/astrbot/images",  # drive-lettered source
    "/var/lib/astrbot/images:C:\\data\\images",  # drive-lettered target
    "/srv/audio:/var/lib/astrbot/audio",  # plain mapping
    "D:\\media:relative-target",
]


def test_path_mapping_patterns_are_precompiled():
    """F3: both patterns are compiled once at import, not per mapping."""
    assert isinstance(DRIVE_SOURCE_MAPPING_PATTERN, re.Pattern)
    assert isinstance(DRIVE_TARGET_MAPPING_PATTERN, re.Pattern)
    source = PREPROCESS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    per_call = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fullmatch"
        and getattr(node.func.value, "id", "") == "re"
    ]
    assert per_call == [], f"re.fullmatch still called per mapping at {per_call}"


def test_path_mapping_patterns_match_like_the_inline_literals():
    """F3: the compiled patterns are behaviourally identical to the literals."""
    source_pattern = r"([A-Za-z]:[^:]+):(.+)"
    target_pattern = r"(.+):([A-Za-z]:.+)"

    for mapping in PATH_MAPPINGS + ["no-separator", "C:\\only:a", ":C:\\only"]:
        inline_source = re.fullmatch(source_pattern, mapping)
        inline_target = re.fullmatch(target_pattern, mapping)
        cached_source = DRIVE_SOURCE_MAPPING_PATTERN.fullmatch(mapping)
        cached_target = DRIVE_TARGET_MAPPING_PATTERN.fullmatch(mapping)

        assert (cached_source is None) == (inline_source is None)
        assert (cached_target is None) == (inline_target is None)
        if inline_source is not None:
            assert cached_source.groups() == inline_source.groups()
        if inline_target is not None:
            assert cached_target.groups() == inline_target.groups()


def test_waking_check_does_not_bind_arguments_to_get_extra():
    """F4: no ``get_extra`` call in the stage passes keyword arguments."""
    source = WAKING_CHECK_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bound = [
        (node.lineno, [keyword.arg for keyword in node.keywords])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "get_extra"
        and node.keywords
    ]
    assert bound == [], f"get_extra still called with keywords: {bound}"
    assert "get_extra(default={})" not in source


def test_get_extra_without_key_returns_the_mapping():
    """F4: the fixed expression is equivalent to the previous one, and safer.

    ``get_extra(default={})`` binds by keyword to the real ``default`` parameter
    of ``AstrMessageEvent.get_extra``, so the previous call raised nothing: it
    also took the ``key is None`` branch. The new call therefore changes no
    behaviour for the shipped event class; it only stops relying on the optional
    ``default`` parameter, which duck-typed events (``get_extra(self, key)``)
    do not always accept.
    """

    class Carrier:
        _extras = {"parsed_params": {"city": "Beijing"}}
        get_extra = AstrMessageEvent.get_extra

    carrier = Carrier()
    assert "parsed_params" in carrier.get_extra()
    assert carrier.get_extra("parsed_params") == {"city": "Beijing"}
    assert "parsed_params" in carrier.get_extra(default={})  # the previous form

    class NarrowCarrier:
        """Event-like object whose get_extra has no `default` parameter."""

        _extras = {"parsed_params": {"city": "Beijing"}}

        def get_extra(self, key=None):
            if key is None:
                return self._extras
            return self._extras.get(key)

    narrow = NarrowCarrier()
    assert "parsed_params" in narrow.get_extra()
    with pytest.raises(TypeError):
        "parsed_params" in narrow.get_extra(default={})


class FakeEvent:
    """Event stub exposing only what the scheduler touches."""

    def __init__(self) -> None:
        self.stopped = False

    def is_stopped(self) -> bool:
        """Return whether the event was stopped by a stage."""
        return self.stopped


class FastStage:
    """Coroutine stage that finishes far below the threshold."""

    async def process(self, event) -> None:  # noqa: ARG002
        return None


class SlowStage:
    """Coroutine stage that sleeps beyond the (patched) threshold."""

    async def process(self, event) -> None:  # noqa: ARG002
        await asyncio.sleep(0.05)
        return None


class OuterStage:
    """Generator stage that only delegates to the later stages."""

    async def process(self, event):  # noqa: ARG002
        yield
        await asyncio.sleep(0)


def make_scheduler(stages: list) -> PipelineScheduler:
    """Build a scheduler with a fixed stage list, skipping stage discovery."""
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = stages
    return scheduler


@pytest.mark.asyncio
async def test_scheduler_is_silent_below_the_threshold():
    """S2: a normal pipeline execution must not add a single log line."""
    scheduler = make_scheduler([FastStage(), OuterStage(), FastStage()])

    with patch.object(scheduler_module, "logger") as mock_logger:
        await scheduler._process_stages(FakeEvent())

    mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_scheduler_warns_above_the_threshold(monkeypatch):
    """S2: a stage above the threshold produces exactly one searchable warning."""
    monkeypatch.setattr(scheduler_module, "SLOW_STAGE_THRESHOLD_MS", 10)
    scheduler = make_scheduler([SlowStage(), FastStage()])

    with patch.object(scheduler_module, "logger") as mock_logger:
        await scheduler._process_stages(FakeEvent())

    assert mock_logger.warning.call_count == 1
    message = mock_logger.warning.call_args[0][0]
    assert "SlowStage" in message
    assert "took" in message


@pytest.mark.asyncio
async def test_scheduler_attributes_slow_time_to_the_slow_stage(monkeypatch):
    """S2: a fast generator stage does not absorb the time of its successors.

    The generator stage may be entered again by the onion model after a yield,
    so the slow stage can be measured more than once; what must never happen is
    the fast generator stage being reported as slow.
    """
    monkeypatch.setattr(scheduler_module, "SLOW_STAGE_THRESHOLD_MS", 10)
    scheduler = make_scheduler([OuterStage(), SlowStage()])

    with patch.object(scheduler_module, "logger") as mock_logger:
        await scheduler._process_stages(FakeEvent())

    messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert messages, "the slow stage was not reported"
    assert all("SlowStage" in message for message in messages)
    assert not any("OuterStage" in message for message in messages)


@pytest.mark.asyncio
async def test_scheduler_stops_are_unchanged():
    """Behaviour guard: a stopped event still halts later stages."""
    executed: list[str] = []

    class StoppingStage:
        async def process(self, event) -> None:
            executed.append("stopping")
            event.stopped = True

    class NeverStage:
        async def process(self, event) -> None:  # noqa: ARG002
            executed.append("never")

    scheduler = make_scheduler([StoppingStage(), NeverStage()])
    await scheduler._process_stages(FakeEvent())

    assert executed == ["stopping"]


def test_waking_check_stage_is_importable_with_fixed_call():
    """F4: the module exposes the stage class and the fixed call site."""
    assert WakingCheckStage.__name__ == "WakingCheckStage"
    source = WAKING_CHECK_SOURCE.read_text(encoding="utf-8")
    assert '"parsed_params" in event.get_extra():' in source
