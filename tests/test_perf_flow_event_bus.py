"""Performance regression tests for the event-bus conf resolution cache (F2).

The event bus resolved the configuration of every incoming event through
``AstrBotConfigManager.get_conf_info``: a UMO re-parse plus a linear scan of the
routing table with ``fnmatch``. The result is now cached until the routing data
or the profile metadata changes. These tests pin the invalidation rules, because
a stale mapping would silently send a session to the wrong configuration.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from astrbot.core.astrbot_config_mgr import AstrBotConfigManager
from astrbot.core.event_bus import EventBus
from astrbot.core.umop_config_router import UmopConfigRouter

# Routing data of the production deployment (read-only snapshot of data_v4.db).
PROD_ROUTES = {
    "weixin_personal_ultk:*:*": "f222c784-451a-41a2-b35a-0726e3d38ae0",
    "weixin_personal_pvsx:*:*": "f222c784-451a-41a2-b35a-0726e3d38ae0",
    "哈尼娅:GroupMessage:1035699087": "596ffb44-a64e-42cf-bfd0-c257384656f5",
    "哈尼娅:GroupMessage:744868236": "596ffb44-a64e-42cf-bfd0-c257384656f5",
    "哈尼娅:FriendMessage:*": "596ffb44-a64e-42cf-bfd0-c257384656f5",
}
PROD_ABCONF = {
    "596ffb44-a64e-42cf-bfd0-c257384656f5": {
        "path": "abconf_596ffb44-a64e-42cf-bfd0-c257384656f5.json",
        "name": "樱铃",
    },
    "f222c784-451a-41a2-b35a-0726e3d38ae0": {
        "path": "abconf_f222c784-451a-41a2-b35a-0726e3d38ae0.json",
        "name": "微信",
    },
}
PROD_UMO = "哈尼娅:GroupMessage:1035699087"


class FakeConfigManager:
    """Config manager stub that counts resolutions and serves a live routing table."""

    def __init__(self, routes: dict[str, str], abconf: dict) -> None:
        self.ucr = SimpleNamespace(umop_to_conf_id=routes)
        self.abconf_data = abconf
        self.calls = 0
        self.by_umo: dict[str, str] = {}

    def get_conf_info(self, umo: str) -> dict:
        """Return the conf info the routing table currently implies for `umo`."""
        self.calls += 1
        conf_id = self.by_umo.get(umo, "default")
        meta = self.abconf_data.get(conf_id)
        name = meta["name"] if meta else conf_id
        return {"id": conf_id, "name": name}


def make_bus(config_mgr) -> EventBus:
    """Build an EventBus without a running pipeline scheduler mapping."""
    return EventBus(
        event_queue=asyncio.Queue(),
        pipeline_scheduler_mapping={},
        astrbot_config_mgr=config_mgr,
    )


def make_manager(routes: dict[str, str], abconf: dict) -> AstrBotConfigManager:
    """Build a real AstrBotConfigManager over the given routing data, without I/O."""
    ucr = UmopConfigRouter.__new__(UmopConfigRouter)
    ucr.umop_to_conf_id = dict(routes)
    manager = AstrBotConfigManager.__new__(AstrBotConfigManager)
    manager.ucr = ucr
    manager.abconf_data = dict(abconf)
    return manager


def test_resolution_matches_the_real_config_manager():
    """The cached path returns exactly what get_conf_info returns (real code)."""
    manager = make_manager(PROD_ROUTES, PROD_ABCONF)
    bus = make_bus(manager)

    for umo in [
        PROD_UMO,
        "哈尼娅:FriendMessage:2060958352",  # wildcard route
        "weixin_personal_ultk:GroupMessage:42",
        "unknown_platform:GroupMessage:1",  # falls back to the default profile
    ]:
        conf_info = manager.get_conf_info(umo)
        expected = (conf_info["id"], conf_info.get("name") or conf_info["id"])
        assert bus._resolve_conf(umo) == expected


def test_repeated_events_resolve_once():
    """A repeated session origin reuses the cached resolution."""
    manager = FakeConfigManager(dict(PROD_ROUTES), dict(PROD_ABCONF))
    manager.by_umo[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"
    bus = make_bus(manager)

    first = bus._resolve_conf(PROD_UMO)
    second = bus._resolve_conf(PROD_UMO)

    assert first == second == ("596ffb44-a64e-42cf-bfd0-c257384656f5", "樱铃")
    assert manager.calls == 1


def test_inplace_reroute_takes_effect_immediately():
    """update_route() mutates the table in place; the cache must notice it."""
    routes = dict(PROD_ROUTES)
    manager = FakeConfigManager(routes, dict(PROD_ABCONF))
    manager.by_umo[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"
    bus = make_bus(manager)

    assert bus._resolve_conf(PROD_UMO)[0] == "596ffb44-a64e-42cf-bfd0-c257384656f5"

    routes[PROD_UMO] = "f222c784-451a-41a2-b35a-0726e3d38ae0"
    manager.by_umo[PROD_UMO] = "f222c784-451a-41a2-b35a-0726e3d38ae0"

    assert bus._resolve_conf(PROD_UMO) == (
        "f222c784-451a-41a2-b35a-0726e3d38ae0",
        "微信",
    )


def test_added_and_removed_routes_take_effect_immediately():
    """update_routing_data() / delete_route() must invalidate the cache."""
    routes = {"哈尼娅:FriendMessage:*": "596ffb44-a64e-42cf-bfd0-c257384656f5"}
    manager = FakeConfigManager(routes, dict(PROD_ABCONF))
    manager.by_umo[PROD_UMO] = "default"
    bus = make_bus(manager)

    assert bus._resolve_conf(PROD_UMO) == ("default", "default")

    routes[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"
    manager.by_umo[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"
    assert bus._resolve_conf(PROD_UMO) == (
        "596ffb44-a64e-42cf-bfd0-c257384656f5",
        "樱铃",
    )

    del routes[PROD_UMO]
    manager.by_umo[PROD_UMO] = "default"
    assert bus._resolve_conf(PROD_UMO) == ("default", "default")


def test_profile_metadata_change_takes_effect_immediately():
    """A renamed or added profile refreshes the cached display name."""
    routes = {PROD_UMO: "596ffb44-a64e-42cf-bfd0-c257384656f5"}
    manager = FakeConfigManager(routes, dict(PROD_ABCONF))
    manager.by_umo[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"
    bus = make_bus(manager)

    assert bus._resolve_conf(PROD_UMO)[1] == "樱铃"

    renamed_abconf = {
        "596ffb44-a64e-42cf-bfd0-c257384656f5": {
            "path": "abconf_596ffb44-a64e-42cf-bfd0-c257384656f5.json",
            "name": "renamed",
        },
    }
    manager.abconf_data = renamed_abconf

    assert bus._resolve_conf(PROD_UMO) == (
        "596ffb44-a64e-42cf-bfd0-c257384656f5",
        "renamed",
    )


def test_cache_is_bounded():
    """Session origins are unbounded, so the cache must not grow without limit."""
    manager = FakeConfigManager({"::": "default"}, {})
    bus = make_bus(manager)

    for i in range(1100):
        bus._resolve_conf(f"platform:GroupMessage:{i}")

    assert len(bus._conf_cache) <= 1024


@pytest.mark.asyncio
async def test_dispatch_routes_cached_events_and_keeps_the_cache_hot():
    """Two events of one session hit get_conf_info once and reach the scheduler."""
    routes = {PROD_UMO: "596ffb44-a64e-42cf-bfd0-c257384656f5"}
    manager = FakeConfigManager(routes, dict(PROD_ABCONF))
    manager.by_umo[PROD_UMO] = "596ffb44-a64e-42cf-bfd0-c257384656f5"

    executed = asyncio.Event()
    executions: list = []
    scheduler = MagicMock()

    async def execute(event):
        executions.append(event)
        if len(executions) == 2:
            executed.set()

    scheduler.execute = execute
    bus = EventBus(
        event_queue=asyncio.Queue(),
        pipeline_scheduler_mapping={"596ffb44-a64e-42cf-bfd0-c257384656f5": scheduler},
        astrbot_config_mgr=manager,
    )

    events = []
    for _ in range(2):
        event = MagicMock()
        event.unified_msg_origin = PROD_UMO
        event.get_platform_id.return_value = "platform"
        event.get_platform_name.return_value = "Platform"
        event.get_sender_name.return_value = "User"
        event.get_sender_id.return_value = "user"
        event.get_message_outline.return_value = "Hello"
        events.append(event)
        await bus.event_queue.put(event)

    task = asyncio.create_task(bus.dispatch())
    try:
        await asyncio.wait_for(executed.wait(), timeout=2.0)
    finally:
        for pending in list(bus._pending_tasks):
            pending.cancel()
        task.cancel()
        await asyncio.gather(task, *list(bus._pending_tasks), return_exceptions=True)

    assert len(executions) == 2
    assert manager.calls == 1


def test_dispatch_still_resolves_with_mocked_manager():
    """A MagicMock manager keeps the previous per-event call behaviour."""
    manager = MagicMock()
    manager.get_conf_info = MagicMock(return_value={"id": "x", "name": "X"})
    bus = make_bus(manager)

    assert bus._resolve_conf("platform:GroupMessage:1") == ("x", "X")
    assert bus._resolve_conf("platform:GroupMessage:1") == ("x", "X")
    assert manager.get_conf_info.call_count == 2


def test_cache_does_not_call_get_conf_info_for_known_sessions():
    """The cache is the only path: no hidden call when the entry is present."""
    manager = FakeConfigManager({PROD_UMO: "default"}, {})
    manager.by_umo[PROD_UMO] = "default"
    bus = make_bus(manager)

    with patch.object(manager, "get_conf_info", wraps=manager.get_conf_info) as spy:
        for _ in range(100):
            bus._resolve_conf(PROD_UMO)
        assert spy.call_count == 1
