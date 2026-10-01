"""Regression tests for the provider-domain SharedPreferences read cache (P1 / I2).

Contract: .agent/perf/02-plan.md §4.1 P1 and §4.2 I2. The metrics that matter are
deterministic: how many point queries a message-shaped read burst issues, and
whether a write is visible to the next read. Latency assertions stay far above
the observed cost so they only catch a lost cache, never scheduler noise.
"""

import asyncio

import pytest
import pytest_asyncio

from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.utils.shared_preferences import SharedPreferences

UMO = "哈尼娅:GroupMessage:1035699087"
HOT_KEY = "session_service_config"


@pytest_asyncio.fixture
async def preferences(tmp_path):
    database = SQLiteDatabase(str(tmp_path / "preferences.db"))
    await database.initialize()
    store = SharedPreferences(database, tmp_path / "preferences.json")
    await store.initialize()
    try:
        yield store, database
    finally:
        await store.close()
        await database.engine.dispose()


def count_point_queries(database) -> dict[str, int]:
    """Count get_preference calls by wrapping the database helper.

    Args:
        database: SQLiteDatabase instance used by the store under test.

    Returns:
        Mutable counter whose ``"n"`` field tracks point-query invocations.
    """
    calls = {"n": 0}
    original = database.get_preference

    async def counting_get_preference(scope, scope_id, key):
        calls["n"] += 1
        return await original(scope, scope_id, key)

    database.get_preference = counting_get_preference
    return calls


@pytest.mark.asyncio
async def test_repeated_absent_key_reads_collapse_to_one_point_query(preferences):
    """A burst of reads for an absent hot key must not query per read."""
    store, database = preferences
    calls = count_point_queries(database)

    # Three reads per message, as the session / LLM / TTS status checks do.
    for _ in range(3 * 20):
        assert (await store.get_async("umo", UMO, HOT_KEY, {})) == {}

    assert calls["n"] == 1
    assert store._cache == {}, "reads must not populate the write overlay"


@pytest.mark.asyncio
async def test_absent_key_read_takes_the_overlay_fast_path(preferences):
    """The cached-miss path must stay orders of magnitude below a point query."""
    store, _database = preferences
    await store.get_async("umo", UMO, HOT_KEY, {})

    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(200):
        await store.get_async("umo", UMO, HOT_KEY, {})
    mean_ms = (loop.time() - start) / 200 * 1e3

    # Observed on the frozen baseline: 1.239 ms per point query. A regression to
    # per-read queries is caught here with a wide margin against CI noise.
    assert mean_ms < 0.1, f"cached miss still costs {mean_ms:.4f} ms"


@pytest.mark.asyncio
async def test_negative_cache_expires_after_ttl(preferences):
    """A cached miss must not outlive its short TTL."""
    store, database = preferences
    calls = count_point_queries(database)

    await store.get_async("umo", UMO, HOT_KEY, {})
    await store.get_async("umo", UMO, HOT_KEY, {})
    assert calls["n"] == 1

    await asyncio.sleep(1.7)
    await store.get_async("umo", UMO, HOT_KEY, {})
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_persisted_value_is_never_served_from_the_negative_cache(tmp_path):
    """Historical values must stay readable through the point-query fallback."""
    database = SQLiteDatabase(str(tmp_path / "historical.db"))
    await database.initialize()
    await database.insert_preference_or_update(
        "umo",
        UMO,
        HOT_KEY,
        {"val": {"llm_enabled": False}},
    )
    store = SharedPreferences(database, tmp_path / "preferences.json")
    try:
        await store.initialize()
        assert await store.get_async("umo", UMO, HOT_KEY, {}) == {"llm_enabled": False}
        assert await store.get_async("umo", UMO, HOT_KEY, {}) == {"llm_enabled": False}
    finally:
        await store.close()
        await database.engine.dispose()


@pytest.mark.asyncio
async def test_write_is_visible_immediately_after_a_cached_miss(preferences):
    """I2: a write must invalidate the negative cache without waiting for its TTL."""
    store, _database = preferences

    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {}
    await store.put_async("umo", UMO, HOT_KEY, {"llm_enabled": False})

    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {"llm_enabled": False}

    await store.remove_async("umo", UMO, HOT_KEY)
    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {}


@pytest.mark.asyncio
async def test_clear_invalidates_negative_entries_of_the_scope(preferences):
    """A scoped clear must drop negative entries so later writes are visible."""
    store, _database = preferences

    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {}
    await store.clear_async("umo", UMO)
    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {}

    await store.put_async("umo", UMO, HOT_KEY, {"section_enabled": False})
    assert await store.get_async("umo", UMO, HOT_KEY, {}) == {"section_enabled": False}


@pytest.mark.asyncio
async def test_concurrent_write_during_point_query_wins(preferences, monkeypatch):
    """A write landing mid-query must not be masked by the recorded absence."""
    store, database = preferences
    original = database.get_preference

    async def write_then_query(scope, scope_id, key):
        await store.put_async(scope, scope_id, key, "concurrent")
        return await original(scope, scope_id, key)

    monkeypatch.setattr(database, "get_preference", write_then_query)

    assert await store.get_async("umo", UMO, HOT_KEY, None) == "concurrent"
    assert await store.get_async("umo", UMO, HOT_KEY, None) == "concurrent"


@pytest.mark.asyncio
async def test_deprecated_sync_read_still_sees_historical_values(preferences):
    """The sync fallback must keep working and must not be served by the negative cache."""
    store, database = preferences
    await database.insert_preference_or_update(
        "global",
        "global",
        "inactivated_llm_tools",
        {"val": ["tool-a"]},
    )

    assert store.get(HOT_KEY, {}, scope="umo", scope_id=UMO) == {}
    assert store.get("inactivated_llm_tools", scope="global", scope_id="global") == [
        "tool-a"
    ]
