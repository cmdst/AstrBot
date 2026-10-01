"""Runtime-domain performance guards (SQLite PRAGMA coverage and history contract)."""

import json
from dataclasses import asdict

import pytest
from sqlalchemy import event, text

from astrbot.core.db.po import Conversation

EXPECTED_CONNECTION_PRAGMAS = (1, 8000, 2, 134217728)


@pytest.mark.asyncio
async def test_every_pooled_connection_receives_performance_pragmas(temp_db):
    """Connection-scoped PRAGMAs must apply to every connection, not just one."""
    await temp_db.initialize()
    connections = [await temp_db.engine.connect() for _ in range(4)]
    try:
        rows = [
            [
                (await conn.execute(text(f"PRAGMA {pragma}"))).scalar()
                for pragma in ("synchronous", "cache_size", "temp_store", "mmap_size")
            ]
            for conn in connections
        ]
    finally:
        for conn in connections:
            await conn.close()

    assert all(tuple(row) == EXPECTED_CONNECTION_PRAGMAS for row in rows), rows


@pytest.mark.asyncio
async def test_journal_mode_is_wal(temp_db):
    """WAL mode is a persisted database property and must survive initialization."""
    await temp_db.initialize()
    async with temp_db.engine.connect() as conn:
        assert (await conn.execute(text("PRAGMA journal_mode"))).scalar() == "wal"


@pytest.mark.asyncio
async def test_update_conversation_does_not_read_the_row_back(temp_db):
    """`update_conversation` must not re-select the content column it just wrote."""
    await temp_db.initialize()
    conversation = await temp_db.create_conversation(
        user_id="umo:test",
        platform_id="test",
        content=[{"role": "user", "content": "hello"}],
    )
    statements: list[str] = []

    def record(conn, cursor, statement, params, context, executemany):
        statements.append(statement.split()[0].upper())

    event.listen(temp_db.engine.sync_engine, "before_cursor_execute", record)
    try:
        result = await temp_db.update_conversation(
            conversation.conversation_id,
            content=[{"role": "user", "content": "updated"}],
        )
    finally:
        event.remove(temp_db.engine.sync_engine, "before_cursor_execute", record)

    assert result is None
    assert statements == ["UPDATE"], statements


@pytest.mark.asyncio
async def test_updated_conversation_is_persisted(temp_db):
    """The removed read-back must not change what ends up stored."""
    await temp_db.initialize()
    conversation = await temp_db.create_conversation(
        user_id="umo:test",
        platform_id="test",
        content=[{"role": "user", "content": "hello"}],
    )
    await temp_db.update_conversation(
        conversation.conversation_id,
        content=[{"role": "user", "content": "updated"}],
    )
    stored = await temp_db.get_conversation_by_id(conversation.conversation_id)
    assert stored is not None
    assert stored.content == [{"role": "user", "content": "updated"}]


def test_conversation_history_stays_a_json_string():
    """`Conversation.history` keeps its public `str` contract."""
    conversation = Conversation(
        platform_id="p",
        user_id="u",
        cid="c",
        history="",
        history_list=[{"role": "user", "content": "hi"}],
    )
    assert isinstance(conversation.history, str)
    assert json.loads(conversation.history) == [{"role": "user", "content": "hi"}]


def test_conversation_accepts_legacy_history_string():
    """Legacy `Conversation(history=<str>)` construction still populates the list."""
    payload = [{"role": "assistant", "content": "ok"}]
    conversation = Conversation(
        platform_id="p", user_id="u", cid="c", history=json.dumps(payload)
    )
    assert conversation.history_list == payload
    assert json.loads(conversation.history) == payload


def test_history_assignment_refreshes_the_list():
    """Plugins assign `conversation.history`; the list must follow that write."""
    conversation = Conversation(platform_id="p", user_id="u", cid="c")
    conversation.history = json.dumps([{"role": "user", "content": "from plugin"}])
    assert conversation.history_list == [{"role": "user", "content": "from plugin"}]


def test_empty_history_is_stable():
    """Default construction keeps both representations empty."""
    conversation = Conversation(platform_id="p", user_id="u", cid="c")
    assert conversation.history == "[]"
    assert conversation.history_list == []
    assert asdict(conversation)["history"] == "[]"


def test_invalid_history_string_does_not_break_construction():
    """A non-JSON `history` value must not raise while building the object."""
    conversation = Conversation(
        platform_id="p", user_id="u", cid="c", history="{not json"
    )
    assert conversation.history == "{not json"
    assert conversation.history_list == []
