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


def test_contexts_never_alias_the_conversation_history_list():
    """Every consumer site must copy `history_list`, never take it over (I1).

    The pipeline mutates `req.contexts` in place, so an alias would rewrite the
    conversation's stored history. This is an AST gate over the real consumer files.
    """
    import ast
    import pathlib

    offenders = []
    for relative in (
        "astrbot/core/astr_main_agent.py",
        "astrbot/core/cron/manager.py",
        "astrbot/core/astr_agent_tool_exec.py",
    ):
        tree = ast.parse(pathlib.Path(relative).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            assigns_contexts = any(
                isinstance(target, ast.Attribute) and target.attr == "contexts"
                for target in node.targets
            )
            if not assigns_contexts:
                continue
            source = ast.unparse(node.value)
            if "history_list" in source and not source.startswith("list("):
                offenders.append(f"{relative}:{node.lineno} -> {source}")

    assert not offenders, f"req.contexts aliases history_list at: {offenders}"


def test_boundary_copy_isolates_contexts_from_stored_history():
    """In-place writes to `req.contexts` must not rewrite the stored history (I1)."""
    stored = [{"role": "user", "content": "hello"}]
    conversation = Conversation(
        platform_id="p", user_id="u", cid="c", history="", history_list=list(stored)
    )

    # Exactly what the consumers do: a copy at the boundary.
    req_contexts = list(conversation.history_list)
    assert req_contexts is not conversation.history_list

    # Exactly what the pipeline does to that copy: persona dialogs prepended and
    # file-extract results appended, both in place.
    req_contexts[:0] = [{"role": "system", "content": "persona begin dialog"}]
    req_contexts.append({"role": "system", "content": "file extract"})

    assert json.loads(conversation.history) == stored
    assert conversation.history_list == stored


@pytest.mark.asyncio
async def test_persona_begin_dialogs_do_not_leak_into_stored_history(monkeypatch):
    """The real persona path must not push `begin_dialogs` into the stored history.

    Exercises `_ensure_persona_and_skills` with a non-empty persona begin-dialog list,
    which is the case that rewrote the conversation history before the boundary copy.
    """
    from unittest.mock import AsyncMock, MagicMock

    from astrbot.core.astr_main_agent import _ensure_persona_and_skills
    from astrbot.core.provider.entities import ProviderRequest

    stored = [{"role": "user", "content": "hello"}]
    conversation = Conversation(
        platform_id="p", user_id="u", cid="c", history="", history_list=list(stored)
    )

    begin_dialogs = [
        {"role": "user", "content": f"persona dialog {i}"} for i in range(3)
    ]
    persona = {
        "prompt": "you are a test persona",
        "name": "test-persona",
        "_begin_dialogs_processed": begin_dialogs,
        "tools": [],
        "skills": [],
    }

    plugin_context = MagicMock()
    plugin_context.get_config.return_value = {"computer_use_runtime": "none"}
    plugin_context.subagent_orchestrator = None
    plugin_context.persona_manager.resolve_selected_persona = AsyncMock(
        return_value=("test-persona", persona, None, False)
    )

    event = MagicMock()
    event.unified_msg_origin = "test:FriendMessage:invariant"
    event.get_platform_name.return_value = "test"
    event.get_extra.return_value = None

    req = ProviderRequest()
    req.conversation = conversation
    req.contexts = list(conversation.history_list)
    assert req.contexts is not conversation.history_list

    await _ensure_persona_and_skills(req, dict(persona), plugin_context, event)

    # The persona dialogs are in the request, not in what the conversation stores.
    assert [m for m in req.contexts if m.get("content") == "persona dialog 0"]
    assert json.loads(conversation.history) == stored
    assert conversation.history_list == stored
