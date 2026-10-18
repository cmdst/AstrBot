"""客户端断连（`RuntimeError("No response returned.")` + `anyio.EndOfStream`）的回归。

背景：见 `astrbot/dashboard/disconnect.py`。生产现场（2026-10-19，SSE `/api/v1/logs/live`）：

    starlette/middleware/base.py:151 call_next -> anyio/streams/memory.py:132 raise EndOfStream
    -> RuntimeError("No response returned.") 冒泡到 dashboard 兜底处理器，被记成
       `[ERRO] Unhandled exception in dashboard API`（进而被日志巡检判成 HIGH）。

本文件锁两件事：① 这种形态**被识别**并按 499 正常收尾；② **其它** `RuntimeError`
（例如应用真的没返回响应）绝不能被吞——否则会把真 bug 掩盖成静默 499。
"""

from __future__ import annotations

import json

import anyio
import httpx
import pytest
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse as StarlettePlainTextResponse
from starlette.routing import Route

from astrbot.dashboard.disconnect import (
    CLIENT_DISCONNECTED_STATUS,
    call_next_tolerant,
    client_disconnected_response,
    is_client_disconnect,
)


def disconnect_shaped_error() -> RuntimeError:
    """复刻 starlette 现场形态：`RuntimeError` 的直接 cause 是 `anyio.EndOfStream`。"""

    exc = RuntimeError("No response returned.")
    exc.__cause__ = anyio.EndOfStream()
    return exc


def test_disconnect_shape_is_recognized() -> None:
    assert is_client_disconnect(disconnect_shaped_error()) is True


@pytest.mark.parametrize(
    "exc",
    [
        None,
        ValueError("boom"),
        RuntimeError("No response returned."),  # 没有 cause：可能是真 bug
        RuntimeError("boom", ValueError("other")),  # cause 不是 EndOfStream
    ],
)
def test_other_errors_are_not_treated_as_disconnect(exc: BaseException | None) -> None:
    assert is_client_disconnect(exc) is False


def test_client_disconnected_response_shape() -> None:
    response = client_disconnected_response()

    assert isinstance(response, JSONResponse)
    assert response.status_code == CLIENT_DISCONNECTED_STATUS
    assert json.loads(response.body) == {
        "status": "error",
        "message": "Client disconnected",
    }


@pytest.mark.asyncio
async def test_call_next_tolerant_converts_disconnect_to_499() -> None:
    async def call_next(_request: object) -> object:  # pragma: no cover - 只负责抛
        raise disconnect_shaped_error()

    response = await call_next_tolerant(object(), call_next)

    assert isinstance(response, JSONResponse)
    assert response.status_code == CLIENT_DISCONNECTED_STATUS


@pytest.mark.asyncio
async def test_call_next_tolerant_reraises_runtime_error_without_end_of_stream_cause() -> None:
    """反证：没有 `EndOfStream` cause 的 `RuntimeError` 必须原样抛出（不吞真错）。"""

    async def call_next(_request: object) -> object:  # pragma: no cover - 只负责抛
        raise RuntimeError("No response returned.")

    with pytest.raises(RuntimeError, match="No response returned."):
        await call_next_tolerant(object(), call_next)


@pytest.mark.asyncio
async def test_call_next_tolerant_passes_through_normal_response() -> None:
    expected = PlainTextResponse("pong")

    async def call_next(_request: object) -> object:
        return expected

    assert await call_next_tolerant(object(), call_next) is expected


@pytest.mark.asyncio
async def test_middleware_shape_is_unaffected_for_normal_requests() -> None:
    """把真实 helper 放进真实的 `BaseHTTPMiddleware` 形态里，正常请求行为不变。"""

    class _TolerantBodyLimit(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            return await call_next_tolerant(request, call_next)

    async def ping(_request) -> StarlettePlainTextResponse:  # type: ignore[no-untyped-def]
        return StarlettePlainTextResponse("pong")

    app = Starlette(
        routes=[Route("/ping", ping)],
        middleware=[Middleware(_TolerantBodyLimit)],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert response.text == "pong"
