"""客户端断连的识别与容错调用。

背景（2026-10-19 实测）：Starlette 的 ``BaseHTTPMiddleware`` 在「客户端在响应开始前
就断开」时，会把内部接收流的 ``anyio.EndOfStream`` 包成
``RuntimeError("No response returned.")`` 抛出。对 SSE（``/api/v1/logs/live``）这类
长连接端点，浏览器建连后立刻取消就会走到这条路径——它是**正常路径**（对方已经走了，
响应无处可送），但会一路冒泡到 dashboard 的兜底处理器并被记成
``Unhandled exception in dashboard API`` 的 ``[ERRO]``，进而被日志巡检插件判为 HIGH。

本模块把这段识别逻辑集中到一处，供中间件与兜底处理器共用：

* :func:`is_client_disconnect` —— 只认「``RuntimeError`` 且其 **直接 cause** 是
  ``anyio.EndOfStream``」。其它 ``RuntimeError``（例如应用真的没返回响应）必须原样
  抛出/上报，不能被这里吞掉。
* :func:`call_next_tolerant` —— 中间件里 ``await call_next(request)`` 的替代品，
  命中上一条时返回 499（客户端已断开，没人接收这个响应），否则照旧抛出。
"""

from __future__ import annotations

from typing import Any

import anyio
from fastapi.responses import JSONResponse

from astrbot.core.log import LogManager
from astrbot.dashboard.responses import error

#: 非标准状态码，仅用于表达「客户端在响应开始前已断开」（Nginx 同名语义）。
CLIENT_DISCONNECTED_STATUS = 499


def is_client_disconnect(exc: BaseException | None) -> bool:
    """判断异常是否为「客户端在响应开始前断开」这一类。

    Args:
        exc: 待判断的异常。

    Returns:
        命中返回 ``True``；其余情况（含 ``RuntimeError`` 但没有 ``EndOfStream``
        直接 cause 的情况）返回 ``False``。
    """

    return isinstance(exc, RuntimeError) and isinstance(exc.__cause__, anyio.EndOfStream)


def client_disconnected_response() -> JSONResponse:
    """构造「客户端已断开」的响应（调用方按需返回，用于让栈正常收尾）。"""

    return JSONResponse(
        error("Client disconnected"),
        status_code=CLIENT_DISCONNECTED_STATUS,
    )


async def call_next_tolerant(request: Any, call_next: Any) -> Any:
    """``await call_next(request)`` 的容错版本：客户端断连时不再抛异常。

    Args:
        request: 当前请求（Starlette ``Request``）。
        call_next: Starlette 中间件链的 ``call_next``。

    Returns:
        下游响应；客户端在响应开始前断开时返回 499 响应。

    Raises:
        RuntimeError: 非「客户端断连」的 ``RuntimeError`` 原样再抛（不吞真错）。
    """

    try:
        return await call_next(request)
    except RuntimeError as exc:
        if not is_client_disconnect(exc):
            raise
        LogManager.GetLogger("astrbot.dashboard").debug(
            "客户端在响应开始前断开，按正常收尾处理：%s",
            getattr(getattr(request, "url", None), "path", "?"),
        )
        return client_disconnected_response()
