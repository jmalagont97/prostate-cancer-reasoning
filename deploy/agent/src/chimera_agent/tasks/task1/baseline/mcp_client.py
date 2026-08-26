"""MCP stdio client for the three fixed subtask tools.

Launches ``chimera_agent.tasks.task1.baseline.mcp_server`` as a subprocess and exposes
a thin ``call(tool_name, args)`` coroutine. Used only by the orchestrator
(``run.py``) — the LLM never sees these tools.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


class SubtaskMCPClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def __aenter__(self) -> "SubtaskMCPClient":
        repo_root = Path(__file__).resolve().parents[7]
        src_dir = str(Path(__file__).resolve().parents[4])
        env = dict(os.environ)
        env["PYTHONPATH"] = src_dir + os.pathsep + str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["SUBMODELS_DIR"] = str(Path(self.cfg.paths.submodels_dir).resolve())

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "chimera_agent.tasks.task1.baseline.mcp_server"],
            env=env,
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        tools = await self._session.list_tools()
        log.info("MCP server ready with tools: %s", [t.name for t in tools.tools])
        return self

    async def call(self, tool_name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError("MCP client not started")
        result = await self._session.call_tool(tool_name, arguments)
        return "\n".join(getattr(c, "text", "") for c in result.content)

    async def __aexit__(self, *exc) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(*exc)
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(*exc)
