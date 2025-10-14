"""Pytest configuration for compatibility patches and utilities."""
from __future__ import annotations

import inspect
import json
import sys
import types
import uuid
from typing import Any, Dict, Iterable, Tuple

import anyio
from typing import ForwardRef


def _patch_forward_ref() -> None:
    if not hasattr(ForwardRef, "_evaluate"):
        return

    signature = inspect.signature(ForwardRef._evaluate)
    parameter = signature.parameters.get("recursive_guard")
    if parameter is None:
        original = ForwardRef._evaluate

        def _patched_evaluate(self, globalns, localns, recursive_guard=None):  # type: ignore[override]
            if recursive_guard is None:
                recursive_guard = set()
            return original(self, globalns, localns)

        ForwardRef._evaluate = _patched_evaluate  # type: ignore[assignment]
        return

    if parameter.default is not inspect._empty:
        return

    original = ForwardRef._evaluate

    if "type_params" in signature.parameters:

        def _patched_evaluate(  # type: ignore[override]
            self,
            globalns,
            localns,
            type_params=None,
            *,
            recursive_guard=None,
        ):
            if recursive_guard is None:
                recursive_guard = set()
            return original(
                self,
                globalns,
                localns,
                type_params,
                recursive_guard=recursive_guard,
            )

    else:

        def _patched_evaluate(self, globalns, localns, recursive_guard=None):  # type: ignore[override]
            if recursive_guard is None:
                recursive_guard = set()
            return original(self, globalns, localns, recursive_guard)

    ForwardRef._evaluate = _patched_evaluate  # type: ignore[assignment]


_patch_forward_ref()

Headers = Iterable[Tuple[bytes, bytes]]


class SimpleResponse:
    """Minimal HTTP response object mimicking ``requests.Response``."""

    def __init__(self, status_code: int, headers: Headers, body: bytes) -> None:
        self.status_code = status_code
        self.headers = list(headers)
        self._body = body

    @property
    def content(self) -> bytes:
        return self._body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="ignore")

    def json(self) -> Any:
        return json.loads(self.text or "{}")


class TestClient:
    """A lightweight ASGI test client used for unit tests."""

    __test__ = False  # Prevent pytest from collecting this class as tests.

    def __init__(self, app: Any):
        self.app = app

    def get(self, path: str, *, headers: Dict[str, str] | None = None) -> SimpleResponse:
        return self._request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: Dict[str, Any] | None = None,
        files: Dict[str, Tuple[str, bytes | str, str]] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> SimpleResponse:
        return self._request("POST", path, json_body=json, files=files, headers=headers)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Dict[str, Any] | None = None,
        files: Dict[str, Tuple[str, bytes | str | Any, str]] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> SimpleResponse:
        body, header_list = self._build_body(json_body=json_body, files=files)
        if headers:
            for key, value in headers.items():
                header_list.append((key.encode("latin-1"), value.encode("latin-1")))
        return anyio.run(self._send_request, method, path, header_list, body)

    async def _send_request(
        self,
        method: str,
        path: str,
        headers: list[Tuple[bytes, bytes]],
        body: bytes,
    ) -> SimpleResponse:
        response: Dict[str, Any] = {"headers": [], "body": b""}
        more_body = bool(body)

        async def receive() -> Dict[str, Any]:
            nonlocal more_body
            if more_body:
                more_body = False
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                response["headers"] = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response["body"] += message.get("body", b"")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode("latin-1"),
            "query_string": b"",
            "headers": headers,
            "client": ("testclient", 0),
            "server": ("testserver", 80),
        }

        await self.app(scope, receive, send)
        return SimpleResponse(response.get("status", 500), response["headers"], response["body"])

    def _build_body(
        self,
        *,
        json_body: Dict[str, Any] | None = None,
        files: Dict[str, Tuple[str, bytes | str | Any, str]] | None = None,
    ) -> Tuple[bytes, list[Tuple[bytes, bytes]]]:
        headers: list[Tuple[bytes, bytes]] = []
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers.append((b"content-type", b"application/json"))
            headers.append((b"content-length", str(len(body)).encode()))
            return body, headers
        if files:
            boundary = uuid.uuid4().hex
            parts: list[bytes] = []
            for field, (filename, content, content_type) in files.items():
                if hasattr(content, "read"):
                    raw = content.read()
                    if hasattr(content, "seek"):
                        content.seek(0)
                    content_bytes = bytes(raw)
                elif isinstance(content, str):
                    content_bytes = content.encode("utf-8")
                else:
                    content_bytes = bytes(content)
                disposition = (
                    f"form-data; name=\"{field}\"; filename=\"{filename}\""
                )
                part = (
                    f"--{boundary}\r\n".encode()
                    + f"Content-Disposition: {disposition}\r\n".encode()
                    + f"Content-Type: {content_type}\r\n\r\n".encode()
                    + content_bytes
                    + b"\r\n"
                )
                parts.append(part)
            parts.append(f"--{boundary}--\r\n".encode())
            body = b"".join(parts)
            headers.append((b"content-type", f"multipart/form-data; boundary={boundary}".encode()))
            headers.append((b"content-length", str(len(body)).encode()))
            return body, headers
        return b"", headers


# Ensure tests importing ``fastapi.testclient`` receive our lightweight client.
module = types.ModuleType("fastapi.testclient")
module.TestClient = TestClient
module.__all__ = ["TestClient"]
sys.modules.setdefault("fastapi.testclient", module)
