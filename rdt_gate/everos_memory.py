from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


DEFAULT_EVEROS_BASE_URL = "https://api.evermind.ai"
DEFAULT_MEMORY_TYPES = ["episodic_memory", "profile", "agent_memory"]


class EverOSMemoryError(RuntimeError):
    pass


class EverOSMemoryClient:
    """Small EverOS adapter with SDK-first behavior and stdlib HTTP fallback."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 20.0) -> None:
        self.api_key = api_key or os.environ.get("EVEROS_API_KEY")
        if not self.api_key:
            raise EverOSMemoryError("EVEROS_API_KEY is not set.")

        self.base_url = (base_url or os.environ.get("EVEROS_API_BASE") or DEFAULT_EVEROS_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.backend = "rest"
        self._sdk_client = None

        try:
            from everos import EverOS
        except Exception:
            return

        self._sdk_client = EverOS(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        self.backend = "sdk"

    def add_agent_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        async_mode: bool = True,
    ) -> Dict[str, Any]:
        if self._sdk_client is not None:
            response = self._sdk_client.v1.memories.agent.add(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                async_mode=async_mode,
                timeout=self.timeout,
            )
            return _to_plain(response)

        return self._post(
            "/api/v1/memories/agent",
            {
                "user_id": user_id,
                "session_id": session_id,
                "messages": messages,
                "async_mode": async_mode,
            },
        )

    def flush_agent_memories(self, user_id: str, session_id: str) -> Dict[str, Any]:
        if self._sdk_client is not None:
            response = self._sdk_client.v1.memories.agent.flush(
                user_id=user_id,
                session_id=session_id,
                timeout=self.timeout,
            )
            return _to_plain(response)

        return self._post(
            "/api/v1/memories/agent/flush",
            {"user_id": user_id, "session_id": session_id},
        )

    def search_user_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> Dict[str, Any]:
        if self._sdk_client is not None:
            response = self._sdk_client.v1.memories.search(
                filters={"user_id": user_id},
                query=query,
                method=method,
                memory_types=DEFAULT_MEMORY_TYPES,
                top_k=top_k,
                timeout=self.timeout,
            )
            return _to_plain(response)

        return self._post(
            "/api/v1/memories/search",
            {
                "filters": {"user_id": user_id},
                "query": query,
                "method": method,
                "memory_types": DEFAULT_MEMORY_TYPES,
                "top_k": top_k,
            },
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EverOSMemoryError(f"EverOS HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EverOSMemoryError(f"EverOS request failed: {exc}") from exc


def make_everos_session_id(video_label: str, output_dir: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    source = os.path.splitext(os.path.basename(video_label))[0] or "synthetic"
    out = os.path.basename(os.path.normpath(output_dir)) or "outputs"
    return f"rdt_gate:{_slug(source)}:{_slug(out)}:{stamp}"


def sync_experiment_run(
    *,
    user_id: str,
    session_id: str,
    video_label: str,
    output_dir: str,
    clip_seconds: float,
    frames_per_clip: int,
    embedding_backend: str,
    prototype_params: Dict[str, Any],
    adjacent_params: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    report: str,
    search_query: str | None = None,
    search_top_k: int = 5,
    flush: bool = True,
) -> Dict[str, Any]:
    client = EverOSMemoryClient()
    status: Dict[str, Any] = {
        "enabled": True,
        "backend": client.backend,
        "user_id": user_id,
        "session_id": session_id,
        "saved": False,
        "flushed": False,
        "context_count": 0,
        "warnings": [],
    }

    if search_query:
        search_response = client.search_user_memories(user_id=user_id, query=search_query, top_k=search_top_k)
        context_lines = format_search_context(search_response)
        status["context_count"] = len(context_lines)
        context_path = os.path.join(output_dir, "everos_context.md")
        with open(context_path, "w", encoding="utf-8") as f:
            f.write("# EverOS Retrieved Context\n\n")
            f.write(f"Query: `{search_query}`\n\n")
            if context_lines:
                f.write("\n\n".join(context_lines))
                f.write("\n")
            else:
                f.write("No prior context returned.\n")
        status["context_path"] = context_path

    messages = build_experiment_messages(
        video_label=video_label,
        clip_seconds=clip_seconds,
        frames_per_clip=frames_per_clip,
        embedding_backend=embedding_backend,
        prototype_params=prototype_params,
        adjacent_params=adjacent_params,
        metrics=metrics,
        report=report,
    )
    add_response = client.add_agent_messages(user_id=user_id, session_id=session_id, messages=messages)
    add_data = _get_data(add_response)
    status["saved"] = True
    status["add_status"] = _get_field(add_data, "status")
    status["task_id"] = _get_field(add_data, "task_id")
    status["message_count"] = _get_field(add_data, "message_count")

    if flush:
        flush_response = client.flush_agent_memories(user_id=user_id, session_id=session_id)
        flush_data = _get_data(flush_response)
        status["flushed"] = True
        status["flush_status"] = _get_field(flush_data, "status")

    return status


def build_experiment_messages(
    *,
    video_label: str,
    clip_seconds: float,
    frames_per_clip: int,
    embedding_backend: str,
    prototype_params: Dict[str, Any],
    adjacent_params: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    report: str,
) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    config = {
        "video": video_label,
        "clip_seconds": clip_seconds,
        "frames_per_clip": frames_per_clip,
        "embedding_backend": embedding_backend,
        "prototype_params": prototype_params,
        "adjacent_params": adjacent_params,
    }
    return [
        {
            "role": "user",
            "timestamp": now_ms,
            "content": (
                "Run a Fast-Slow RDT-Gate experiment and remember the setup for future threshold tuning.\n\n"
                f"Configuration:\n{json.dumps(config, ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "assistant",
            "timestamp": now_ms + 1000,
            "content": (
                "Fast-Slow RDT-Gate experiment completed.\n\n"
                f"Metrics:\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
                f"Report:\n{report}"
            ),
        },
    ]


def format_search_context(response: Dict[str, Any]) -> List[str]:
    data = _get_data(response)
    lines: List[str] = []
    for key in ("episodes", "profiles", "agent_memories", "raw_messages", "results"):
        items = _get_field(data, key) or []
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            title = _first_present(item, ("subject", "name", "task_intent", "id")) or f"{key} {index}"
            body = _first_present(
                item,
                ("episode", "summary", "content", "approach", "profile_data", "original_data"),
            )
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, indent=2)
            lines.append(f"## {key} / {title}\n\n{body or json.dumps(item, ensure_ascii=False, indent=2)}")
    return lines


def _slug(value: str) -> str:
    chars = []
    for char in value.lower():
        chars.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(chars).split("_") if part)[:48] or "run"


def _to_plain(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {"data": value}


def _get_data(response: Any) -> Any:
    return _get_field(response, "data") or response


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _first_present(value: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        item = _get_field(value, name)
        if item:
            return item
    return None
