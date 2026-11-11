"""Append-only API token-usage logging for the pinned R-Bot runner."""

from __future__ import annotations

import itertools
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CALL_IDS = itertools.count(1)
_LOCK = threading.Lock()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    for method in ("model_dump", "to_dict", "dict"):
        function = getattr(value, method, None)
        if callable(function):
            try:
                return _plain(function())
            except Exception:
                pass
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return _plain({key: item for key, item in data.items() if not key.startswith("_")})
    return str(value)


def _nested(data: dict, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def normalize_usage(usage: Any) -> dict[str, Any]:
    raw = _plain(usage)
    data = raw if isinstance(raw, dict) else {}
    input_tokens = _nested(data, ("input_tokens",), ("prompt_tokens",))
    output_tokens = _nested(data, ("output_tokens",), ("completion_tokens",))
    reasoning_tokens = _nested(
        data,
        ("reasoning_tokens",),
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    )
    cached_input_tokens = _nested(
        data,
        ("cached_input_tokens",),
        ("cache_read_input_tokens",),
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
    )
    cache_write_tokens = _nested(
        data, ("cache_write_tokens",), ("cache_creation_input_tokens",)
    )
    total_tokens = _nested(data, ("total_tokens",))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)
    return {
        "usage_available": bool(data),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "embedding_tokens": None,
        "total_tokens": total_tokens,
        "raw_usage": raw if data else None,
    }


def _response_usage(response: Any) -> Any:
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict) and additional.get("usage") is not None:
        return additional["usage"]
    raw = getattr(response, "raw", None)
    if raw is not None:
        usage = getattr(raw, "usage", None)
        if usage is not None:
            return usage
        if isinstance(raw, dict) and raw.get("usage") is not None:
            return raw["usage"]
    return None


def record_response(response: Any, *, call_kind: str) -> None:
    path_value = os.environ.get("TOKEN_USAGE_LOG")
    if not path_value:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_id": next(_CALL_IDS),
        "system": os.environ.get("TOKEN_USAGE_SYSTEM", "rbot"),
        "model": os.environ.get("TOKEN_USAGE_MODEL"),
        "workload": os.environ.get("TOKEN_USAGE_WORKLOAD"),
        "query_id": os.environ.get("TOKEN_USAGE_QUERY_ID", "unknown"),
        "stage": os.environ.get("TOKEN_USAGE_STAGE", "rewrite"),
        "call_kind": call_kind,
        "usage_source": "api_response",
        "usage_is_estimated": False,
    }
    record.update(normalize_usage(_response_usage(response)))
    if not record["usage_available"]:
        record["usage_source"] = "unavailable"
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
