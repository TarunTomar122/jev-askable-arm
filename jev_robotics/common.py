from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines without adding python-dotenv."""
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop runner-only keys before a Jev API call."""
    return {key: jsonable(value) for key, value in state.items() if not str(key).startswith("_")}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if type(value).__module__ not in {"builtins", None} and hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            return str(value)
    return value


@dataclass
class Candidate:
    id: str
    skill: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    selected: Candidate
    confidence: float
    candidate_confidence: float
    latency_ms: float
    answers: dict[str, Any]
    raw_choice: str
    overridden_for_confidence: bool = False


class JEVClient:
    """Stdlib client for TypeSafe System One (Jev)."""

    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        timeout: float = 20.0,
        retries: int = 3,
    ):
        load_local_env()
        self.api_key = (
            api_key
            or os.environ.get("JEV_API_KEY")
            or os.environ.get("TYPESAFE_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError("JEV_API_KEY or TYPESAFE_API_KEY is missing")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def evaluate(
        self, state: dict[str, Any], questions: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": self.model,
            "state": public_state(state) if isinstance(state, dict) else jsonable(state),
            "questions": questions,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                return json.loads(body), (time.perf_counter() - started) * 1000.0
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in (429, 529) or attempt >= self.retries:
                    raise RuntimeError(f"JEV HTTP {exc.code}: {body[:500]}") from exc
                time.sleep(0.25 * (2**attempt))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"JEV request failed: {exc}") from exc
                time.sleep(0.25 * (2**attempt))
        raise RuntimeError("JEV request failed without a response")
