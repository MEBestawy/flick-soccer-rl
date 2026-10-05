"""
Thin HTTP client for the TypeSafe / Jev System One API.

Uses the official endpoint for `apikey_...` keys:
  POST https://api.typesafe.ai/v1/systemone
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import httpx


DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevClientError(RuntimeError):
    """Raised when the Jev API call fails."""


class JevClient:
    """Minimal typed-decision client (no SDK required)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("TYPESAFE_API_KEY")
            or os.environ.get("JEV_API_KEY")
            or ""
        )
        if not self.api_key:
            raise JevClientError(
                "Missing TYPESAFE_API_KEY / JEV_API_KEY. "
                "Set it in the environment or .env file."
            )

        self.endpoint = (
            endpoint
            or os.environ.get("JEV_API_ENDPOINT")
            or DEFAULT_ENDPOINT
        )
        self.model = model
        self.timeout = timeout

    def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Call System One and return the answers map.

        Args:
            state: Text or structured state for Jev to evaluate.
            questions: Map of question_id → {type, instructions, criteria?}

        Returns:
            The `answers` object from the API response.
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": dict(questions),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise JevClientError(f"Jev request failed: {exc}") from exc

        if response.status_code >= 400:
            raise JevClientError(
                f"Jev API {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise JevClientError(f"Unexpected Jev response shape: {data!r}")
        return answers
