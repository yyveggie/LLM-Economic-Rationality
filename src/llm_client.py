"""Unified LLM client supporting OpenAI / Anthropic / OpenAI-compatible."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class ModelConfig:
    key: str                    # safe-for-filesystem id derived from `model`
    provider: str               # openai | anthropic | openai_compatible
    model: str
    base_url: str
    api_key: str
    default_temperature: float
    max_tokens: int
    request_timeout: int
    concurrency: int

    @classmethod
    def from_yaml(cls, raw: dict) -> "ModelConfig":
        """Parse the single-block models.yaml (top-level fields)."""
        model_id = str(raw["model"])
        # Filesystem-safe key: replace path separators
        key = model_id.replace("/", "__").replace(":", "_")
        return cls(
            key=key,
            provider=raw["provider"],
            model=model_id,
            base_url=raw.get("base_url", "") or "",
            api_key=str(raw.get("api_key", "") or ""),
            default_temperature=float(raw.get("default_temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 256)),
            request_timeout=int(raw.get("request_timeout", 60)),
            concurrency=int(raw.get("concurrency", 4)),
        )


class LLMClient:
    """Thin wrapper that hides provider-specific SDK differences."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._client = None
        self._init_client()

    def _api_key(self) -> Optional[str]:
        return self.cfg.api_key or None

    def _init_client(self):
        provider = self.cfg.provider
        if provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            kwargs = {"api_key": self._api_key() or "EMPTY",
                      "timeout": self.cfg.request_timeout}
            if provider == "openai_compatible" and self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            self._client = OpenAI(**kwargs)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key(),
                                     timeout=self.cfg.request_timeout)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @retry(stop=stop_after_attempt(4),
           wait=wait_random_exponential(min=1, max=20),
           reraise=True)
    def complete(self,
                 system: str,
                 messages: List[Dict[str, str]],
                 temperature: Optional[float] = None) -> str:
        """messages = [{role: 'assistant'|'user', content: ...}, ...]

        For Chen et al.'s protocol, the assistant's background message is sent
        as a pre-filled assistant turn followed by the user's task prompt.
        """
        temp = self.cfg.default_temperature if temperature is None else temperature
        if self.cfg.provider in ("openai", "openai_compatible"):
            full = [{"role": "system", "content": system}] + messages
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=full,
                temperature=temp,
                max_tokens=self.cfg.max_tokens,
            )
            return resp.choices[0].message.content or ""
        if self.cfg.provider == "anthropic":
            # Anthropic requires the messages array to start with a user turn.
            # We fold any leading assistant "background" content into the system
            # prompt so the conversation begins with the actual task user-msg.
            sys_parts = [system] if system else []
            anthro_msgs = []
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if not content:
                    continue
                if role == "assistant" and not anthro_msgs:
                    sys_parts.append(content)
                elif role in ("user", "assistant"):
                    anthro_msgs.append({"role": role, "content": content})
            if not anthro_msgs:
                anthro_msgs = [{"role": "user", "content": ""}]
            resp = self._client.messages.create(
                model=self.cfg.model,
                system="\n\n".join(sys_parts),
                messages=anthro_msgs,
                max_tokens=self.cfg.max_tokens,
                temperature=temp,
            )
            chunks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "".join(chunks)
        raise ValueError(f"Unknown provider: {self.cfg.provider}")
