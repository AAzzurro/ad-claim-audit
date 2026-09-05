"""本地小模型推理：默认走本机 Ollama（任务书要求小规模 LM，禁止 28B）。"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://127.0.0.1:11434"


class LLMError(RuntimeError):
    pass


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LLMError(f"无法连接 Ollama（{url}）：{exc}") from exc


def list_ollama_models(host: str = DEFAULT_HOST) -> list[str]:
    body = _request_json(f"{host.rstrip('/')}/api/tags", timeout=10)
    names: list[str] = []
    for item in body.get("models") or []:
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def ensure_model(model: str, host: str = DEFAULT_HOST) -> None:
    names = list_ollama_models(host)
    wanted = {model, f"{model}:latest"}
    if any(n in wanted or n.startswith(f"{model}-") or n.startswith(f"{model}:") for n in names):
        return
    raise LLMError(
        f"Ollama 没有模型 {model}。已安装：{names or '（空）'}。"
        f"请执行: ollama pull {model}"
    )


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    timeout: float = 180.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": 512,
        },
    }
    body = _request_json(f"{host.rstrip('/')}/api/chat", payload, timeout=timeout)
    msg = body.get("message") or {}
    content = str(msg.get("content") or "").strip()
    if not content:
        raise LLMError(f"模型返回空内容: {body.get('error') or body}")
    return content
