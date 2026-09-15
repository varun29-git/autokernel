"""OpenAI-compatible coding-model adapter for the command-generator protocol.

It is deliberately a separate process. Credentials reach this process only;
candidate build/evaluation processes receive a scrubbed environment.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .agent import candidate_from_response, system_prompt, user_prompt


def completion(request, *, endpoint, api_key, model, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(request)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    wire = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    http_request = urllib.request.Request(endpoint, data=wire, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"model provider request failed: {type(error).__name__}") from error
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("model provider returned no assistant content") from error
    if not isinstance(content, str):
        raise ValueError("model provider returned non-text assistant content")
    return content


def main():
    request = json.load(sys.stdin)
    api_key = os.environ.get("AUTOKERNEL_MODEL_API_KEY")
    model = os.environ.get("AUTOKERNEL_MODEL_NAME")
    endpoint = os.environ.get("AUTOKERNEL_MODEL_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    if not api_key or not model:
        raise SystemExit("AUTOKERNEL_MODEL_API_KEY and AUTOKERNEL_MODEL_NAME are required")
    try:
        timeout = float(os.environ.get("AUTOKERNEL_MODEL_TIMEOUT_SECONDS", "120"))
    except ValueError as error:
        raise SystemExit("AUTOKERNEL_MODEL_TIMEOUT_SECONDS must be numeric") from error
    if not 1 <= timeout <= 600:
        raise SystemExit("AUTOKERNEL_MODEL_TIMEOUT_SECONDS must be between 1 and 600")
    response = completion(request, endpoint=endpoint, api_key=api_key, model=model, timeout=timeout)
    candidate, source = candidate_from_response(request, response)
    print(json.dumps({"candidate": candidate.model_dump(mode="json"), "source": source.decode()}, allow_nan=False))


if __name__ == "__main__":
    main()
