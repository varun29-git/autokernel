import hashlib
import json

import pytest

from autokernel.agent import candidate_from_response, user_prompt
from autokernel.integrity import canonical
from autokernel.provider import completion


def request_for(campaign):
    return {
        "task": campaign.task.model_dump(mode="json"),
        "observations": {"candidates": [], "release_status": "not_qualified"},
        "candidate_schema": {"type": "object"},
    }


def test_agent_derives_digests_and_rejects_extra_fields(campaign):
    request = request_for(campaign)
    response = json.dumps({
        "backend": "numpy_demo", "family": "agent_add", "parameters": {"chunk": 128},
        "hypothesis": "Use one larger chunk to reduce loop overhead safely.",
        "source": '{"operation":"add"}', "parent_ids": [],
    })
    candidate, source = candidate_from_response(request, response)
    assert candidate.task_digest == hashlib.sha256(canonical(request["task"])).hexdigest()
    assert candidate.source_digest == hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError, match="exactly"):
        candidate_from_response(request, response[:-1] + ',"score":999}')


def test_agent_prompt_contains_only_protocol_fields(campaign):
    prompt = json.loads(user_prompt(request_for(campaign)))
    assert set(prompt) == {"task", "observations", "candidate_contract", "response_contract"}
    assert prompt["response_contract"]["source"] == "complete source only"


def test_openai_compatible_provider_extracts_json_content(monkeypatch, campaign):
    captured = {}

    class Reply:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return Reply()

    monkeypatch.setattr("autokernel.provider.urllib.request.urlopen", fake_open)
    content = completion(request_for(campaign), endpoint="https://provider.invalid/v1/chat/completions",
                         api_key="secret", model="kernel-model", timeout=17)
    assert content == '{"ok":true}'
    assert captured["body"]["model"] == "kernel-model"
    assert captured["authorization"] == "Bearer secret"
    assert captured["timeout"] == 17
