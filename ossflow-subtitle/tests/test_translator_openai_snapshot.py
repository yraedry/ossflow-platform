"""Snapshot tests for OpenAITranslator pre-refactor.

Estos tests fijan el comportamiento ANTES de extraer _BaseChatTranslator.
Deben seguir verdes después del refactor sin modificación.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from subtitle_generator.translator import OpenAITranslator


def _mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


def test_translate_texts_calls_openai_with_correct_body():
    t = OpenAITranslator(api_key="sk-test", model="gpt-4o-mini")
    fake_resp = _mock_openai_response('{"t":["hola mundo"]}')

    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        out = t.translate_texts(["hello world"])

    assert out == ["hola mundo"]
    args, kwargs = m.call_args
    assert args[0] == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    body = kwargs["json_body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert len(body["messages"]) == 2


def test_translate_for_dubbing_uses_budget_prompt():
    t = OpenAITranslator(api_key="sk-test")
    fake_resp = _mock_openai_response('{"t":["coge el grip"]}')

    items = [{"text": "Get your grip.", "duration_ms": 1500}]
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        out = t.translate_for_dubbing(items, cps=17.0)

    assert out == ["coge el grip"]
    body = m.call_args.kwargs["json_body"]
    system_prompt = body["messages"][0]["content"]
    assert "DUBBING" in system_prompt
    assert "max_chars" in system_prompt or "target_chars" in system_prompt


def test_translate_for_dubbing_fill_budget_uses_fill_prompt():
    t = OpenAITranslator(api_key="sk-test")
    fake_resp = _mock_openai_response('{"t":["coge el grip, fíjate"]}')

    items = [{"text": "Get your grip.", "duration_ms": 1500}]
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        t.translate_for_dubbing(items, cps=17.0, fill_budget=True)

    body = m.call_args.kwargs["json_body"]
    system_prompt = body["messages"][0]["content"]
    assert "target_chars" in system_prompt
    assert "FILL THE SLOT" in system_prompt
