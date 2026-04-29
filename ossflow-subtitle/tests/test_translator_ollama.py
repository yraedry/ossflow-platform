"""Unit tests for OllamaTranslator (mocked, no requiere Ollama corriendo)."""
from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from subtitle_generator.translator import OllamaTranslator, make_translator


def _mock_ollama_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"content": content}}
    return resp


def test_default_base_url_and_model():
    t = OllamaTranslator()
    assert t.base_url == "http://ollama:11434"
    assert t.model == "qwen2.5:7b-instruct-q4_K_M"
    assert t._endpoint_url() == "http://ollama:11434/api/chat"
    assert t.provider_label == "Ollama"


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    t = OllamaTranslator()
    assert t.base_url == "http://localhost:11434"


def test_request_headers_no_auth():
    t = OllamaTranslator()
    h = t._request_headers()
    assert h == {"Content-Type": "application/json"}
    assert "Authorization" not in h


def test_wrap_chat_body_json_mode_true():
    t = OllamaTranslator(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.2)
    msgs = [{"role": "user", "content": "hi"}]
    body = t._wrap_chat_body(msgs, json_mode=True)
    assert body == {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0.2},
        "format": "json",
    }


def test_wrap_chat_body_json_mode_false_omits_format():
    t = OllamaTranslator()
    body = t._wrap_chat_body([], json_mode=False)
    assert "format" not in body


def test_extract_message_content():
    t = OllamaTranslator()
    assert t._extract_message_content({"message": {"content": "hola"}}) == "hola"


def test_translate_texts_e2e_mocked():
    t = OllamaTranslator()
    fake = _mock_ollama_response('{"t":["hola"]}')
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake) as m:
        out = t.translate_texts(["hello"])
    assert out == ["hola"]
    args, kwargs = m.call_args
    assert args[0] == "http://ollama:11434/api/chat"
    body = kwargs["json_body"]
    assert body["format"] == "json"
    assert body["stream"] is False


def test_count_mismatch_retry():
    t = OllamaTranslator()
    short = _mock_ollama_response('{"t":["one"]}')
    correct = _mock_ollama_response('{"t":["one","two"]}')
    with patch(
        "subtitle_generator.translator._post_with_retry",
        side_effect=[short, correct],
    ) as m:
        out = t.translate_texts(["a", "b"])
    assert out == ["one", "two"]
    assert m.call_count == 2


def test_make_translator_ollama():
    t = make_translator("ollama")
    assert isinstance(t, OllamaTranslator)


def test_make_translator_ollama_with_custom_model():
    t = make_translator("ollama", model="qwen2.5:14b-instruct-q4_K_M")
    assert isinstance(t, OllamaTranslator)
    assert t.model == "qwen2.5:14b-instruct-q4_K_M"
