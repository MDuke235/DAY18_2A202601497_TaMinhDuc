import io
from contextlib import redirect_stdout

import config
import openai

from src.m2_search import SearchResult
from src.pipeline import run_query


class SearchStub:
    def search(self, query):
        return [SearchResult("Relevant context", 1.0, {"source": "test"}, "hybrid")]


class RerankerStub:
    def rerank(self, query, documents, top_k):
        return []


def test_generation_failure_falls_back_on_windows_console(monkeypatch):
    class BrokenCompletions:
        def create(self, **kwargs):
            raise ConnectionError("offline")

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": BrokenCompletions()})()

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", Client)
    output = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    with redirect_stdout(output):
        answer, contexts = run_query("question", SearchStub(), RerankerStub())

    assert answer == "Relevant context"
    assert contexts == ["Relevant context"]
