"""Tests for long-term memory: the SQLite FTS5 store and its cell-runner wiring.

Covers:
  - Namespaces are isolated; recall ranks by relevance
  - Re-remembering a fact bumps its hit count instead of duplicating it
  - Free text is never parsed as FTS5 query syntax
  - Credential-shaped text is refused, not stored
  - A recall failure degrades to no prompt, never to a failed cell
"""

from __future__ import annotations

from typing import Any

import pytest

from northstack.adapters.sqlite_memory import MAX_MEMORY_BYTES, SqliteMemory
from northstack.application.cell_runner import CellRunner
from northstack.domain import Budget, GraphCell, WorkContract


@pytest.fixture
def memory(tmp_path):
    with SqliteMemory(tmp_path / "memory.db") as store:
        yield store


class TestRemember:
    def test_a_fact_comes_back(self, memory):
        memory.remember("acme", "The deploy script needs DATABASE_URL set", source="run-1")
        (found,) = memory.recall("acme", "deploy script")
        assert found.text == "The deploy script needs DATABASE_URL set"
        assert found.source == "run-1"
        assert found.hits == 1

    def test_the_same_fact_twice_is_one_record_with_two_hits(self, memory):
        memory.remember("acme", "pytest needs -p no:randomly here")
        again = memory.remember("acme", "  pytest needs -p no:randomly here  ")
        assert again.hits == 2
        assert len(memory.recall("acme", "pytest randomly")) == 1

    def test_namespaces_do_not_leak(self, memory):
        memory.remember("acme", "acme uses poetry")
        memory.remember("other", "other uses pip")
        assert [m.text for m in memory.recall("acme", "uses")] == ["acme uses poetry"]

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_text_is_not_a_memory(self, memory, text):
        assert memory.remember("acme", text) is None

    def test_oversized_text_is_truncated_not_rejected(self, memory):
        record = memory.remember("acme", "y" * (MAX_MEMORY_BYTES + 500))
        assert len(record.text) == MAX_MEMORY_BYTES

    @pytest.mark.parametrize(
        "text",
        [
            "use Authorization: Bearer abc123 for the API",
            "the key is sk-aaaaaaaaaaaaaaaa",
            "token ghp_bbbbbbbbbbbbbbbb works",
        ],
    )
    def test_credential_shaped_text_is_refused(self, memory, text):
        assert memory.remember("acme", text) is None
        assert memory.recall("acme", "key token api bearer") == []


class TestRecall:
    def test_relevance_beats_insertion_order(self, memory):
        memory.remember("acme", "unrelated note about invoices")
        memory.remember("acme", "the migration script requires a schema lock")
        assert memory.recall("acme", "migration schema")[0].text.startswith("the migration")

    def test_repeated_confirmation_breaks_a_tie(self, memory):
        memory.remember("acme", "flaky test: retry once")
        memory.remember("acme", "flaky test: rerun twice")
        memory.remember("acme", "flaky test: rerun twice")
        assert memory.recall("acme", "flaky test")[0].text.endswith("rerun twice")

    def test_the_limit_is_honoured(self, memory):
        for i in range(10):
            memory.remember("acme", f"note number {i} about caching")
        assert len(memory.recall("acme", "caching", limit=3)) == 3

    def test_an_empty_query_recalls_nothing(self, memory):
        memory.remember("acme", "something")
        assert memory.recall("acme", "   ") == []

    @pytest.mark.parametrize("query", ['a "quoted" phrase', "NEAR AND OR", "co-located (x)", "*"])
    def test_free_text_is_never_fts_syntax(self, memory, query):
        """A user objective is prose, not a query. Unescaped, ``AND`` or a
        stray quote raises inside SQLite instead of searching.
        """
        memory.remember("acme", "a quoted phrase about co-located services")
        assert isinstance(memory.recall("acme", query), list)


class TestForget:
    def test_forget_drops_only_its_namespace(self, memory):
        memory.remember("acme", "acme fact")
        memory.remember("other", "other fact")
        assert memory.forget("acme") == 1
        assert memory.recall("acme", "fact") == []
        assert len(memory.recall("other", "fact")) == 1

    def test_a_forgotten_fact_can_be_remembered_again(self, memory):
        memory.remember("acme", "reusable fact")
        memory.forget("acme")
        assert memory.remember("acme", "reusable fact").hits == 1


class _Store:
    """A memory that records calls and can be told to fail."""

    def __init__(self, *, recalls: list[Any] | None = None, explode: bool = False) -> None:
        self._recalls = recalls or []
        self._explode = explode
        self.written: list[tuple[str, str, str]] = []

    def recall(self, namespace: str, query: str, *, limit: int = 5) -> list[Any]:
        if self._explode:
            raise RuntimeError("memory is corrupt")
        return self._recalls

    def remember(self, namespace: str, text: str, *, source: str = "") -> Any:
        if self._explode:
            raise RuntimeError("memory is read-only")
        self.written.append((namespace, text, source))
        return None


def _runner(store: Any) -> CellRunner:
    return CellRunner(
        worker=None,
        router=None,
        retry_policy=None,
        recovery=None,
        artifact_store=None,
        memory=store,
        memory_namespace="acme",
    )


def _cell() -> GraphCell:
    return GraphCell(
        id="cell-1",
        name="Build it",
        contract=WorkContract(
            id="wc-1",
            objective="add retries to the uploader",
            budget=Budget(token_limit=1000, cost_limit_usd=1.0),
        ),
    )


class TestCellRunnerWiring:
    def test_recalled_facts_become_an_advisory_prompt(self, memory):
        memory.remember("acme", "the uploader retries three times")
        prompt = _runner(memory)._recalled_prompt(_cell())
        assert "advisory" in prompt
        assert "- the uploader retries three times" in prompt

    def test_no_memory_configured_is_an_empty_prompt(self):
        assert _runner(None)._recalled_prompt(_cell()) == ""

    def test_nothing_recalled_is_an_empty_prompt(self, memory):
        assert _runner(memory)._recalled_prompt(_cell()) == ""

    def test_a_broken_store_does_not_stop_the_cell(self):
        """Recall is an optimisation; a corrupt store must cost context, not
        the run.
        """
        assert _runner(_Store(explode=True))._recalled_prompt(_cell()) == ""

    def test_a_broken_store_does_not_fail_a_succeeded_cell(self):
        _runner(_Store(explode=True))._remember_outcome(
            _cell(), "run-1", type("R", (), {"text": "done"})()
        )

    def test_the_outcome_is_written_with_the_run_as_its_source(self):
        store = _Store()
        _runner(store)._remember_outcome(_cell(), "run-7", type("R", (), {"text": "done"})())
        (namespace, text, source) = store.written[0]
        assert namespace == "acme"
        assert source == "run-7"
        assert "add retries to the uploader" in text
        assert "done" in text


class TestBuildWiring:
    """``memory_enabled`` is the only thing standing between the store and the
    cell prompt, so the toggle is worth an end-to-end assertion.
    """

    def _components(self, tmp_path, **run_kw):
        from unittest.mock import MagicMock, patch

        from northstack.application.build import build_company
        from northstack.config import ModelProfile, NorthStackConfig, Protocol, RunConfig

        config = NorthStackConfig(
            name="wired",
            profiles=[
                ModelProfile(
                    name="cheap",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost/v1",
                    model="cheap",
                    max_concurrency=2,
                )
            ],
            run=RunConfig(**run_kw),
        )
        with patch("northstack.application.build.ModelGateway", MagicMock()):
            return build_company(config, tmp_path)

    def test_enabled_opens_a_store_beside_the_ledger(self, tmp_path):
        components = self._components(tmp_path, memory_enabled=True)
        try:
            assert isinstance(components.memory, SqliteMemory)
            assert components.company._memory is components.memory
            assert (tmp_path / ".northstack" / "memory.db").exists()
        finally:
            components.close()

    def test_off_by_default(self, tmp_path):
        components = self._components(tmp_path)
        try:
            assert components.memory is None
            assert components.company._memory is None
            assert not (tmp_path / ".northstack" / "memory.db").exists()
        finally:
            components.close()
