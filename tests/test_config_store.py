"""Tests for the in-memory ConfigStore (Phase 2b).

Seams tested:
  1. view -- no secret values; per-profile key_status OK/UNSET; derived tier; unsaved flag
  2. add_profile -- validates-by-construction; dup name rejected; store unchanged on error
  3. update_profile -- rewrites routing entries that referenced the old name
  4. delete_profile -- rejected when still routed; allowed otherwise
  5. duplicate_profile -- clones a profile under a new name (template)
  6. command CRUD + routing CRUD + run update
  7. frozen replacement -- mutating the returned config does not change the store
  8. save_to_toml -- persists, clears dirty, PRESERVES unknown [northstack.*] sections
  9. reload -- discards in-memory edits, clears dirty
  10. reset -- wipes to minimal default, marks dirty
  11. validate -- dry-run re-asserts the current state
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from northstack.config import (
    CommandConfig,
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RouteMapping,
    RunConfig,
    SecretEnvRef,
)
from northstack.interfaces.web.config_store import ConfigStore, key_status

# Fixtures


def _profile(name: str, *, roles=None, price_in=0.0, price_out=0.0, key="MY_KEY") -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost",
        model="m",
        api_key_env=SecretEnvRef(env_var=key) if key else None,
        roles=roles or {Role.WORKER},
        max_concurrency=1,
        input_price_per_million_usd=price_in,
        output_price_per_million_usd=price_out,
    )


@pytest.fixture
def toml_path(tmp_path: Path) -> Path:
    return tmp_path / "northstack.toml"


@pytest.fixture
def store(toml_path: Path) -> ConfigStore:
    """A store seeded from a file that also carries an unknown [northstack.workspace] section."""
    toml_path.write_text(
        '[northstack]\nname = "Co"\n\n'
        "[northstack.workspace]\nmax_list_entries = 1000\nmax_read_bytes = 5\n\n"
        "[northstack.run]\ndefault_budget_tokens = 200000\n\n"
        '[[northstack.profiles]]\nname = "p1"\nprotocol = "openai_chat"\n'
        'base_url = "http://localhost"\nmodel = "m"\nmax_concurrency = 2\n'
        'api_key_env = "MY_KEY"\nroles = ["worker"]\ncapabilities = ["tool_use"]\n',
        encoding="utf-8",
    )
    cfg = NorthStackConfig.from_toml(toml_path)
    return ConfigStore(cfg, toml_path)


@pytest.fixture(autouse=True)
def _env_keys():
    os.environ["MY_KEY"] = "sk-dummy"
    os.environ.pop("MISSING_KEY", None)
    yield
    os.environ.pop("MY_KEY", None)


# Reads


def test_view_no_secret_value_key_status_and_tier(store: ConfigStore) -> None:
    v = store.view()
    assert "sk-dummy" not in str(v)  # no secret value
    p1 = v["profiles"][0]
    assert p1["api_key_env"] == "MY_KEY"  # name only
    assert p1["key_status"] == "env:MY_KEY OK"
    assert p1["tier"] == 1  # derived from price 0
    assert "value" not in p1  # no leaked value field
    assert v["unsaved"] is False  # fresh, no edits


def test_key_status_unset() -> None:
    p = _profile("p", key="MISSING_KEY")
    assert key_status(p) == "env:MISSING_KEY UNSET"


def test_key_status_no_key() -> None:
    p = _profile("p", key=None)
    assert key_status(p) == "no key"


# Profile CRUD


def test_add_profile_rejects_dup_name_store_unchanged(store: ConfigStore) -> None:
    before = store.get()
    with pytest.raises(ValueError, match="already exists"):
        store.add_profile(_profile("p1"))  # dup
    assert store.get() == before  # unchanged
    assert store.unsaved() is False


def test_add_profile_valid(store: ConfigStore) -> None:
    store.add_profile(_profile("p2", roles={Role.REVIEWER}))
    assert [p["name"] for p in store.view()["profiles"]] == ["p1", "p2"]
    assert store.unsaved() is True


def test_update_profile_rewrites_routing_references(store: ConfigStore) -> None:
    # Route worker -> p1, then rename p1 -> p1-renamed.
    store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1"])])
    store.update_profile("p1", _profile("p1-renamed"))
    routing = store.view()["routing"]
    assert routing == [{"role": "worker", "profiles": ["p1-renamed"]}]
    assert [p["name"] for p in store.view()["profiles"]] == ["p1-renamed"]


def test_update_profile_unknown_raises(store: ConfigStore) -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        store.update_profile("nope", _profile("x"))


def test_delete_profile_rejected_when_routed(store: ConfigStore) -> None:
    store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1"])])
    with pytest.raises(ValueError, match="still routed"):
        store.delete_profile("p1")


def test_delete_profile_allowed_when_unrouted(store: ConfigStore) -> None:
    store.add_profile(_profile("p2"))
    store.delete_profile("p2")
    assert [p["name"] for p in store.view()["profiles"]] == ["p1"]


def test_delete_profile_can_remove_routing_atomically(store: ConfigStore) -> None:
    store.add_profile(_profile("fallback"))
    store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1", "fallback"])])

    store.delete_profile("p1", remove_from_routing=True)

    assert [p["name"] for p in store.view()["profiles"]] == ["fallback"]
    assert store.view()["routing"] == [{"role": "worker", "profiles": ["fallback"]}]


def test_delete_profile_drops_empty_routing_entry(store: ConfigStore) -> None:
    store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1"])])

    store.delete_profile("p1", remove_from_routing=True)

    assert store.view()["profiles"] == []
    assert store.view()["routing"] == []


def test_duplicate_profile(store: ConfigStore) -> None:
    store.duplicate_profile("p1", "p1-copy")
    names = [p["name"] for p in store.view()["profiles"]]
    assert names == ["p1", "p1-copy"]
    # the clone carries the original's roles/key reference (name only)
    copy = next(p for p in store.view()["profiles"] if p["name"] == "p1-copy")
    assert copy["api_key_env"] == "MY_KEY"
    assert "worker" in copy["roles"]


def test_duplicate_profile_unknown_source(store: ConfigStore) -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        store.duplicate_profile("nope", "x")


def test_duplicate_profile_name_collision(store: ConfigStore) -> None:
    with pytest.raises(ValueError, match="already exists"):
        store.duplicate_profile("p1", "p1")


# Command + routing + run


def test_command_crud_includes_isolation(store: ConfigStore) -> None:
    store.add_command(CommandConfig(name="lint", argv=["ruff", "check", "."]))
    store.add_command(
        CommandConfig(
            name="scan",
            argv=["python", "-m", "scan"],
            isolation="docker",
            docker_image="python:3.12-slim",
        )
    )
    cmds = store.view()["commands"]
    lint = next(c for c in cmds if c["name"] == "lint")
    assert lint["isolation"] == "host"
    assert lint["docker_image"] == ""
    scan = next(c for c in cmds if c["name"] == "scan")
    assert scan["isolation"] == "docker"
    assert scan["docker_image"] == "python:3.12-slim"
    store.update_command("lint", CommandConfig(name="lint2", argv=["ruff", "check"]))
    assert [c["name"] for c in store.view()["commands"]] == ["lint2", "scan"]
    store.delete_command("lint2")
    store.delete_command("scan")
    assert store.view()["commands"] == []


def test_command_dup_name_rejected(store: ConfigStore) -> None:
    store.add_command(CommandConfig(name="lint", argv=["ruff"]))
    with pytest.raises(ValueError, match="already exists"):
        store.add_command(CommandConfig(name="lint", argv=["ruff"]))


def test_routing_update_and_run(store: ConfigStore) -> None:
    store.add_profile(_profile("p2", roles={Role.REVIEWER}))
    store.update_routing(
        [
            RouteMapping(role=Role.WORKER, profiles=["p1"]),
            RouteMapping(role=Role.REVIEWER, profiles=["p2"]),
        ]
    )
    store.update_run(
        RunConfig(
            default_budget_tokens=50_000,
            default_budget_cost_usd=2.0,
            stall_window_seconds=45.0,
            planner_mode="model",
            falsifier_mode="model",
            calibration_path="cal.jsonl",
        )
    )
    v = store.view()
    assert len(v["routing"]) == 2
    assert v["run"]["default_budget_tokens"] == 50_000
    assert v["run"]["stall_window_seconds"] == 45.0
    assert v["run"]["planner_mode"] == "model"
    assert v["run"]["falsifier_mode"] == "model"
    assert v["run"]["calibration_path"] == "cal.jsonl"


def test_routing_to_unknown_profile_rejected_by_constructors(store: ConfigStore) -> None:
    # NorthStackConfig validators reject routing referencing a missing profile.
    with pytest.raises(Exception):
        store.update_routing([RouteMapping(role=Role.WORKER, profiles=["ghost"])])


# Frozen-immutability + persistence


def test_returned_config_is_independent_snapshot(store: ConfigStore) -> None:
    # The store hands out the current frozen config; subsequent edits replace
    # the internal reference rather than mutating the handed-out object, so
    # an already-captured snapshot stays stable and reflects the old state.
    snap_before = store.get()
    store.add_profile(_profile("p2"))
    snap_after = store.get()
    # The pre-edit snapshot still describes the old profile set...
    assert [p.name for p in snap_before.profiles] == ["p1"]
    # ...while the store now has both.
    assert [p.name for p in snap_after.profiles] == ["p1", "p2"]


def test_returned_config_cannot_mutate_store_nested_state(store: ConfigStore) -> None:
    snapshot = store.get()
    snapshot.profiles.clear()
    assert [p.name for p in store.get().profiles] == ["p1"]
    assert store.unsaved() is False


def test_constructor_detaches_mutable_input(toml_path: Path) -> None:
    config = NorthStackConfig(name="Detached", profiles=[_profile("p1")])
    store = ConfigStore(config, toml_path)
    config.profiles.clear()
    assert [p.name for p in store.get().profiles] == ["p1"]


def test_save_preserves_unknown_sections_and_clears_dirty(
    store: ConfigStore, toml_path: Path
) -> None:
    store.add_profile(_profile("p2"))
    store.save_to_toml()
    saved = toml_path.read_text("utf-8")
    assert "northstack.workspace]" in saved  # unknown section preserved
    assert "max_list_entries = 1000" in saved
    assert "default_budget_tokens = 200000" in saved  # modeled run preserved
    assert "p2" in saved
    assert "sk-dummy" not in saved  # no secret value
    assert store.unsaved() is False


def test_failed_atomic_save_preserves_file_and_dirty_state(
    store: ConfigStore, toml_path: Path, monkeypatch
) -> None:
    original = toml_path.read_text("utf-8")
    store.add_profile(_profile("p2"))

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr("northstack.adapters.atomic_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save_to_toml()

    assert toml_path.read_text("utf-8") == original
    assert store.unsaved() is True


def test_save_round_trips_back_through_from_toml(store: ConfigStore, toml_path: Path) -> None:
    store.add_profile(_profile("p2"))
    store.save_to_toml()
    reloaded = NorthStackConfig.from_toml(toml_path)
    assert [p.name for p in reloaded.profiles] == ["p1", "p2"]


def test_reload_discards_in_memory(store: ConfigStore) -> None:
    store.add_profile(_profile("p2"))
    store.update_name("Renamed")
    assert store.unsaved() is True
    store.reload()
    assert store.get().name == "Co"
    assert [p["name"] for p in store.view()["profiles"]] == ["p1"]
    assert store.unsaved() is False


def test_reload_reads_one_snapshot_and_preserves_unknown_sections(
    store: ConfigStore, monkeypatch
) -> None:
    import builtins

    original, calls = builtins.open, 0

    def fail_second_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("transient second read")
        return original(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_second_open)
    store.reload()
    assert calls == 1
    assert "northstack.workspace" in store.toml_document()


def test_reload_missing_file_raises(tmp_path: Path) -> None:
    cfg = NorthStackConfig(name="x")
    s = ConfigStore(cfg, tmp_path / "nope.toml")
    with pytest.raises(FileNotFoundError):
        s.reload()


def test_reset_clears_to_minimal(store: ConfigStore) -> None:
    store.reset()
    v = store.view()
    assert v["name"] == "Co"  # name retained
    assert v["profiles"] == []
    assert v["commands"] == []
    assert v["routing"] == []
    assert store.unsaved() is True


def test_toml_document_and_apply_round_trip(store: ConfigStore, toml_path: Path) -> None:
    doc = store.toml_document()
    assert "northstack.workspace]" in doc  # unknown sections stay in the document
    text = doc.replace('name = "Co"', 'name = "TomlCo"').replace(
        "default_budget_tokens = 200000", "default_budget_tokens = 1"
    )
    store.apply_toml(text)
    assert store.get().name == "TomlCo"
    assert store.unsaved() is True
    store.save_to_toml()
    saved = toml_path.read_text("utf-8")
    assert 'name = "TomlCo"' in saved
    assert "default_budget_tokens = 1" in saved
    assert "northstack.workspace]" in saved  # unknown section survives apply + save
    assert store.unsaved() is False


def test_apply_toml_rejects_bad_documents_and_keeps_store(store: ConfigStore) -> None:
    before = store.get()
    with pytest.raises(ValueError, match="invalid TOML"):
        store.apply_toml("not toml = = =")
    with pytest.raises(ValueError, match=r"\[northstack\]"):
        store.apply_toml("[other]\nx = 1")
    with pytest.raises(Exception):
        store.apply_toml('[northstack]\nname = "Co"\n[northstack.run]\nplanner_mode = "bogus"')
    assert store.get() == before  # unchanged after every failure
    assert store.unsaved() is False


def test_validate_is_noop_on_valid_state(store: ConfigStore) -> None:
    store.add_profile(_profile("p2"))
    store.validate()  # must not raise


def test_save_to_file_without_existing_unknown_sections(toml_path: Path) -> None:
    """Saving when the file has no unknown sections still works (no splice)."""
    toml_path.write_text(
        '[northstack]\nname = "Co"\n[[northstack.profiles]]\nname = "p1"\n'
        'protocol = "openai_chat"\nbase_url = "http://x"\nmodel = "m"\n'
        'max_concurrency = 1\nroles = ["worker"]\n',
        encoding="utf-8",
    )
    s = ConfigStore(NorthStackConfig.from_toml(toml_path), toml_path)
    s.add_profile(_profile("p2"))
    s.save_to_toml()
    reloaded = NorthStackConfig.from_toml(toml_path)
    assert [p.name for p in reloaded.profiles] == ["p1", "p2"]


def test_save_to_brand_new_file(toml_path: Path) -> None:
    """Saving when no file exists yet writes a valid config."""
    s = ConfigStore(NorthStackConfig(name="Brand"), toml_path)
    s.add_profile(_profile("p1"))
    s.save_to_toml()
    assert toml_path.exists()
    reloaded = NorthStackConfig.from_toml(toml_path)
    assert [p.name for p in reloaded.profiles] == ["p1"]


def test_concurrent_save_and_update_are_serializable(store: ConfigStore, toml_path: Path) -> None:
    gate = threading.Barrier(2)

    def save() -> None:
        gate.wait()
        store.save_to_toml()

    def update() -> None:
        gate.wait()
        store.update_name("Raced")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda fn: fn(), (save, update)))
    disk = NorthStackConfig.from_toml(toml_path)
    assert store.get().name == "Raced"
    assert (disk.name, store.unsaved()) in {("Co", True), ("Raced", False)}


def test_concurrent_reload_and_update_are_serializable(store: ConfigStore) -> None:
    gate = threading.Barrier(2)

    def reload() -> None:
        gate.wait()
        store.reload()

    def update() -> None:
        gate.wait()
        store.update_name("Raced")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda fn: fn(), (reload, update)))
    assert (store.get().name, store.unsaved()) in {("Co", False), ("Raced", True)}


def test_routing_update_racing_with_profile_deletion_stays_valid(store: ConfigStore) -> None:
    store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1"])])
    gate = threading.Barrier(2)

    def delete() -> ValueError | None:
        gate.wait()
        try:
            store.delete_profile("p1", remove_from_routing=True)
        except ValueError as exc:
            return exc
        return None

    def route() -> ValueError | None:
        gate.wait()
        try:
            store.update_routing([RouteMapping(role=Role.WORKER, profiles=["p1"])])
        except ValueError as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        errors = list(pool.map(lambda fn: fn(), (delete, route)))
    config = store.get()
    assert config.profile("p1") is None
    assert all(config.profile(name) for entry in config.routing for name in entry.profiles)
    assert sum(error is not None for error in errors) <= 1
