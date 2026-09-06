from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from light_learning import web_server


@pytest.fixture(autouse=True)
def reset_web_state():
    web_server._SESSIONS.clear()
    web_server._RL_BY_BUDGET.clear()
    web_server._EMERGENT = None
    web_server._METRICS_CACHE = None
    yield
    web_server._SESSIONS.clear()
    web_server._RL_BY_BUDGET.clear()
    web_server._EMERGENT = None
    web_server._METRICS_CACHE = None


@pytest.fixture
def handler() -> web_server.Handler:
    # The domain methods do not require a bound socket. Keeping these tests
    # in-process makes the information-boundary checks fast and deterministic.
    return object.__new__(web_server.Handler)


def _new_deterministic_session(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: int = 4,
) -> dict:
    secret_values = iter((987_654_321, 123_456_789, 555_000_111))
    public_ids = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(web_server.secrets, "randbits", lambda _bits: next(secret_values))
    monkeypatch.setattr(
        web_server.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(public_ids)),
    )
    return handler._new({"budget": budget})


def _finish_episode(handler: web_server.Handler, session_id: str, budget: int) -> dict:
    for slot in range(budget):
        result = handler._look({"session": session_id, "slot": slot})
        assert "error" not in result
    return handler._commit({"session": session_id, "slot": 12})


def test_new_session_exposes_only_opaque_ids_and_public_state(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)

    assert payload["session"] == "a" * 32
    assert payload["episode_id"] == "b" * 32
    assert {
        "seed",
        "mle_query_seed",
        "emergent_seed",
        "theta",
        "theta_hat",
        "absolute_error",
        "reward",
        "curve",
    }.isdisjoint(payload)
    stored = web_server._SESSIONS[payload["session"]]
    assert stored["seed"] == 987_654_321
    assert stored["mle_query_seed"] == 123_456_789
    assert stored["emergent_seed"] == 555_000_111
    assert stored["seed"] != stored["mle_query_seed"] != stored["emergent_seed"]


def test_clients_cannot_supply_a_room_seed(handler: web_server.Handler) -> None:
    with pytest.raises(web_server.ApiError, match="server-owned") as caught:
        handler._new({"budget": 4, "seed": 7})
    assert caught.value.status == 400


@pytest.mark.parametrize("operation", ["mle", "rl", "emergent"])
def test_comparisons_are_rejected_before_original_episode_terminates(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)
    if operation == "rl":
        web_server._RL_BY_BUDGET[4] = {"budget": 4}

    call = {
        "mle": handler._mle,
        "rl": handler._rl_play,
        "emergent": handler._emergent_play,
    }[operation]
    with pytest.raises(web_server.ApiError, match="commit this episode") as caught:
        call({"session": payload["session"]})
    assert caught.value.status == 409


def test_http_handler_maps_terminal_gate_to_conflict_without_binding_socket(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)
    responses: list[tuple[int, dict]] = []
    handler.path = "/api/mle"
    handler._read_json = lambda: {"session": payload["session"]}  # type: ignore[method-assign]
    handler._json = lambda code, body: responses.append((code, body))  # type: ignore[method-assign]

    handler.do_POST()

    assert responses == [
        (409, {"error": "commit this episode before revealing comparison results"})
    ]


def test_terminal_reveal_prevents_any_more_episode_actions(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)
    committed = _finish_episode(handler, payload["session"], budget=4)

    assert committed["done"] is True
    assert "theta" in committed
    history_length = len(committed["history"])
    late_look = handler._look({"session": payload["session"], "slot": 10})
    second_commit = handler._commit({"session": payload["session"], "slot": 10})
    assert late_look["error"] == "episode already finished"
    assert second_commit["error"] == "episode already finished"
    assert len(late_look["history"]) == history_length


def test_mle_comparison_runs_only_after_commit_with_independent_query_seed(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)
    _finish_episode(handler, payload["session"], budget=4)

    comparison = handler._mle({"session": payload["session"]})

    assert comparison["session"] == payload["session"]
    assert comparison["condition"] == "oracle_mle"
    assert len(comparison["trace"]) == 5
    assert comparison["trace"][-1]["phase"] == "terminal"
    assert "theta" in comparison
    assert "curve" in comparison


def test_emergent_play_runs_only_after_commit_without_the_formula(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch)
    _finish_episode(handler, payload["session"], budget=4)

    comparison = handler._emergent_play({"session": payload["session"]})

    assert comparison["session"] == payload["session"]
    assert comparison["condition"] == "emergent_in_episode"
    assert comparison["canonical_formula_disclosed"] is False
    assert len(comparison["trace"]) == 5
    assert comparison["trace"][-1]["phase"] == "terminal"
    assert 0 <= comparison["theta_hat"] < 32
    assert len(comparison["inferred_curve"]) == 32
    assert comparison["sigma"] in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0}


def test_reinforce_policy_must_match_episode_budget(
    handler: web_server.Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _new_deterministic_session(handler, monkeypatch, budget=4)
    _finish_episode(handler, payload["session"], budget=4)
    web_server._RL_BY_BUDGET[8] = {"budget": 8, "W": [[0.0]]}

    with pytest.raises(web_server.ApiError, match="no REINFORCE policy for budget 4") as caught:
        handler._rl_play({"session": payload["session"]})
    assert caught.value.status == 409


def test_demo_metrics_are_explicitly_non_reportable() -> None:
    metrics = web_server._demo_mle_metrics()

    assert metrics["reportable"] is False
    assert metrics["evaluation_label"] == "non_reportable_demo_smoke"
    assert set(metrics["budgets"]) == {"4", "8", "16", "32"}
    assert all(row["reportable"] is False for row in metrics["budgets"].values())
    assert "official benchmark results" in metrics["warning"]

    web_server._RL_BY_BUDGET[8] = {
        "budget": 8,
        "n_episodes": 2,
        "history": [],
        "eval_mae": 3.0,
        "eval_hit_within_one": 0.25,
        "algo": "reinforce",
        "note": "Small demo.",
        "train_seed": 0,
    }
    public = web_server._rl_public(8)
    assert public["reportable"] is False
    assert public["evaluation_label"] == "non_reportable_demo_smoke"
    assert "demo_mae" in public
    assert "eval_mae" not in public


def test_web_root_is_inside_the_installed_package() -> None:
    expected = Path(web_server.__file__).resolve().parent / "web"
    assert web_server.WEB_ROOT == expected
    assert (expected / "index.html").is_file()
    assert {"index.html", "app.js", "style.css"} <= {
        path.name for path in expected.iterdir()
    }


def test_http_root_serves_the_explainer() -> None:
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    class ReuseServer(ThreadingHTTPServer):
        allow_reuse_address = True

    httpd = ReuseServer(("127.0.0.1", 0), web_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            body = response.read()
            assert response.status == 200
            assert b"Light Learning Agents" in body
        with urlopen(f"http://127.0.0.1:{port}/app.js", timeout=5) as response:
            assert response.status == 200
            assert b"budget" in response.read()
        with urlopen(f"http://127.0.0.1:{port}/style.css", timeout=5) as response:
            assert response.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
