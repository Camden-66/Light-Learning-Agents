"""Interactive explainer over the canonical RoomEnv (peer environment)."""

from __future__ import annotations

import json
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .config import BUDGETS, RoomConfig
from .mle import PassiveUniformOracleMLE, likelihood_profile, run_mle_episode
from .room import RoomEnv

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}
_RL: dict | None = None
_METRICS_CACHE: dict | None = None


def _counts(env: RoomEnv) -> tuple[list[int], list[int]]:
    on = [0] * env.config.slot_count
    vis = [0] * env.config.slot_count
    for obs in env.state.history:
        vis[obs.time_slot] += 1
        if obs.light_on:
            on[obs.time_slot] += 1
    return on, vis


def _obs_vector(env: RoomEnv) -> list[float]:
    on, vis = _counts(env)
    b = float(env.budget)
    remaining = env.state.remaining_observations / b
    commit = 1.0 if env.state.phase == "estimate" and env.terminal_outcome is None else 0.0
    return [c / b for c in on] + [c / b for c in vis] + [remaining, commit]


def _curve(config: RoomConfig, theta: int) -> list[float]:
    return [config.light_probability(t, theta) for t in range(config.slot_count)]


def _public(session: dict, extra: dict | None = None) -> dict:
    env: RoomEnv = session["env"]
    on, vis = _counts(env)
    done = env.terminal_outcome is not None
    payload = {
        "session": session["id"],
        "episode_id": session["episode_id"],
        "budget": env.budget,
        "remaining": env.state.remaining_observations if not done else 0,
        "commit_phase": env.state.phase == "estimate" and not done,
        "done": done,
        "on_counts": on,
        "visit_counts": vis,
        "history": [
            {"slot": o.time_slot, "light_on": int(o.light_on)} for o in env.state.history
        ],
        "obs": _obs_vector(env),
    }
    if extra:
        payload.update(extra)
    return payload


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("[web]", fmt % args)

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n).decode())

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/metrics":
            global _METRICS_CACHE
            with _LOCK:
                if _METRICS_CACHE is None:
                    _METRICS_CACHE = _pilot_mle_metrics()
                metrics = _METRICS_CACHE
            self._json(200, metrics)
            return
        if path == "/api/rl":
            with _LOCK:
                payload = _rl_public()
            self._json(200, payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if path == "/api/rl/train":
            try:
                payload = _train_rl(body)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
                return
            self._json(200, payload)
            return
        with _LOCK:
            try:
                if path == "/api/new":
                    self._json(200, self._new(body))
                    return
                if path == "/api/look":
                    self._json(200, self._look(body))
                    return
                if path == "/api/commit":
                    self._json(200, self._commit(body))
                    return
                if path == "/api/mle":
                    self._json(200, self._mle(body))
                    return
                if path == "/api/rl/play":
                    self._json(200, self._rl_play(body))
                    return
            except KeyError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(500, {"error": str(exc)})
                return
        self._json(404, {"error": "unknown endpoint"})

    def _session(self, body: dict) -> dict:
        sid = body.get("session")
        if not sid or sid not in _SESSIONS:
            raise KeyError("missing session")
        return _SESSIONS[sid]

    def _new(self, body: dict) -> dict:
        budget = int(body.get("budget", 8))
        seed = int(body["seed"]) if body.get("seed") is not None else int(
            np.random.default_rng().integers(0, 2**31 - 1)
        )
        env = RoomEnv(budget=budget)
        env.reset(seed=seed)
        sid = uuid.uuid4().hex
        session = {
            "id": sid,
            "env": env,
            "episode_id": f"play-{budget}-{seed}",
            "seed": seed,
        }
        _SESSIONS[sid] = session
        return _public(session)

    def _look(self, body: dict) -> dict:
        session = self._session(body)
        env: RoomEnv = session["env"]
        if env.terminal_outcome is not None:
            return _public(session, {"error": "episode already finished"})
        if env.state.phase != "observe":
            return _public(session, {"error": "budget exhausted; commit a peak"})
        slot = int(body["slot"])
        env.observe(slot)
        last = env.state.history[-1]
        return _public(
            session,
            {
                "just_looked": last.time_slot,
                "light_on": int(last.light_on),
                "reward": 0.0,
            },
        )

    def _commit(self, body: dict) -> dict:
        session = self._session(body)
        env: RoomEnv = session["env"]
        if env.state.phase != "estimate":
            return _public(session, {"error": "still have looks left"})
        outcome = env.estimate(int(body["slot"]))
        return _public(
            session,
            {
                "theta": outcome.theta,
                "theta_hat": outcome.theta_hat,
                "absolute_error": outcome.absolute_error,
                "reward": outcome.reward,
                "curve": _curve(env.config, outcome.theta),
                "done": True,
            },
        )

    def _mle(self, body: dict) -> dict:
        session = self._session(body)
        env: RoomEnv = session["env"]
        seed = session["seed"]
        theta = env._theta
        assert theta is not None
        clone = RoomEnv(env.config, budget=env.budget)
        clone.reset(seed=seed, theta=theta)
        agent = PassiveUniformOracleMLE(env.config, rng=np.random.default_rng(seed))
        outcome = run_mle_episode(clone, agent)
        history = [(o.time_slot, 1 if o.light_on else 0) for o in clone.state.history]
        return {
            "session": session["id"],
            "condition": "oracle_mle",
            "theta": outcome.theta,
            "theta_hat": outcome.theta_hat,
            "absolute_error": outcome.absolute_error,
            "trace": [
                {
                    "step": i,
                    "phase": "observe",
                    "action": o.time_slot,
                    "light_on": int(o.light_on),
                }
                for i, o in enumerate(clone.state.history)
            ]
            + [
                {
                    "step": env.budget,
                    "phase": "terminal",
                    "action": outcome.theta_hat,
                    "light_on": None,
                }
            ],
            "likelihood": likelihood_profile(history, env.config),
            "curve": _curve(env.config, outcome.theta),
        }

    def _rl_play(self, body: dict) -> dict:
        global _RL
        if _RL is None:
            return {"error": "train RL first"}
        session = self._session(body)
        env: RoomEnv = session["env"]
        theta = env._theta
        assert theta is not None
        from .rl_train import play_policy

        result = play_policy(
            _RL["W"],
            budget=env.budget,
            seed=session["seed"],
            theta=theta,
        )
        result["curve"] = _curve(env.config, result["theta"])
        result["session"] = session["id"]
        return result


def _rl_public() -> dict:
    if _RL is None:
        return {"trained": False}
    return {
        "trained": True,
        "budget": _RL["budget"],
        "n_episodes": _RL["n_episodes"],
        "history": _RL["history"],
        "eval_mae": _RL["eval_mae"],
        "eval_hit_within_one": _RL["eval_hit_within_one"],
        "algo": _RL["algo"],
        "note": _RL["note"],
    }


def _train_rl(body: dict) -> dict:
    global _RL
    from .rl_train import train_reinforce

    budget = int(body.get("budget", 8))
    n_episodes = int(body.get("episodes", 250))
    seed = int(body.get("seed", 0))
    result = train_reinforce(budget=budget, n_episodes=n_episodes, seed=seed)
    with _LOCK:
        _RL = result
    return _rl_public()


def _pilot_mle_metrics() -> dict:
    config = RoomConfig()
    out: dict[str, dict] = {}
    for budget in BUDGETS:
        errors: list[int] = []
        hits = 0
        for i in range(10):
            env = RoomEnv(config, budget=budget)
            env.reset(seed=20260907 + budget * 100 + i)
            agent = PassiveUniformOracleMLE(config, rng=np.random.default_rng(i + 17))
            outcome = run_mle_episode(env, agent)
            errors.append(outcome.absolute_error)
            hits += int(outcome.absolute_error <= 1)
        mae = sum(errors) / len(errors)
        out[str(budget)] = {
            "n": 10,
            "mae": mae,
            "hit_within_one": hits / len(errors),
            "condition": "oracle_mle",
            "budget": budget,
        }
    return out


def serve(host: str = "127.0.0.1", port: int = 8767) -> None:
    WEB_ROOT.mkdir(parents=True, exist_ok=True)

    class ReuseServer(ThreadingHTTPServer):
        allow_reuse_address = True

    httpd = ReuseServer((host, port), Handler)
    print(f"Light Learning demo  http://{host}:{port}")
    httpd.serve_forever()
