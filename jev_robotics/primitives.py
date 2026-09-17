"""Askable arm: a fixed primitive catalog + Jev picks the next one.

No task policy. The chain is the loop. Options stay mutually exclusive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .common import JEVClient, Selection, jsonable, public_state


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


# One unit of ee-delta action is 0.1 m in this controller.
CM_TO_ACTION = 0.01 / 0.1

KINDS: dict[str, str] = {
    "wait": "Hold still once and reassess. Do not pick this twice in a row.",
    "done": "Stop. Use only when the English goal is already complete.",
    "open_gripper": "Open the fingers. Use before grasping or after placing.",
    "close_gripper": "Close the fingers. Use when the fingertip is already around an object.",
    "forward_1cm": "Move the fingertip 1 cm forward (+x, toward the table workspace).",
    "back_1cm": "Move the fingertip 1 cm back (−x).",
    "left_1cm": "Move the fingertip 1 cm left (+y).",
    "right_1cm": "Move the fingertip 1 cm right (−y).",
    "up_1cm": "Move the fingertip 1 cm up (+z).",
    "down_1cm": "Move the fingertip 1 cm down (−z).",
    "forward_3cm": "Move the fingertip 3 cm forward (+x).",
    "back_3cm": "Move the fingertip 3 cm back (−x).",
    "left_3cm": "Move the fingertip 3 cm left (+y).",
    "right_3cm": "Move the fingertip 3 cm right (−y).",
    "up_3cm": "Move the fingertip 3 cm up (+z).",
    "down_3cm": "Move the fingertip 3 cm down (−z).",
    "forward_5cm": "Move the fingertip 5 cm forward (+x).",
    "back_5cm": "Move the fingertip 5 cm back (−x).",
    "left_5cm": "Move the fingertip 5 cm left (+y).",
    "right_5cm": "Move the fingertip 5 cm right (−y).",
    "up_5cm": "Move the fingertip 5 cm up (+z). Use this to lift a held object.",
    "down_5cm": "Move the fingertip 5 cm down (−z).",
    "retreat": "Raise the fingertip 8 cm. Use after a grasp or to leave a button.",
    "home_height": "Move the fingertip to a safe working height (~12 cm) without changing x/y much.",
    "approach": (
        "Move toward a named object while staying a few cm above it. "
        "Use this to get close without hitting the table. Pick the object in target."
    ),
    "hover_over": (
        "Go above a named object and stop ~6 cm over it. "
        "Use this as the first step of a grasp or press. Pick the object in target."
    ),
    "slide_to": (
        "Slide in x/y toward a named object, keeping the current height. "
        "Pick the object in target."
    ),
    "go_to": (
        "Move directly to a named object's grasp/press pose, including down onto it. "
        "Pick the object in target."
    ),
    "descend_onto": (
        "From above, lower onto a named object to wrap the fingers around it. "
        "Use this right before close_gripper when picking something up. Pick the object in target."
    ),
    "press_onto": (
        "Drive down onto a named object to press it. Repeat if it is not pressed yet. "
        "Pick the object in target."
    ),
}

TARGET_KINDS = {
    "approach",
    "hover_over",
    "slide_to",
    "go_to",
    "descend_onto",
    "press_onto",
}

MOVE_DELTA_CM: dict[str, tuple[float, float, float]] = {
    "forward_1cm": (1.0, 0.0, 0.0),
    "back_1cm": (-1.0, 0.0, 0.0),
    "left_1cm": (0.0, 1.0, 0.0),
    "right_1cm": (0.0, -1.0, 0.0),
    "up_1cm": (0.0, 0.0, 1.0),
    "down_1cm": (0.0, 0.0, -1.0),
    "forward_3cm": (3.0, 0.0, 0.0),
    "back_3cm": (-3.0, 0.0, 0.0),
    "left_3cm": (0.0, 3.0, 0.0),
    "right_3cm": (0.0, -3.0, 0.0),
    "up_3cm": (0.0, 0.0, 3.0),
    "down_3cm": (0.0, 0.0, -3.0),
    "forward_5cm": (5.0, 0.0, 0.0),
    "back_5cm": (-5.0, 0.0, 0.0),
    "left_5cm": (0.0, 5.0, 0.0),
    "right_5cm": (0.0, -5.0, 0.0),
    "up_5cm": (0.0, 0.0, 5.0),
    "down_5cm": (0.0, 0.0, -5.0),
}


def scene_objects(env: gym.Env) -> dict[str, dict[str, Any]]:
    u = env.unwrapped
    tcp = _np(u.agent.tcp_pose.p)
    objects: dict[str, dict[str, Any]] = {}

    def add(name: str, xyz: np.ndarray, extra: dict[str, Any] | None = None) -> None:
        pos = np.asarray(xyz, dtype=np.float32).reshape(-1)[:3]
        entry = {
            "position": pos.tolist(),
            "distance_cm": round(float(np.linalg.norm(pos - tcp)) * 100.0, 1),
        }
        if extra:
            entry.update(extra)
        objects[name] = entry

    if hasattr(u, "cube") and not hasattr(u, "cubeA"):
        cube = _np(u.cube.pose.p)
        holding = _bool(u.agent.is_grasping(u.cube))
        add("red_cube", cube, {"grasped": holding})
        add("green_target", _np(u.goal_site.pose.p), {})
    if hasattr(u, "cubeA"):
        add("red_cube", _np(u.cubeA.pose.p), {"grasped": _bool(u.agent.is_grasping(u.cubeA))})
        add("green_cube", _np(u.cubeB.pose.p), {})
    if hasattr(u, "button") and hasattr(u, "_button_top"):
        extra = {}
        if hasattr(u, "_blue_pressed"):
            extra["pressed"] = bool(u._blue_pressed[0])
        add("blue_button", _np(u._button_top()), extra)
    if hasattr(u, "button_red") and hasattr(u, "_button_top_red"):
        extra = {}
        if hasattr(u, "_red_pressed"):
            extra["pressed"] = bool(u._red_pressed[0])
        add("red_button", _np(u._button_top_red()), extra)
    return objects


def gripper_closed(env: gym.Env) -> bool:
    u = env.unwrapped
    qpos = _np(u.agent.robot.get_qpos())
    # Panda finger joints are the last two; closed is near 0.
    return float(np.mean(np.abs(qpos[-2:]))) < 0.02


def tcp_position(env: gym.Env) -> np.ndarray:
    return _np(env.unwrapped.agent.tcp_pose.p)


def build_ask_state(env: gym.Env, goal: str, history: list[str]) -> dict[str, Any]:
    tcp = tcp_position(env)
    objects = scene_objects(env)
    holding = None
    for name, payload in objects.items():
        if payload.get("grasped"):
            holding = name
            break
    return {
        "goal": goal,
        "how_to_act": (
            "Choose exactly one next primitive. Chain primitives across steps. "
            "DONE only when every part of the English goal is already true. "
            "If the goal names two buttons, press the first, leave it, then press the second. "
            "Do not stop after the first press. "
            "To pick something up: open_gripper, hover_over it, descend_onto it, "
            "close_gripper, then up_5cm. "
            "To press a button: hover_over it, then press_onto it until pressed. "
            "Do not wait twice in a row."
        ),
        "robot": {
            "tcp_position_m": [round(float(x), 3) for x in tcp.tolist()],
            "gripper": "closed" if gripper_closed(env) else "open",
            "holding": holding,
        },
        "objects": objects,
        "recent_primitives": history[-12:],
        "primitive_count": len(KINDS),
    }


def target_criteria(objects: dict[str, dict[str, Any]]) -> dict[str, str]:
    criteria = {
        "none": "No object. Use this for centimetre moves, gripper, wait, and done.",
    }
    for name, payload in objects.items():
        extra = []
        if "grasped" in payload:
            extra.append("grasped" if payload["grasped"] else "free")
        if payload.get("pressed"):
            extra.append("already pressed")
        suffix = f" ({', '.join(extra)})" if extra else ""
        criteria[name] = (
            f"{name} is {payload['distance_cm']} cm from the fingertip{suffix}."
        )
    return criteria


class PrimitiveChooser:
    def __init__(self, min_confidence: float = 0.0, client: JEVClient | None = None):
        self.client = client or JEVClient()
        self.min_confidence = min_confidence

    def select(self, state: dict[str, Any], objects: dict[str, dict[str, Any]]) -> Selection:
        from .common import Candidate

        questions = {
            "kind": {
                "type": "choice",
                "instructions": (
                    "Which one primitive makes the most progress toward the goal "
                    "right now? Do not repeat a move that just failed to change anything."
                ),
                "criteria": dict(KINDS),
            },
            "target": {
                "type": "choice",
                "instructions": (
                    "If kind needs an object (hover_over, descend_onto, press_onto, "
                    "go_to, approach, slide_to), pick that object. Otherwise choose none."
                ),
                "criteria": target_criteria(objects),
            },
        }
        response, latency_ms = self.client.evaluate(public_state(state), questions)
        answers = response.get("answers", {})
        kind_ans = answers.get("kind", {})
        target_ans = answers.get("target", {})
        kind = kind_ans.get("choice") or "wait"
        target = target_ans.get("choice") or "none"
        if kind not in TARGET_KINDS:
            target = "none"
        if kind in TARGET_KINDS and target == "none":
            target = next(iter(objects), "none")
        ident = kind if target == "none" else f"{kind}:{target}"
        confidence = float(kind_ans.get("confidence") or 0.0)
        overridden = confidence < self.min_confidence
        if overridden:
            kind, target, ident = "wait", "none", "wait"
        selected = Candidate(
            id=ident,
            skill=kind,
            target=target,
            description=KINDS.get(kind, kind),
        )
        return Selection(
            selected=selected,
            confidence=confidence,
            candidate_confidence=float(target_ans.get("confidence") or 0.0),
            latency_ms=latency_ms,
            answers={"response": jsonable(response)},
            raw_choice=ident,
            overridden_for_confidence=overridden,
        )


class PrimitiveExecutor:
    def __init__(self, env: gym.Env, on_physics: Callable[[], None] | None = None):
        self.env = env
        self.steps = 0
        self.last_done = False
        self.last_info: dict[str, Any] = {}
        self._grip = 1.0
        self.on_physics = on_physics

    def _action_dim(self) -> int:
        shape = self.env.action_space.shape
        return int(shape[-1])

    def _pack(self, dx: float, dy: float, dz: float, grip: float) -> np.ndarray:
        dim = self._action_dim()
        action = np.zeros(dim, dtype=np.float32)
        action[0], action[1], action[2] = dx, dy, dz
        action[-1] = grip
        return action

    def _step(self, action: np.ndarray) -> None:
        _obs, _reward, _terminated, truncated, info = self.env.step(action)
        self.last_info = info
        self.last_done = _bool(truncated)
        self.steps += 1
        if self.on_physics is not None:
            self.on_physics()

    def _hold(self, grip: float, count: int) -> None:
        action = self._pack(0.0, 0.0, 0.0, grip)
        for _ in range(count):
            self._step(action)
            if self.last_done:
                return

    def _move_toward(self, goal: np.ndarray, loops: int = 18, stop: float = 0.006) -> None:
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)[:3]
        for _ in range(loops):
            tcp = tcp_position(self.env)
            error = goal - tcp
            if float(np.linalg.norm(error)) <= stop:
                return
            step = np.clip(error / 0.1, -1.0, 1.0)
            self._step(self._pack(float(step[0]), float(step[1]), float(step[2]), self._grip))
            if self.last_done:
                return

    def _object_goal(self, kind: str, target: str, payload: dict[str, Any]) -> np.ndarray:
        pos = np.asarray(payload["position"], dtype=np.float32).reshape(-1)[:3]
        tcp = tcp_position(self.env)
        table_object = target in {"red_cube", "green_cube", "green_target"}
        if kind == "hover_over":
            return pos + np.array([0.0, 0.0, 0.06], dtype=np.float32)
        if kind == "approach":
            lift = 0.04 if table_object else 0.03
            return pos + np.array([0.0, 0.0, lift], dtype=np.float32)
        if kind == "slide_to":
            return np.array([pos[0], pos[1], tcp[2]], dtype=np.float32)
        if kind == "descend_onto":
            if table_object:
                return pos + np.array([0.0, 0.0, 0.01], dtype=np.float32)
            return pos.copy()
        if kind == "press_onto":
            return pos + np.array([0.0, 0.0, -0.008], dtype=np.float32)
        if kind == "go_to":
            if table_object:
                return pos + np.array([0.0, 0.0, 0.01], dtype=np.float32)
            return pos.copy()
        return pos.copy()

    def execute(self, kind: str, target: str, objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        before = self.steps
        if kind == "open_gripper":
            self._grip = 1.0
            self._hold(1.0, 8)
        elif kind == "close_gripper":
            self._grip = -1.0
            self._hold(-1.0, 10)
        elif kind == "wait":
            self._hold(self._grip, 2)
        elif kind == "done":
            self._hold(self._grip, 2)
        elif kind == "retreat":
            tcp = tcp_position(self.env)
            self._move_toward(tcp + np.array([0.0, 0.0, 0.08], dtype=np.float32), loops=10)
        elif kind == "home_height":
            tcp = tcp_position(self.env)
            self._move_toward(np.array([tcp[0], tcp[1], 0.12], dtype=np.float32), loops=12)
        elif kind in MOVE_DELTA_CM:
            dx, dy, dz = MOVE_DELTA_CM[kind]
            tcp = tcp_position(self.env)
            goal = tcp + np.array([dx, dy, dz], dtype=np.float32) / 100.0
            self._move_toward(goal, loops=8, stop=0.003)
        elif kind in TARGET_KINDS:
            payload = objects.get(target)
            if payload is None:
                self._hold(self._grip, 2)
            else:
                self._move_toward(self._object_goal(kind, target, payload))
        else:
            self._hold(self._grip, 1)
        return {
            "kind": kind,
            "target": target,
            "steps": self.steps - before,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "env_done": self.last_done,
        }


class ContinueEpisode(gym.Wrapper):
    """Keep the scene running after a button press / task success.

    ManiSkill sets terminated=True whenever evaluate()['success'] is true.
    ResetButton-v7 uses that for *any* red press, which kills red-then-blue.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.press_order: list[str] = []
        self._seen: set[str] = set()
        self._patch_success_termination()

    def _patch_success_termination(self) -> None:
        base = self.env.unwrapped
        if getattr(base, "_ask_success_patched", False):
            return
        orig = base.evaluate

        def evaluate():
            info = orig()
            info = dict(info)
            if "success" in info:
                flag = info["success"]
                info["goal_success"] = flag
                info["success"] = flag ^ flag
            return info

        base.evaluate = evaluate
        base._ask_success_patched = True

    def reset(self, **kwargs):
        self.press_order = []
        self._seen = set()
        return self.env.reset(**kwargs)

    def _track_buttons(self) -> None:
        u = self.env.unwrapped
        if hasattr(u, "_red_pressed") and _bool(u._red_pressed) and "red" not in self._seen:
            self._seen.add("red")
            self.press_order.append("red")
        if hasattr(u, "_blue_pressed") and _bool(u._blue_pressed) and "blue" not in self._seen:
            self._seen.add("blue")
            self.press_order.append("blue")

    def step(self, action):
        obs, reward, _terminated, truncated, info = self.env.step(action)
        self._track_buttons()
        return obs, reward, False, truncated, info


def press_order(env: gym.Env) -> list[str]:
    cur: Any = env
    while cur is not None:
        order = getattr(cur, "press_order", None)
        if isinstance(order, list):
            return list(order)
        cur = getattr(cur, "env", None)
    return []


def make_arm_env(
    env_id: str,
    max_steps: int = 250,
    render_mode: str | None = None,
) -> gym.Env:
    import mani_skill.envs  # noqa: F401
    import reset_env  # noqa: F401

    kwargs: dict[str, Any] = {
        "obs_mode": "state",
        "num_envs": 1,
        "sim_backend": "cpu",
        "render_mode": render_mode,
        "max_episode_steps": max_steps,
        "control_mode": "pd_ee_delta_pos",
    }
    return ContinueEpisode(gym.make(env_id, **kwargs))


@dataclass
class Task:
    name: str
    env_id: str
    goal: str
    success: Callable[[gym.Env, dict[str, Any]], bool]
    max_decisions: int = 40
    phase: int = 1


def _pickcube_holding(env: gym.Env) -> bool:
    return _bool(env.unwrapped.agent.is_grasping(env.unwrapped.cube))


def TASKS() -> list[Task]:
    return [
        Task(
            "raise_hand",
            "PickCube-v1",
            "Move the fingertip upward by at least 4 centimeters. Do not grab anything.",
            lambda env, start: tcp_position(env)[2] >= start["tcp"][2] + 0.04,
            max_decisions=12,
            phase=1,
        ),
        Task(
            "lower_hand",
            "PickCube-v1",
            "Move the fingertip downward by at least 3 centimeters. Do not grab anything.",
            lambda env, start: tcp_position(env)[2] <= start["tcp"][2] - 0.03,
            max_decisions=12,
            phase=1,
        ),
        Task(
            "open_fingers",
            "PickCube-v1",
            "Open the gripper fully. Do not move the arm much.",
            lambda env, start: not gripper_closed(env),
            max_decisions=8,
            phase=1,
        ),
        Task(
            "close_fingers",
            "PickCube-v1",
            "Close the gripper fully. Do not worry about the cube.",
            lambda env, start: gripper_closed(env),
            max_decisions=8,
            phase=1,
        ),
        Task(
            "reach_cube",
            "PickCube-v1",
            "Move the fingertip to within 5 cm of the red cube. Do not grasp yet.",
            lambda env, start: float(
                np.linalg.norm(tcp_position(env) - _np(env.unwrapped.cube.pose.p))
            )
            <= 0.05,
            max_decisions=25,
            phase=2,
        ),
        Task(
            "hover_cube",
            "PickCube-v1",
            "Hover the fingertip above the red cube: within 4 cm in x/y and 3 to 10 cm above it. Do not grasp.",
            lambda env, start: _hovering(env, _np(env.unwrapped.cube.pose.p)),
            max_decisions=25,
            phase=2,
        ),
        Task(
            "grasp_cube",
            "PickCube-v1",
            "Pick up the red cube in the gripper so it is held.",
            lambda env, start: _pickcube_holding(env),
            max_decisions=40,
            phase=3,
        ),
        Task(
            "lift_cube",
            "PickCube-v1",
            "Pick up the red cube and lift it at least 6 cm above the table.",
            lambda env, start: _pickcube_holding(env)
            and _float(env.unwrapped.cube.pose.p[0, 2]) >= 0.06,
            max_decisions=50,
            phase=3,
        ),
        Task(
            "place_cube",
            "PickCube-v1",
            "Pick up the red cube and put it on the green target.",
            lambda env, start: _bool(env.unwrapped.evaluate()["success"]),
            max_decisions=70,
            phase=3,
        ),
        Task(
            "reach_blue",
            "ResetButton-v7",
            "Move the fingertip to within 5 cm of the blue button. Do not press the red button.",
            lambda env, start: float(
                np.linalg.norm(tcp_position(env) - _np(env.unwrapped._button_top()))
            )
            <= 0.05
            and not bool(env.unwrapped._red_pressed[0]),
            max_decisions=30,
            phase=4,
        ),
        Task(
            "press_blue",
            "ResetButton-v7",
            "Press the blue button. Do not press the red button.",
            lambda env, start: bool(env.unwrapped._blue_pressed[0])
            and not bool(env.unwrapped._red_pressed[0]),
            max_decisions=50,
            phase=4,
        ),
        Task(
            "blue_then_red",
            "ResetButton-v7",
            "Press the blue button first, then press the red button.",
            lambda env, start: press_order(env)[:2] == ["blue", "red"],
            max_decisions=70,
            phase=4,
        ),
        Task(
            "red_then_blue",
            "ResetButton-v7",
            "Press the red button first, then press the blue button.",
            lambda env, start: press_order(env)[:2] == ["red", "blue"],
            max_decisions=70,
            phase=4,
        ),
    ]


def _hovering(env: gym.Env, xyz: np.ndarray) -> bool:
    tcp = tcp_position(env)
    xy = float(np.linalg.norm(tcp[:2] - xyz[:2]))
    dz = float(tcp[2] - xyz[2])
    return xy <= 0.04 and 0.03 <= dz <= 0.10


def run_task(
    task: Task,
    seed: int,
    chooser: PrimitiveChooser,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    env = make_arm_env(task.env_id)
    env.reset(seed=seed)
    start = {
        "tcp": tcp_position(env).copy(),
        "objects": scene_objects(env),
    }
    executor = PrimitiveExecutor(env)
    history: list[str] = []
    latencies: list[float] = []
    tokens = 0
    success = False
    started = time.perf_counter()
    limit = max_decisions or task.max_decisions
    final_tcp: list[float] = []
    final_objects: dict[str, Any] = {}
    try:
        for index in range(limit):
            if task.success(env, start):
                success = True
                break
            objects = scene_objects(env)
            state = build_ask_state(env, task.goal, history)
            selection = chooser.select(state, objects)
            latencies.append(selection.latency_ms)
            usage = {}
            response = selection.answers.get("response", {})
            if isinstance(response, dict):
                usage = response.get("usage", {}) or {}
            tokens += int(usage.get("input_tokens", 0) or 0)
            kind = selection.selected.skill
            target = selection.selected.target
            executor.execute(kind, target, objects)
            history.append(selection.selected.id)
            if kind == "done":
                success = task.success(env, start)
                break
            if executor.last_done:
                success = task.success(env, start)
                break
        else:
            success = task.success(env, start)
        final_tcp = tcp_position(env).tolist()
        final_objects = jsonable(scene_objects(env))
    finally:
        env.close()
    return {
        "task": task.name,
        "phase": task.phase,
        "env_id": task.env_id,
        "goal": task.goal,
        "seed": seed,
        "success": success,
        "decisions": len(history),
        "chain": history,
        "env_steps": executor.steps,
        "input_tokens": tokens,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "wall_clock_ms": (time.perf_counter() - started) * 1000.0,
        "final_tcp": final_tcp,
        "final_objects": final_objects,
    }
