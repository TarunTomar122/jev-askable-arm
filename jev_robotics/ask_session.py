"""Live ask-arm session used by the browser lab."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

import numpy as np

from .common import JEVClient, jsonable, load_local_env
from .primitives import (
    KINDS,
    TASKS,
    PrimitiveChooser,
    PrimitiveExecutor,
    Task,
    build_ask_state,
    gripper_closed,
    make_arm_env,
    press_order,
    scene_objects,
    tcp_position,
)


def hide_goal_marker(env: Any) -> None:
    """Leave the arm and cube; hide the green target used by PickCube."""
    import sapien

    site = getattr(env.unwrapped, "goal_site", None)
    if site is None:
        return
    objs = getattr(site, "_objs", None) or []
    for obj in objs:
        entity = getattr(obj, "entity", obj)
        finder = getattr(entity, "find_component_by_type", None)
        if finder is None:
            continue
        body = finder(sapien.render.RenderBodyComponent)
        if body is not None:
            body.visibility = 0.0


def first_array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("rgb", "rgb_array", "Color"):
            if key in value:
                value = value[key]
                break
        else:
            value = next(iter(value.values()))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def jpeg_b64(frame: np.ndarray | None) -> str | None:
    if frame is None:
        return None
    import base64
    from io import BytesIO

    from PIL import Image

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=78)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def match_task(env_id: str, goal: str, task_name: str | None) -> Task | None:
    if task_name:
        for task in TASKS():
            if task.name == task_name:
                return task
    for task in TASKS():
        if task.goal == goal and task.env_id == env_id:
            return task
    for task in TASKS():
        if task.goal == goal:
            return task
    return None


def infer_success(env: Any, goal: str, start: dict[str, Any], task: Task | None) -> bool:
    if task is not None:
        try:
            return bool(task.success(env, start))
        except Exception:
            return False
    text = goal.lower()
    objects = scene_objects(env)
    holding = any(bool(item.get("grasped")) for item in objects.values())
    blue = bool(objects.get("blue_button", {}).get("pressed"))
    red = bool(objects.get("red_button", {}).get("pressed"))
    if "red" in text and "blue" in text and ("press" in text or "then" in text):
        if "then" in text:
            before, _after = text.split("then", 1)
            first = "red" if "red" in before else "blue"
            second = "blue" if first == "red" else "red"
            return press_order(env)[:2] == [first, second]
        return blue and red
    if "press" in text and "blue" in text:
        return blue and not red
    if "press" in text and "red" in text:
        return red
    if "place" in text or "put it" in text or "green target" in text:
        try:
            return bool(env.unwrapped.evaluate()["success"])
        except Exception:
            return False
    if "lift" in text:
        cube = objects.get("red_cube", {})
        z = float(np.asarray(cube.get("position", [0, 0, 0])).reshape(-1)[2])
        return holding and z >= 0.06
    if "grasp" in text or "pick" in text or "held" in text:
        return holding
    if "hover" in text:
        cube = objects.get("red_cube")
        if cube is None:
            return False
        from .primitives import _hovering

        return _hovering(env, np.asarray(cube["position"], dtype=np.float32))
    if "open" in text and "gripper" in text:
        return not gripper_closed(env)
    if "close" in text and "gripper" in text:
        return gripper_closed(env)
    if "upward" in text or "raise" in text:
        return float(tcp_position(env)[2]) >= float(start["tcp"][2]) + 0.04
    if "downward" in text or "lower" in text:
        return float(tcp_position(env)[2]) <= float(start["tcp"][2]) - 0.03
        return False


def goal_key(text: str) -> str:
    return " ".join(str(text).lower().split())


class AskArmSession:
    def __init__(self, camera: bool = True):
        load_local_env()
        self.camera = camera
        self.chooser = PrimitiveChooser(min_confidence=0.0, client=JEVClient())
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.jobs: queue.Queue[tuple[Callable[..., Any] | None, tuple, dict, dict | None]] = (
            queue.Queue()
        )
        self.env = None
        self.executor: PrimitiveExecutor | None = None
        self.task: Task | None = None
        self.start: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.chain: list[str] = []
        self.frame: np.ndarray | None = None
        self.running = False
        self.ready = False
        self.success = False
        self.jev_done = False
        self.env_done = False
        self.status = "Load a goal, then Run or Step."
        self.error: str | None = None
        self.env_id = "PickCube-v1"
        self.seed = 0
        self.goal = ""
        self.max_decisions = 40
        self.tokens = 0
        self.render_ok = camera
        self._physics = 0
        self.live_robot: dict[str, Any] = {
            "tcp_position_m": [0.0, 0.0, 0.0],
            "gripper": "open",
            "holding": None,
        }
        self.live_objects: dict[str, Any] = {}
        self.spoken_queue: list[str] = []

    def serve_sim(self) -> None:
        """Run forever on the main thread. SAPIEN aborts if gym.make happens elsewhere."""
        while True:
            fn, args, kwargs, box = self.jobs.get()
            if fn is None:
                if box is not None:
                    box["event"].set()
                break
            try:
                result = fn(*args, **kwargs)
                if box is not None:
                    box["ok"] = result
            except Exception as error:
                if box is None:
                    with self.lock:
                        self.error = str(error)
                        self.status = f"Error: {error}"
                        self.running = False
                else:
                    box["err"] = error
            if box is not None:
                box["event"].set()

    def _submit(self, fn: Callable[..., Any] | None, *args: Any, wait: bool = True, **kwargs: Any) -> Any:
        box = None
        if wait:
            box = {"event": threading.Event()}
        self.jobs.put((fn, args, kwargs, box))
        if box is None:
            return None
        box["event"].wait()
        if "err" in box:
            raise box["err"]
        return box.get("ok")

    def close(self) -> None:
        self.stop_event.set()
        self._close_body()
        self.jobs.put((None, (), {}, None))

    def _close_body(self) -> None:
        env = self.env
        self.env = None
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    def meta(self) -> dict[str, Any]:
        return {
            "primitive_count": len(KINDS),
            "primitives": [
                {"id": name, "description": text} for name, text in KINDS.items()
            ],
            "presets": [
                {
                    "name": task.name,
                    "env_id": task.env_id,
                    "goal": task.goal,
                    "phase": task.phase,
                    "max_decisions": task.max_decisions,
                }
                for task in TASKS()
            ],
            "envs": ["PickCube-v1", "ResetButton-v7"],
        }

    def public_state(self) -> dict[str, Any]:
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            payload = {
                "ready": self.ready,
                "running": self.running,
                "status": self.status,
                "error": self.error,
                "env_id": self.env_id,
                "seed": self.seed,
                "goal": self.goal,
                "task": None if self.task is None else self.task.name,
                "success": self.success,
                "jev_done": self.jev_done,
                "env_done": self.env_done,
                "decisions": len(self.history),
                "max_decisions": self.max_decisions,
                "chain": list(self.history),
                "robot": self.live_robot,
                "objects": self.live_objects,
                "tokens": self.tokens,
                "camera": self.render_ok,
                "primitive_count": len(KINDS),
                "queue": list(self.spoken_queue),
            }
        payload["frame"] = jpeg_b64(frame)
        return jsonable(payload)

    def _publish_scene(self) -> None:
        if self.env is None:
            return
        packed = build_ask_state(self.env, self.goal, self.chain)
        with self.lock:
            self.live_robot = packed["robot"]
            self.live_objects = packed["objects"]

    def _grab_frame(self, force: bool = False) -> None:
        if not self.render_ok or self.env is None:
            return
        if not force and self._physics % 2 != 0:
            return
        try:
            frame = first_array(self.env.render())
        except Exception as error:
            self.render_ok = False
            with self.lock:
                self.error = f"Camera off: {error}"
            return
        with self.lock:
            self.frame = frame

    def _on_physics(self) -> None:
        self._physics += 1
        self._publish_scene()
        self._grab_frame(force=False)

    def reset(
        self,
        env_id: str,
        seed: int,
        goal: str,
        task_name: str | None = None,
        max_decisions: int | None = None,
        hide_target: bool = False,
        cinematic: bool = False,
    ) -> dict[str, Any]:
        if not str(goal).strip():
            raise ValueError("goal is empty")
        self.stop_event.set()
        return self._submit(
            self._reset_body,
            env_id,
            seed,
            goal,
            task_name,
            max_decisions,
            hide_target,
            cinematic,
        )

    def _reset_body(
        self,
        env_id: str,
        seed: int,
        goal: str,
        task_name: str | None,
        max_decisions: int | None,
        hide_target: bool = False,
        cinematic: bool = False,
    ) -> dict[str, Any]:
        old = self.env
        self.env = None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        render_mode = "rgb_array" if self.camera else None
        env = make_arm_env(
            env_id,
            max_steps=800,
            render_mode=render_mode,
            cinematic=cinematic,
        )
        env.reset(seed=int(seed))
        if hide_target:
            hide_goal_marker(env)
        task = match_task(env_id, goal, task_name)
        if max_decisions is None:
            max_decisions = 40 if task is None else task.max_decisions
        start = {"tcp": tcp_position(env).copy(), "objects": scene_objects(env)}
        self.env = env
        self.executor = PrimitiveExecutor(env, on_physics=self._on_physics)
        self.task = task
        self.start = start
        self.history = []
        self.chain = []
        self.running = False
        self.ready = True
        self.success = infer_success(env, goal, start, task)
        self.jev_done = False
        self.env_done = False
        self.status = "Ready. Run or Step."
        self.error = None
        self.env_id = env_id
        self.seed = int(seed)
        self.goal = goal
        self.max_decisions = int(max_decisions)
        self.tokens = 0
        self.render_ok = self.camera
        self._physics = 0
        self.frame = None
        self.spoken_queue = []
        self.stop_event.clear()
        self._publish_scene()
        self._grab_frame(force=True)
        return self.public_state()

    def boot_demo(self, seed: int = 0) -> dict[str, Any]:
        state = self.reset(
            "PickCube-v1",
            seed,
            "Listen for the next spoken goal.",
            hide_target=True,
            cinematic=True,
            max_decisions=8,
        )
        self._kick_spoken()
        return state

    def say(self, goal: str) -> dict[str, Any]:
        text = str(goal).strip()
        if not text:
            raise ValueError("empty")
        key = goal_key(text)
        with self.lock:
            if any(goal_key(item) == key for item in self.spoken_queue):
                return {"ok": True, "duplicate": True, "goal": text}
            if self.running and key == goal_key(self.goal):
                return {"ok": True, "duplicate": True, "goal": text}
            if self.running or not self.ready:
                self.spoken_queue.append(text)
                self.status = f"Queued: {text}"
                return {"ok": True, "queued": True, "goal": text}
            self.running = True
            self.status = text
        self._submit(self._say_body, text, wait=False)
        return {"ok": True, "queued": False, "goal": text}

    def _kick_spoken(self) -> None:
        with self.lock:
            if self.running or not self.ready or not self.spoken_queue:
                return
            text = self.spoken_queue.pop(0)
            self.running = True
            self.status = text
        self._submit(self._say_body, text, wait=False)

    def _begin_goal(self, goal: str) -> None:
        self.stop_event.clear()
        self.goal = goal
        self.task = match_task(self.env_id, goal, None)
        self.start = {
            "tcp": tcp_position(self.env).copy(),
            "objects": scene_objects(self.env),
        }
        self.history = []
        self.chain = []
        self.jev_done = False
        self.env_done = False
        self.success = infer_success(self.env, goal, self.start, self.task)
        self.max_decisions = 10 if self.task is None else self.task.max_decisions
        self.running = True
        self.status = goal

    def _say_body(self, goal: str) -> None:
        self._begin_goal(goal)
        self._run_body()

    def _run_loop_once(self) -> None:
        while not self.stop_event.is_set():
            if (
                self.success
                or self.jev_done
                or self.env_done
                or len(self.history) >= self.max_decisions
            ):
                break
            self._step_body()

    def step_once(self) -> dict[str, Any]:
        return self._submit(self._step_body)

    def _step_body(self) -> dict[str, Any]:
        if not self.ready or self.env is None or self.executor is None:
            raise RuntimeError("reset a scene first")
        if self.success:
            self.status = "Goal already complete."
            return self.public_state()
        if self.jev_done or self.env_done:
            raise RuntimeError("episode already stopped")
        if len(self.history) >= self.max_decisions:
            raise RuntimeError("decision budget used up")
        env = self.env
        executor = self.executor
        objects = scene_objects(env)
        state = build_ask_state(env, self.goal, self.chain)
        selection = self.chooser.select(state, objects)
        usage = {}
        response = selection.answers.get("response", {})
        if isinstance(response, dict):
            usage = response.get("usage", {}) or {}
        kind = selection.selected.skill
        target = selection.selected.target
        ident = selection.selected.id
        self.status = f"Doing {ident}"
        self.tokens += int(usage.get("input_tokens", 0) or 0)
        executor.execute(kind, target, objects)
        self.history.append(
            {
                "id": ident,
                "kind": kind,
                "target": target,
                "latency_ms": round(float(selection.latency_ms), 1),
                "confidence": round(float(selection.confidence), 3),
            }
        )
        self.chain.append(ident)
        self.jev_done = kind == "done"
        if (
            not self.jev_done
            and len(self.history) >= 3
            and self.history[-1]["kind"] == self.history[-2]["kind"] == self.history[-3]["kind"]
        ):
            self.jev_done = True
            self.status = "Stopped a repeat loop."
            self._publish_scene()
            self._grab_frame(force=True)
            return self.public_state()
        self.env_done = bool(executor.last_done)
        self.success = infer_success(env, self.goal, self.start, self.task)
        if self.success:
            self.status = "Success."
        elif self.jev_done:
            self.status = "Jev said done."
        elif self.env_done:
            self.status = "Episode ended."
        else:
            self.status = f"Did {ident}."
        self._publish_scene()
        self._grab_frame(force=True)
        return self.public_state()

    def start_run(self) -> None:
        with self.lock:
            if not self.ready:
                raise RuntimeError("reset a scene first")
            if self.running:
                raise RuntimeError("already running")
            self.running = True
            self.status = "Running."
        self.stop_event.clear()
        self._submit(self._run_body, wait=False)

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            if self.running:
                self.status = "Stopping."

    def _run_body(self) -> None:
        try:
            while True:
                self._run_loop_once()
                with self.lock:
                    nxt = self.spoken_queue.pop(0) if self.spoken_queue else None
                if nxt is None or self.stop_event.is_set():
                    break
                self._begin_goal(nxt)
            if self.success:
                self.status = "Success."
            elif self.stop_event.is_set() and not self.success:
                self.status = "Stopped."
            elif len(self.history) >= self.max_decisions and not self.success:
                self.status = "Hit the decision limit."
            self.running = False
        except Exception as error:
            self.error = str(error)
            self.status = f"Error: {error}"
            self.running = False
