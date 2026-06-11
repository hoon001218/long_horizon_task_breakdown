#!/usr/bin/python3
"""Deterministic dual-Franka handover CLI for the color-sorting scene.

This version intentionally moves the long-horizon collaboration logic out of the
LLM prompt and into the executor-side planner.  The LLM/image prompt can still be
useful for debugging or high-level explanation, but the robot should not depend
on a vision-language model to remember the mandatory handover state machine.

Core policy
-----------
For every unsorted object:

1. If one robot can reach both the object and the destination goal, do a direct
   pick-and-place with explicit lift/approach waypoints.
2. If the object and the destination are on opposite sides of the table, use the
   object-side robot to move the object to a shared table-center buffer, release
   it, then use the goal-side robot to pick it from the buffer and place it in the
   matching goal.
3. Every horizontal transfer while holding an object is separated by a lift pose
   to reduce table/object/inter-robot collisions.
4. Handover buffer z is computed from the object marker z, not from the
   table-center marker z.
5. Critical releases are verified from a fresh marker/image state before the
   next robot continues the handover.

ROS service compatibility
-------------------------
The service action strings are kept compatible with the original scene code:
`Moving`, `Grip`, `Realease`, and `Homing`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from PIL import Image

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional terminal UI
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    RICH_AVAILABLE = False

ROS_IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from custom_msgs.srv import ControlCommand
    from geometry_msgs.msg import Point, Pose, PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from visualization_msgs.msg import MarkerArray
except Exception as exc:  # pragma: no cover - depends on sourced ROS2 workspace
    ROS_IMPORT_ERROR = exc
    rclpy = None  # type: ignore[assignment]
    ControlCommand = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    Pose = None  # type: ignore[assignment]
    PoseStamped = None  # type: ignore[assignment]
    RosImage = None  # type: ignore[assignment]
    MarkerArray = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# ROS topics and service action strings
# ---------------------------------------------------------------------------

MARKER_TOPIC = "/world/object_markers"
IMAGE_TOPIC = "/world/top_camera/image_raw"
CAMERA_POSE_TOPIC = "/world/top_camera/pose"
ROBOT_POSE_TOPICS = {
    "left": "/franka_left/pose",
    "right": "/franka_right/pose",
}
EEF_POSE_TOPICS = {
    "left": "/franka_left/end_effector_pose",
    "right": "/franka_right/end_effector_pose",
}
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}

ACTION_MOVING = "Moving"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Realease"  # Keep the typo because the existing service uses it.
ACTION_HOMING = "Homing"
ALLOWED_ACTIONS = {ACTION_MOVING, ACTION_GRIP, ACTION_RELEASE, ACTION_HOMING}

DEFAULT_COMMAND = "빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라."
VERTICAL_EEF_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
DEFAULT_TABLE_CENTER = {"x": 0.6, "y": 0.0, "z": 0.46}
DEFAULT_TABLE_SIZE = {"x": 0.9, "y": 0.7, "z": 0.08}

SERVICE_WAIT_TIMEOUT_SEC = 1.0
EEF_POSITION_TOLERANCE_M = 0.035
DEFAULT_LIFT_DELTA_M = 0.12
DEFAULT_SAFE_Z_OFFSET_M = 0.16
DEFAULT_CENTER_REACH_MARGIN_M = 0.06
DEFAULT_GOAL_MARGIN_M = 0.014
DEFAULT_DROP_CLEARANCE_M = 0.052
DEFAULT_BUFFER_CLEARANCE_M = 0.035
DEFAULT_GOAL_Z_OFFSET_M = 0.03
DEFAULT_BUFFER_Z_OFFSET_M = 0.0
DEFAULT_VERIFY_DELAY_SEC = 0.8
DEFAULT_VERIFY_XY_TOLERANCE_M = 0.06
DEFAULT_VERIFY_Z_TOLERANCE_M = 0.07

ROBOT_SCENE_LABELS = {
    "left": "bottom_robot_in_top_view",
    "right": "top_robot_in_top_view",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    robot_id: str | None
    action: str
    target_pose: dict[str, Any] | None = None
    target_object_id: str | None = None
    intent: str = ""
    reason: str = ""
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    message: str
    decision: Decision


@dataclass
class RouteCandidate:
    object_id: str
    route_type: str  # direct, direct_after_grip, or handover
    source_robot_id: str
    destination_robot_id: str
    score: float
    drop_pose: dict[str, Any]
    buffer_pose: dict[str, Any] | None = None
    reason: str = ""


@dataclass
class ObjectTask:
    """High-level task selected by the task director.

    The task director decides *which object to process next*.  The lower-level
    decomposer then expands this object-level task into primitive Moving/Grip/
    Realease/Homing commands.
    """

    object_id: str
    route: RouteCandidate
    rank: int
    rationale: str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def pose_to_dict(pose: Any) -> dict[str, dict[str, float]]:
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def vector3_to_dict(vector: Any) -> dict[str, float]:
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def color_to_dict(color: Any) -> dict[str, float]:
    return {
        "r": float(color.r),
        "g": float(color.g),
        "b": float(color.b),
        "a": float(color.a),
    }


def classify_color(color: dict[str, float]) -> str:
    return "red" if color["r"] >= color["b"] else "blue"


def marker_type_name(marker_type: int) -> str:
    names = {
        0: "arrow",
        1: "cube",
        2: "sphere",
        3: "cylinder",
        4: "line_strip",
        5: "line_list",
        6: "cube_list",
        7: "sphere_list",
        8: "points",
        9: "text",
        10: "mesh",
        11: "triangle_list",
    }
    return names.get(int(marker_type), f"type_{marker_type}")


def shape_from_marker(namespace: str, marker_type: int) -> str:
    prefix = namespace.split("_", 1)[0].lower()
    if prefix in {"cube", "sphere", "capsule"}:
        return prefix
    return marker_type_name(marker_type)


def xy_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def point_in_goal(
    point: dict[str, float], goal: dict[str, Any], margin: float = 0.0
) -> bool:
    center = goal["pose"]["position"]
    scale = goal["scale"]
    half_x = max(0.0, float(scale["x"]) * 0.5 - margin)
    half_y = max(0.0, float(scale["y"]) * 0.5 - margin)
    return (
        abs(float(point["x"]) - float(center["x"])) <= half_x
        and abs(float(point["y"]) - float(center["y"])) <= half_y
    )


def make_pose(position: dict[str, float]) -> dict[str, Any]:
    return {
        "position": {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        },
        "orientation": dict(VERTICAL_EEF_ORIENTATION),
    }


def raised_pose(
    pose_or_position: dict[str, Any], lift_delta: float, safe_z: float
) -> dict[str, Any]:
    if "position" in pose_or_position:
        p = pose_or_position["position"]
    else:
        p = pose_or_position
    return make_pose(
        {
            "x": float(p["x"]),
            "y": float(p["y"]),
            "z": max(float(p["z"]) + lift_delta, safe_z),
        }
    )


def ros_pose_from_dict(pose_dict: dict[str, Any] | None) -> Any:
    pose = Pose()
    if pose_dict is None:
        pose_dict = {"position": DEFAULT_TABLE_CENTER}
    position = pose_dict.get("position", {})
    pose.position = Point(
        x=float(position.get("x", 0.0)),
        y=float(position.get("y", 0.0)),
        z=float(position.get("z", 0.0)),
    )
    pose.orientation.x = VERTICAL_EEF_ORIENTATION["x"]
    pose.orientation.y = VERTICAL_EEF_ORIENTATION["y"]
    pose.orientation.z = VERTICAL_EEF_ORIENTATION["z"]
    pose.orientation.w = VERTICAL_EEF_ORIENTATION["w"]
    return pose


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def image_msg_to_png_bytes(message: Any) -> bytes:
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "rgba8", "bgr8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")

    raw = bytes(message.data)
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channels = 4 if "a" in encoding else 3
    expected_min = height * step
    if len(raw) < expected_min:
        raise ValueError(
            f"Image data is shorter than expected: got={len(raw)}, expected_at_least={expected_min}"
        )

    mode = "RGBA" if channels == 4 else "RGB"
    image = Image.frombytes(mode, (width, height), raw, "raw", mode, step, 1)
    if encoding.startswith("bgr"):
        r, g, b = image.convert("RGB").split()
        image = Image.merge("RGB", (b, g, r))
    else:
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# ROS world node
# ---------------------------------------------------------------------------


class RosWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("handover_franka_cli")
        self._lock = Lock()
        self._table: dict[str, Any] | None = None
        self._goals: dict[str, dict[str, Any]] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._latest_image: Any | None = None
        self._latest_image_time = 0.0
        self._camera_pose: dict[str, Any] | None = None
        self._robot_poses: dict[str, dict[str, Any]] = {}
        self._eef_poses: dict[str, dict[str, Any]] = {}

        self.create_subscription(MarkerArray, MARKER_TOPIC, self._on_markers, 10)
        self.create_subscription(RosImage, IMAGE_TOPIC, self._on_image, 2)
        self.create_subscription(
            PoseStamped, CAMERA_POSE_TOPIC, self._on_camera_pose, 10
        )

        for robot_id, topic in ROBOT_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda message, rid=robot_id: self._on_robot_pose(rid, message),
                10,
            )
        for robot_id, topic in EEF_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda message, rid=robot_id: self._on_eef_pose(rid, message),
                10,
            )

        self.control_clients = {
            robot_id: self.create_client(ControlCommand, topic)
            for robot_id, topic in CONTROL_SERVICE_TOPICS.items()
        }

    def _on_markers(self, message: Any) -> None:
        table: dict[str, Any] | None = None
        goals: dict[str, dict[str, Any]] = {}
        objects: dict[str, dict[str, Any]] = {}

        for marker in message.markers:
            color = color_to_dict(marker.color)
            entry = {
                "id": f"{marker.ns}:{marker.id}",
                "namespace": str(marker.ns),
                "marker_id": int(marker.id),
                "marker_type": marker_type_name(marker.type),
                "shape": shape_from_marker(str(marker.ns), marker.type),
                "color": classify_color(color),
                "rgba": color,
                "pose": pose_to_dict(marker.pose),
                "scale": vector3_to_dict(marker.scale),
            }
            if marker.ns == "table":
                table = entry
            elif marker.ns == "red_goal":
                entry["color"] = "red"
                goals["red"] = entry
            elif marker.ns == "blue_goal":
                entry["color"] = "blue"
                goals["blue"] = entry
            else:
                objects[entry["id"]] = entry

        for obj in objects.values():
            obj["inside_goal"] = None
            obj["sorted"] = False
            for goal_color, goal in goals.items():
                if point_in_goal(obj["pose"]["position"], goal, margin=0.0):
                    obj["inside_goal"] = goal_color
                    obj["sorted"] = obj["color"] == goal_color
                    break

        with self._lock:
            if table is not None:
                self._table = table
            if goals:
                self._goals = goals
            self._objects = objects

    def _on_image(self, message: Any) -> None:
        with self._lock:
            self._latest_image = message
            self._latest_image_time = time.monotonic()

    def _on_camera_pose(self, message: Any) -> None:
        with self._lock:
            self._camera_pose = pose_to_dict(message.pose)

    def _on_robot_pose(self, robot_id: str, message: Any) -> None:
        with self._lock:
            self._robot_poses[robot_id] = pose_to_dict(message.pose)

    def _on_eef_pose(self, robot_id: str, message: Any) -> None:
        with self._lock:
            self._eef_poses[robot_id] = pose_to_dict(message.pose)

    def readiness(self) -> dict[str, bool]:
        with self._lock:
            return {
                "markers": self._table is not None
                and bool(self._goals)
                and bool(self._objects),
                "image": self._latest_image is not None,
                "camera_pose": self._camera_pose is not None,
                "robot_poses": all(
                    robot_id in self._robot_poses for robot_id in ROBOT_POSE_TOPICS
                ),
                "eef_poses": all(
                    robot_id in self._eef_poses for robot_id in EEF_POSE_TOPICS
                ),
            }

    def snapshot(self) -> tuple[dict[str, Any], bytes]:
        with self._lock:
            table = (
                dict(self._table) if self._table is not None else self._default_table()
            )
            goals = {key: dict(value) for key, value in self._goals.items()}
            objects = {key: dict(value) for key, value in self._objects.items()}
            latest_image = self._latest_image
            image_age = (
                time.monotonic() - self._latest_image_time
                if self._latest_image_time
                else None
            )
            camera_pose = self._camera_pose
            robot_poses = dict(self._robot_poses)
            eef_poses = dict(self._eef_poses)

        if latest_image is None:
            raise RuntimeError("No top camera image has been received yet.")
        png_bytes = image_msg_to_png_bytes(latest_image)

        object_list = sorted(objects.values(), key=lambda obj: obj["id"])
        table_pose = table.get("pose", {"position": DEFAULT_TABLE_CENTER})
        table_size = table.get("scale", DEFAULT_TABLE_SIZE)
        table_center = table_pose.get("position", DEFAULT_TABLE_CENTER)

        observation = {
            "timestamp_unix": time.time(),
            "topics": {
                "markers": MARKER_TOPIC,
                "image": IMAGE_TOPIC,
                "camera_pose": CAMERA_POSE_TOPIC,
                "robot_poses": ROBOT_POSE_TOPICS,
                "end_effector_poses": EEF_POSE_TOPICS,
            },
            "camera": {
                "pose": camera_pose,
                "image_topic": IMAGE_TOPIC,
                "image_age_sec": image_age,
                "image_encoding": str(latest_image.encoding),
                "image_size": {
                    "width": int(latest_image.width),
                    "height": int(latest_image.height),
                },
            },
            "table": {
                "id": table.get("id", "table:0"),
                "pose": table_pose,
                "center": table_center,
                "size": table_size,
            },
            "goals": goals,
            "robots": {
                robot_id: {
                    "scene_label": ROBOT_SCENE_LABELS[robot_id],
                    "base_pose": robot_poses.get(robot_id),
                    "end_effector_pose": eef_poses.get(robot_id),
                    "control_service": CONTROL_SERVICE_TOPICS[robot_id],
                }
                for robot_id in ("left", "right")
            },
            "objects": object_list,
            "summary": self._summary(object_list),
        }
        return observation, png_bytes

    @staticmethod
    def _default_table() -> dict[str, Any]:
        return {
            "id": "table:0",
            "pose": {
                "position": DEFAULT_TABLE_CENTER,
                "orientation": VERTICAL_EEF_ORIENTATION,
            },
            "scale": DEFAULT_TABLE_SIZE,
        }

    @staticmethod
    def _summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
        unsorted = [obj for obj in objects if not obj.get("sorted")]
        return {
            "object_count": len(objects),
            "unsorted_count": len(unsorted),
            "unsorted_ids": [obj["id"] for obj in unsorted],
            "red_unsorted": [obj["id"] for obj in unsorted if obj["color"] == "red"],
            "blue_unsorted": [obj["id"] for obj in unsorted if obj["color"] == "blue"],
        }

    def call_control_service(
        self, robot_id: str, action: str, target_pose: dict[str, Any] | None
    ) -> tuple[bool, str]:
        client = self.control_clients[robot_id]
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            return False, f"Service unavailable: {CONTROL_SERVICE_TOPICS[robot_id]}"

        request = ControlCommand.Request()
        request.action = action
        request.target_pose = ros_pose_from_dict(target_pose)
        future = client.call_async(request)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

        try:
            response = future.result()
        except Exception as exc:
            return False, f"{action} failed: {exc}"
        return bool(response.success), str(response.message)

    def wait_for_eef_target(
        self, robot_id: str, target_pose: dict[str, Any], timeout_sec: float
    ) -> bool:
        target_position = target_pose["position"]
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self._lock:
                eef_pose = self._eef_poses.get(robot_id)
            if eef_pose is None:
                continue
            current = eef_pose["position"]
            if xy_distance(current, target_position) <= EEF_POSITION_TOLERANCE_M:
                if (
                    abs(float(current["z"]) - float(target_position["z"]))
                    <= EEF_POSITION_TOLERANCE_M
                ):
                    return True
        return False


# ---------------------------------------------------------------------------
# Deterministic collaborative planner
# ---------------------------------------------------------------------------


class HandoverTaskPlanner:
    """Hierarchical task director + primitive decomposer.

    The planner is intentionally split into two levels.

    1. High-level task director
       - Looks at the current ground-truth state.
       - Computes feasible routes for all unsorted objects.
       - Chooses an object-processing order.
       - Keeps that order unless the scene changes, a task completes, or a
         primitive fails.

    2. Low-level decomposer
       - Takes the current object-level task.
       - Breaks it into safe primitive commands.
       - Inserts lift, buffer, handover, drop, and homing phases explicitly.

    This makes the runtime logic auditable: the log can show "which object is
    next", "why it was selected", "how the high-level task was decomposed", and
    "which primitive is being executed now".
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.queue: list[Decision] = []
        self.active_object_id: str | None = None
        self.active_route_type: str | None = None
        self.active_task_rationale: str = ""
        self.task_order: list[str] = []
        self.task_plan: list[ObjectTask] = []
        self.plan_revision = 0

    def clear_active_plan(self) -> None:
        self.queue.clear()
        self.active_object_id = None
        self.active_route_type = None
        self.active_task_rationale = ""
        self.task_order.clear()
        self.task_plan.clear()
        self.plan_revision += 1

    def next_decision(
        self, observation: dict[str, Any], held_objects: dict[str, str | None]
    ) -> Decision:
        if self._task_complete(observation):
            return Decision(
                None,
                ACTION_HOMING,
                done=True,
                intent="finish",
                reason="All objects are already inside their matching goal regions.",
                metadata={"high_level_state": "complete"},
            )

        # If a decomposed primitive queue exists, continue it.  This is the
        # lower-level plan for the currently selected high-level object task.
        if self.queue:
            decision = self.queue.pop(0)
            decision.metadata.setdefault("high_level_order", list(self.task_order))
            decision.metadata.setdefault("active_object", self.active_object_id)
            decision.metadata.setdefault("active_route", self.active_route_type)
            decision.metadata.setdefault("task_rationale", self.active_task_rationale)
            decision.metadata.setdefault("remaining_primitives", len(self.queue))
            return decision

        # Otherwise run the high-level director first, then decompose only the
        # top-ranked object-level task.
        task = self._select_high_level_task(observation, held_objects)
        self.active_object_id = task.object_id
        self.active_route_type = task.route.route_type
        self.active_task_rationale = task.rationale
        self.queue = self._build_steps(task.route, observation, task)
        if not self.queue:
            raise RuntimeError(
                f"Generated empty primitive plan for object {task.object_id}"
            )
        decision = self.queue.pop(0)
        decision.metadata.setdefault("high_level_order", list(self.task_order))
        decision.metadata.setdefault("active_object", self.active_object_id)
        decision.metadata.setdefault("active_route", self.active_route_type)
        decision.metadata.setdefault("task_rationale", self.active_task_rationale)
        decision.metadata.setdefault("remaining_primitives", len(self.queue))
        return decision

    @staticmethod
    def _task_complete(observation: dict[str, Any]) -> bool:
        objects = observation.get("objects", [])
        return bool(objects) and all(bool(obj.get("sorted")) for obj in objects)

    def _select_high_level_task(
        self, observation: dict[str, Any], held_objects: dict[str, str | None]
    ) -> ObjectTask:
        tasks = self._make_high_level_plan(observation, held_objects)
        if not tasks:
            raise RuntimeError(
                "No feasible object-level task found. Check robot base poses, workspace radius, goals, and object markers."
            )
        self.task_plan = tasks
        self.task_order = [task.object_id for task in tasks]
        self.plan_revision += 1
        return tasks[0]

    def _make_high_level_plan(
        self, observation: dict[str, Any], held_objects: dict[str, str | None]
    ) -> list[ObjectTask]:
        """Compute the object order before primitive decomposition.

        Ranking policy:
        - First, continue any object already held by a robot.
        - Then prefer feasible direct routes over handovers if scores are similar.
        - Otherwise use a route-distance score so the closest currently solvable
          object is processed first.
        - Ties are deterministic by object id.
        """
        unsorted = [
            obj for obj in observation.get("objects", []) if not obj.get("sorted")
        ]
        routes: list[RouteCandidate] = []
        for obj in unsorted:
            routes.extend(self._routes_for_object(observation, held_objects, obj))

        if not routes:
            return []

        def route_priority(route: RouteCandidate) -> tuple[float, int, str]:
            already_held = route.object_id in held_objects.values()
            route_penalty = {
                "direct_after_grip": 0,
                "direct": 1,
                "handover": 2,
            }.get(route.route_type, 9)
            return (
                0.0 if already_held else 1.0,
                route_penalty,
                f"{route.score:012.6f}|{route.object_id}",
            )

        # Keep only the best route per object.
        best_by_object: dict[str, RouteCandidate] = {}
        for route in sorted(routes, key=route_priority):
            best_by_object.setdefault(route.object_id, route)

        ordered_routes = sorted(best_by_object.values(), key=route_priority)
        tasks: list[ObjectTask] = []
        for index, route in enumerate(ordered_routes, start=1):
            rationale = self._high_level_rationale(route, observation, held_objects)
            tasks.append(ObjectTask(route.object_id, route, index, rationale))
        return tasks

    def _high_level_rationale(
        self,
        route: RouteCandidate,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
    ) -> str:
        obj = self._object_by_id(observation, route.object_id)
        color = obj.get("color") if obj else "unknown"
        if route.object_id in held_objects.values():
            holder = next(
                (rid for rid, oid in held_objects.items() if oid == route.object_id),
                "unknown",
            )
            return (
                f"Continue object {route.object_id}: it is already held by {holder}, "
                f"so finish the placement before selecting another object."
            )
        if route.route_type == "direct":
            return (
                f"Process {color} object {route.object_id} first: robot {route.source_robot_id} "
                f"can reach both the object and the corrected {color} goal drop pose."
            )
        if route.route_type == "handover":
            return (
                f"Process {color} object {route.object_id} via handover: robot {route.source_robot_id} "
                f"is assigned to the object side, robot {route.destination_robot_id} is assigned to the goal side, "
                "and the shared table-center buffer bridges their workspaces."
            )
        return route.reason or f"Process object {route.object_id}."

    def _routes_for_object(
        self,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
        obj: dict[str, Any],
    ) -> list[RouteCandidate]:
        object_id = obj["id"]
        goal = observation["goals"].get(obj["color"])
        if goal is None:
            return []

        routes: list[RouteCandidate] = []

        if object_id in held_objects.values():
            holder = next(
                (rid for rid, held in held_objects.items() if held == object_id), None
            )
            if holder is not None:
                drop_pose = self._choose_drop_pose(observation, goal, holder, obj)
                if drop_pose is not None and self._robot_can_reach_pose(
                    observation, holder, drop_pose
                ):
                    routes.append(
                        RouteCandidate(
                            object_id=object_id,
                            route_type="direct_after_grip",
                            source_robot_id=holder,
                            destination_robot_id=holder,
                            score=0.0,
                            drop_pose=drop_pose,
                            reason="The object is already held; finish the drop sequence.",
                        )
                    )
            return routes

        source_robots = [
            rid
            for rid in CONTROL_SERVICE_TOPICS
            if self._robot_can_reach_position(observation, rid, obj["pose"]["position"])
        ]
        if not source_robots:
            return []

        buffer_pose = self._choose_shared_buffer_pose(observation, obj)

        # Direct route: one robot can pick the object and reach a corrected goal pose.
        for source in source_robots:
            drop_pose = self._choose_drop_pose(observation, goal, source, obj)
            if drop_pose is None:
                continue
            routes.append(
                RouteCandidate(
                    object_id=object_id,
                    route_type="direct",
                    source_robot_id=source,
                    destination_robot_id=source,
                    score=self._route_distance_score(
                        observation, source, obj["pose"], drop_pose
                    ),
                    drop_pose=drop_pose,
                    reason="Single robot can reach both the object and corrected goal pose.",
                )
            )

        # Handover route: source handles the object side, destination handles the goal side.
        if buffer_pose is not None:
            for source in source_robots:
                if not self._robot_can_reach_pose(observation, source, buffer_pose):
                    continue
                for dest in CONTROL_SERVICE_TOPICS:
                    if dest == source:
                        continue
                    if not self._robot_can_reach_pose(observation, dest, buffer_pose):
                        continue
                    drop_pose = self._choose_drop_pose(observation, goal, dest, obj)
                    if drop_pose is None:
                        continue
                    routes.append(
                        RouteCandidate(
                            object_id=object_id,
                            route_type="handover",
                            source_robot_id=source,
                            destination_robot_id=dest,
                            score=(
                                self._route_distance_score(
                                    observation, source, obj["pose"], buffer_pose
                                )
                                + self._route_distance_score(
                                    observation, dest, buffer_pose, drop_pose
                                )
                                + 0.25
                            ),
                            drop_pose=drop_pose,
                            buffer_pose=buffer_pose,
                            reason="Object side and goal side are assigned to different robots through a shared buffer.",
                        )
                    )
        return routes

    def _build_steps(
        self, route: RouteCandidate, observation: dict[str, Any], task: ObjectTask
    ) -> list[Decision]:
        obj = self._object_by_id(observation, route.object_id)
        if obj is None:
            raise RuntimeError(f"Object disappeared before planning: {route.object_id}")

        safe_z = self._safe_z(observation)
        object_pose = make_pose(obj["pose"]["position"])
        object_above = raised_pose(object_pose, self.args.lift_delta, safe_z)
        drop_pose = route.drop_pose
        drop_above = raised_pose(drop_pose, self.args.lift_delta, safe_z)

        breakdown = self._breakdown_text(route, obj)
        common_meta = {
            "plan_revision": self.plan_revision,
            "high_level_rank": task.rank,
            "high_level_order": list(self.task_order),
            "active_object": route.object_id,
            "active_route": route.route_type,
            "task_rationale": task.rationale,
            "breakdown": breakdown,
            "corrected_goal_z_offset_m": float(self.args.goal_z_offset),
            "buffer_z_offset_m": float(self.args.buffer_z_offset),
            "drop_pose": drop_pose,
        }

        def attach(decisions: list[Decision]) -> list[Decision]:
            total = len(decisions)
            for idx, decision in enumerate(decisions, start=1):
                decision.metadata.update(common_meta)
                decision.metadata["primitive_index"] = idx
                decision.metadata["primitive_count"] = total
            return decisions

        if route.route_type == "direct_after_grip":
            rid = route.source_robot_id
            return attach(
                [
                    self._moving(
                        rid,
                        object_above,
                        obj["id"],
                        "lift",
                        "Current object is already held; lift it before lateral transfer.",
                    ),
                    self._moving(
                        rid,
                        drop_above,
                        obj["id"],
                        "drop_approach",
                        "Move above the selected corrected goal pose.",
                    ),
                    self._moving(
                        rid,
                        drop_pose,
                        obj["id"],
                        "drop",
                        "Lower to the selected goal pose with z offset for goal thickness.",
                    ),
                    self._release(
                        rid,
                        obj["id"],
                        "Release the object inside its matching goal.",
                        verify_type="final_drop",
                        expected_pose=drop_pose,
                    ),
                    self._homing(
                        rid, obj["id"], "Return the robot home after placing."
                    ),
                ]
            )

        if route.route_type == "direct":
            rid = route.source_robot_id
            return attach(
                [
                    self._moving(
                        rid,
                        object_pose,
                        obj["id"],
                        "pick",
                        "Move to the selected object pose.",
                    ),
                    self._grip(rid, obj["id"], "Grip the selected object."),
                    self._moving(
                        rid,
                        object_above,
                        obj["id"],
                        "lift",
                        "Lift before horizontal transfer.",
                    ),
                    self._moving(
                        rid,
                        drop_above,
                        obj["id"],
                        "drop_approach",
                        "Move above the selected corrected goal pose.",
                    ),
                    self._moving(
                        rid,
                        drop_pose,
                        obj["id"],
                        "drop",
                        "Lower to the corrected goal pose; z includes goal thickness offset.",
                    ),
                    self._release(
                        rid,
                        obj["id"],
                        "Release the object inside the matching goal.",
                        verify_type="final_drop",
                        expected_pose=drop_pose,
                    ),
                    self._homing(
                        rid, obj["id"], "Return the robot home after placing."
                    ),
                ]
            )

        if route.route_type == "handover":
            if route.buffer_pose is None:
                raise RuntimeError("Handover route lacks a buffer pose.")
            source = route.source_robot_id
            dest = route.destination_robot_id
            buffer_pose = route.buffer_pose
            buffer_above = raised_pose(buffer_pose, self.args.lift_delta, safe_z)
            return attach(
                [
                    self._moving(
                        source,
                        object_pose,
                        obj["id"],
                        "pick",
                        "Source robot moves to the selected object pose.",
                    ),
                    self._grip(source, obj["id"], "Source robot grips the object."),
                    self._moving(
                        source,
                        object_above,
                        obj["id"],
                        "lift",
                        "Source robot lifts the object for collision avoidance.",
                    ),
                    self._moving(
                        source,
                        buffer_above,
                        obj["id"],
                        "handover_approach",
                        "Source robot moves above the shared table-center buffer.",
                    ),
                    self._moving(
                        source,
                        buffer_pose,
                        obj["id"],
                        "handover_place",
                        "Source robot lowers the object onto the shared buffer.",
                    ),
                    self._release(
                        source,
                        obj["id"],
                        "Source robot releases the object at the shared buffer.",
                        verify_type="buffer_deposit",
                        expected_pose=buffer_pose,
                    ),
                    self._homing(
                        source,
                        obj["id"],
                        "Source robot homes after the handover deposit.",
                    ),
                    self._moving(
                        dest,
                        buffer_above,
                        obj["id"],
                        "handover_pick_approach",
                        "Destination robot moves above the shared buffer.",
                    ),
                    self._moving(
                        dest,
                        buffer_pose,
                        obj["id"],
                        "handover_pick",
                        "Destination robot lowers to the handed-over object.",
                    ),
                    self._grip(
                        dest,
                        obj["id"],
                        "Destination robot grips the object from the buffer.",
                    ),
                    self._moving(
                        dest,
                        buffer_above,
                        obj["id"],
                        "lift",
                        "Destination robot lifts the object before final transfer.",
                    ),
                    self._moving(
                        dest,
                        drop_above,
                        obj["id"],
                        "drop_approach",
                        "Destination robot moves above the corrected matching goal pose.",
                    ),
                    self._moving(
                        dest,
                        drop_pose,
                        obj["id"],
                        "drop",
                        "Destination robot lowers to the corrected goal pose; z includes goal thickness offset.",
                    ),
                    self._release(
                        dest,
                        obj["id"],
                        "Destination robot releases the object inside the matching goal.",
                        verify_type="final_drop",
                        expected_pose=drop_pose,
                    ),
                    self._homing(
                        dest, obj["id"], "Destination robot homes after placing."
                    ),
                ]
            )

        raise RuntimeError(f"Unsupported route type: {route.route_type}")

    def _breakdown_text(self, route: RouteCandidate, obj: dict[str, Any]) -> str:
        if route.route_type == "direct":
            return (
                f"High-level task: move {obj['id']} to the {obj['color']} goal. "
                f"Breakdown: {route.source_robot_id} pick -> grip -> lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        if route.route_type == "direct_after_grip":
            return (
                f"High-level task: finish placing already-held {obj['id']}. "
                "Breakdown: lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        if route.route_type == "handover":
            return (
                f"High-level task: move {obj['id']} to the {obj['color']} goal through handover. "
                f"Breakdown: {route.source_robot_id} pick -> grip -> lift -> buffer deposit -> release -> home; "
                f"then {route.destination_robot_id} buffer pick -> grip -> lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        return f"High-level task: move {obj['id']}."

    @staticmethod
    def _object_by_id(
        observation: dict[str, Any], object_id: str
    ) -> dict[str, Any] | None:
        for obj in observation.get("objects", []):
            if obj.get("id") == object_id:
                return obj
        return None

    def _moving(
        self,
        robot_id: str,
        pose: dict[str, Any],
        object_id: str | None,
        intent: str,
        reason: str,
    ) -> Decision:
        return Decision(
            robot_id, ACTION_MOVING, pose, object_id, intent=intent, reason=reason
        )

    def _grip(self, robot_id: str, object_id: str, reason: str) -> Decision:
        return Decision(
            robot_id, ACTION_GRIP, None, object_id, intent="grip", reason=reason
        )

    def _release(
        self,
        robot_id: str,
        object_id: str,
        reason: str,
        verify_type: str | None = None,
        expected_pose: dict[str, Any] | None = None,
    ) -> Decision:
        decision = Decision(
            robot_id, ACTION_RELEASE, None, object_id, intent="release", reason=reason
        )
        if verify_type is not None:
            decision.metadata["verify_after_action"] = {
                "type": verify_type,
                "object_id": object_id,
                "expected_pose": expected_pose,
            }
        return decision

    def _homing(self, robot_id: str, object_id: str | None, reason: str) -> Decision:
        return Decision(
            robot_id, ACTION_HOMING, None, object_id, intent="home", reason=reason
        )

    def _safe_z(self, observation: dict[str, Any]) -> float:
        table_center = observation["table"]["center"]
        return float(table_center.get("z", DEFAULT_TABLE_CENTER["z"])) + float(
            self.args.safe_z_offset
        )

    def _robot_workspace_radius(
        self, observation: dict[str, Any], robot_id: str
    ) -> float:
        if self.args.workspace_radius is not None:
            return float(self.args.workspace_radius)
        robot = observation["robots"].get(robot_id, {})
        base_pose = robot.get("base_pose")
        if base_pose is None:
            table_size = observation["table"].get("size", DEFAULT_TABLE_SIZE)
            return (
                max(float(table_size.get("x", 0.9)), float(table_size.get("y", 0.7)))
                * 0.65
            )
        center = observation["table"]["center"]
        return xy_distance(base_pose["position"], center) + float(
            self.args.center_reach_margin
        )

    def _robot_can_reach_position(
        self, observation: dict[str, Any], robot_id: str, position: dict[str, float]
    ) -> bool:
        robot = observation["robots"].get(robot_id, {})
        base_pose = robot.get("base_pose")
        if base_pose is None:
            return False
        return xy_distance(
            base_pose["position"], position
        ) <= self._robot_workspace_radius(observation, robot_id)

    def _robot_can_reach_pose(
        self, observation: dict[str, Any], robot_id: str, pose: dict[str, Any]
    ) -> bool:
        return self._robot_can_reach_position(observation, robot_id, pose["position"])

    def _route_distance_score(
        self,
        observation: dict[str, Any],
        robot_id: str,
        start: dict[str, Any],
        end: dict[str, Any],
    ) -> float:
        robot = observation["robots"].get(robot_id, {})
        eef_pose = robot.get("end_effector_pose") or robot.get("base_pose")
        start_position = start["position"] if "position" in start else start
        end_position = end["position"] if "position" in end else end
        score = xy_distance(start_position, end_position)
        if eef_pose is not None:
            score += 0.5 * xy_distance(eef_pose["position"], start_position)
        return score

    def _choose_shared_buffer_pose(
        self,
        observation: dict[str, Any],
        moving_obj: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Choose the shared handover buffer pose.

        Important: the table marker pose is the table center pose, not a valid
        object-placement pose.  If that z is used directly, the end-effector can
        be commanded into the table.  The buffer z is therefore derived from the
        current object's marker z, with an optional small offset.
        """
        table = observation["table"]
        center = dict(table.get("center", DEFAULT_TABLE_CENTER))
        object_position = moving_obj["pose"]["position"]
        center["z"] = float(object_position["z"]) + float(self.args.buffer_z_offset)
        center_pose = make_pose(center)
        if all(
            self._robot_can_reach_pose(observation, rid, center_pose)
            for rid in CONTROL_SERVICE_TOPICS
        ):
            return center_pose

        size = table.get("size", DEFAULT_TABLE_SIZE)
        half_x = (
            float(size.get("x", DEFAULT_TABLE_SIZE["x"])) * 0.5
            - DEFAULT_BUFFER_CLEARANCE_M
        )
        half_y = (
            float(size.get("y", DEFAULT_TABLE_SIZE["y"])) * 0.5
            - DEFAULT_BUFFER_CLEARANCE_M
        )
        samples: list[dict[str, float]] = []
        for ix in range(7):
            for iy in range(7):
                samples.append(
                    {
                        "x": float(center["x"]) - half_x + (2.0 * half_x * ix / 6.0),
                        "y": float(center["y"]) - half_y + (2.0 * half_y * iy / 6.0),
                        "z": float(center["z"]),
                    }
                )
        samples.sort(key=lambda p: xy_distance(p, center))
        for point in samples:
            pose = make_pose(point)
            if all(
                self._robot_can_reach_pose(observation, rid, pose)
                for rid in CONTROL_SERVICE_TOPICS
            ):
                return pose
        return None

    def _choose_drop_pose(
        self,
        observation: dict[str, Any],
        goal: dict[str, Any],
        robot_id: str,
        moving_obj: dict[str, Any],
    ) -> dict[str, Any] | None:
        occupied = [
            candidate
            for candidate in observation.get("objects", [])
            if candidate.get("id") != moving_obj.get("id")
            and point_in_goal(candidate["pose"]["position"], goal, margin=0.0)
        ]
        for pose in self._drop_pose_candidates(goal, occupied):
            if self._robot_can_reach_pose(observation, robot_id, pose):
                return pose
        return None

    def _drop_pose_candidates(
        self, goal: dict[str, Any], occupied: list[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        center = goal["pose"]["position"]
        scale = goal["scale"]
        half_x = max(0.0, float(scale["x"]) * 0.5 - float(self.args.goal_margin))
        half_y = max(0.0, float(scale["y"]) * 0.5 - float(self.args.goal_margin))
        step = float(self.args.drop_clearance)
        corrected_z = float(center["z"]) + float(self.args.goal_z_offset)
        x_count = max(1, int((half_x * 2.0) // step) + 1)
        y_count = max(1, int((half_y * 2.0) // step) + 1)
        points: list[dict[str, float]] = []
        for xi in range(x_count):
            for yi in range(y_count):
                points.append(
                    {
                        "x": float(center["x"])
                        - half_x
                        + (2.0 * half_x * xi / max(1, x_count - 1)),
                        "y": float(center["y"])
                        - half_y
                        + (2.0 * half_y * yi / max(1, y_count - 1)),
                        "z": corrected_z,
                    }
                )
        points.append(
            {"x": float(center["x"]), "y": float(center["y"]), "z": corrected_z}
        )
        points.sort(key=lambda p: xy_distance(p, center))
        for point in points:
            if all(
                xy_distance(point, obj["pose"]["position"]) >= step for obj in occupied
            ):
                yield make_pose(point)


# ---------------------------------------------------------------------------
# Executor and validation
# ---------------------------------------------------------------------------


class Executor:
    def __init__(
        self, node: RosWorldNode, args: argparse.Namespace, reporter: "Reporter"
    ) -> None:
        self.node = node
        self.args = args
        self.reporter = reporter
        self.history: list[dict[str, Any]] = []
        self.held_objects: dict[str, str | None] = {"left": None, "right": None}
        self.pending_grip_targets: dict[str, str | None] = {"left": None, "right": None}

    def validate(self, decision: Decision) -> None:
        if decision.done:
            return
        if decision.robot_id not in CONTROL_SERVICE_TOPICS:
            raise ValueError(f"Invalid robot_id: {decision.robot_id!r}")
        if decision.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {decision.action!r}")
        if decision.action == ACTION_MOVING and decision.target_pose is None:
            raise ValueError("Moving requires target_pose.")
        if decision.action == ACTION_GRIP and not decision.target_object_id:
            raise ValueError("Grip requires target_object_id for state tracking.")
        if decision.action == ACTION_RELEASE and not self.held_objects.get(
            str(decision.robot_id)
        ):
            raise ValueError(
                "Release requested while the selected robot is not holding an object."
            )
        if decision.action == ACTION_MOVING and decision.intent in {
            "lift",
            "handover_approach",
            "handover_place",
            "drop_approach",
            "drop",
        }:
            if not self.held_objects.get(str(decision.robot_id)):
                raise ValueError(
                    f"Moving intent={decision.intent!r} requires the robot to hold an object."
                )

    def execute(
        self, decision: Decision, observation: dict[str, Any]
    ) -> ExecutionResult:
        self.validate(decision)
        if decision.done:
            result = ExecutionResult(True, "Task marked complete.", decision)
            self._append_history(result)
            return result

        if self.args.dry_run:
            result = ExecutionResult(True, "dry-run: service call skipped", decision)
            self._update_held_state(decision, observation, True)
            self._append_history(result)
            return result

        assert decision.robot_id is not None
        success, message = self.node.call_control_service(
            decision.robot_id, decision.action, decision.target_pose
        )
        result = ExecutionResult(success, message, decision)

        if (
            success
            and decision.action == ACTION_MOVING
            and decision.target_pose is not None
        ):
            arrived = self.node.wait_for_eef_target(
                decision.robot_id,
                decision.target_pose,
                timeout_sec=float(self.args.motion_timeout),
            )
            if not arrived:
                result = ExecutionResult(
                    False,
                    f"{message}; EEF did not reach target before timeout",
                    decision,
                )
        elif success:
            self._settle()

        self._update_held_state(decision, observation, result.success)
        result = self._verify_after_action(result)
        self._append_history(result)
        return result

    def _verify_after_action(self, result: ExecutionResult) -> ExecutionResult:
        """Verify critical scene changes after a short observation delay.

        This prevents the next robot from blindly continuing a handover when the
        previous release did not actually put the object at the buffer or goal.
        The verification is marker/vision-state based: after the delay, a fresh
        ROS snapshot is read and compared with the expected object state.
        """
        decision = result.decision
        verify_spec = (
            decision.metadata.get("verify_after_action") if decision.metadata else None
        )
        if not result.success or not verify_spec or self.args.dry_run:
            return result

        self._wait_for_fresh_observation(float(self.args.verify_delay))
        try:
            updated_observation, _ = self.node.snapshot()
        except Exception as exc:
            return ExecutionResult(
                False,
                f"{result.message}; post-action verification snapshot failed: {exc}",
                decision,
            )

        ok, detail = self._check_verification_spec(updated_observation, verify_spec)
        if ok:
            return ExecutionResult(
                True, f"{result.message}; verified: {detail}", decision
            )
        return ExecutionResult(
            False, f"{result.message}; verification failed: {detail}", decision
        )

    def _wait_for_fresh_observation(self, delay_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, delay_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _check_verification_spec(
        self, observation: dict[str, Any], verify_spec: dict[str, Any]
    ) -> tuple[bool, str]:
        object_id = str(verify_spec.get("object_id") or "")
        obj = self._object_by_id(observation, object_id)
        if obj is None:
            return (
                False,
                f"object {object_id!r} is not visible in the updated marker state",
            )

        verification_type = str(verify_spec.get("type") or "")
        expected_pose = verify_spec.get("expected_pose") or {}
        expected_position = (
            expected_pose.get("position") if isinstance(expected_pose, dict) else None
        )
        actual_position = obj["pose"]["position"]

        if verification_type == "buffer_deposit":
            if not isinstance(expected_position, dict):
                return False, "buffer verification lacks expected_pose.position"
            xy_err = xy_distance(actual_position, expected_position)
            z_err = abs(float(actual_position["z"]) - float(expected_position["z"]))
            if xy_err <= float(self.args.verify_xy_tolerance) and z_err <= float(
                self.args.verify_z_tolerance
            ):
                return True, (
                    f"object {object_id} is at the shared buffer "
                    f"within xy_err={xy_err:.4f} m, z_err={z_err:.4f} m"
                )
            return False, (
                f"object {object_id} was expected at buffer {expected_position}, "
                f"but marker is at {actual_position}; xy_err={xy_err:.4f} m, z_err={z_err:.4f} m"
            )

        if verification_type == "final_drop":
            expected_color = obj.get("color")
            if bool(obj.get("sorted")):
                return (
                    True,
                    f"object {object_id} is marked sorted in the {expected_color} goal",
                )
            goal = observation.get("goals", {}).get(expected_color)
            inside_matching_goal = goal is not None and point_in_goal(
                actual_position, goal, margin=0.0
            )
            if inside_matching_goal:
                return (
                    True,
                    f"object {object_id} is inside the matching {expected_color} goal",
                )
            return False, (
                f"object {object_id} is not in its matching goal after release; "
                f"inside_goal={obj.get('inside_goal')}, sorted={obj.get('sorted')}, position={actual_position}"
            )

        return False, f"unknown verification type: {verification_type!r}"

    @staticmethod
    def _object_by_id(
        observation: dict[str, Any], object_id: str
    ) -> dict[str, Any] | None:
        for obj in observation.get("objects", []):
            if obj.get("id") == object_id:
                return obj
        return None

    def _settle(self) -> None:
        deadline = time.monotonic() + float(self.args.settle_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _update_held_state(
        self, decision: Decision, observation: dict[str, Any], success: bool
    ) -> None:
        if not success or decision.robot_id not in self.held_objects:
            return
        robot_id = str(decision.robot_id)
        if decision.action == ACTION_MOVING and decision.intent in {
            "pick",
            "handover_pick",
        }:
            self.pending_grip_targets[robot_id] = decision.target_object_id
        elif decision.action == ACTION_GRIP:
            self.held_objects[robot_id] = (
                decision.target_object_id or self.pending_grip_targets.get(robot_id)
            )
            self.pending_grip_targets[robot_id] = None
        elif decision.action == ACTION_RELEASE:
            self.held_objects[robot_id] = None
            self.pending_grip_targets[robot_id] = None

    def _append_history(self, result: ExecutionResult) -> None:
        d = result.decision
        self.history.append(
            {
                "time": time.time(),
                "success": result.success,
                "message": result.message,
                "robot_id": d.robot_id,
                "action": d.action,
                "intent": d.intent,
                "target_object_id": d.target_object_id,
                "target_pose": d.target_pose,
                "reason": d.reason,
                "held_objects": dict(self.held_objects),
                "pending_grip_targets": dict(self.pending_grip_targets),
            }
        )


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------


class Reporter:
    def __init__(self, plain: bool, verbose: bool) -> None:
        self.plain = plain or not RICH_AVAILABLE
        self.verbose = verbose
        self.console = Console() if not self.plain else None

    def info(self, message: str) -> None:
        if self.console is not None:
            self.console.print(message)
        else:
            print(message)

    def error(self, message: str) -> None:
        if self.console is not None:
            self.console.print(f"[bold red]{message}[/bold red]")
        else:
            print(message, file=sys.stderr)

    def status(self, title: str, rows: dict[str, Any]) -> None:
        if self.plain:
            flat = ", ".join(f"{key}={value}" for key, value in rows.items())
            print(f"{title}: {flat}")
            return
        assert self.console is not None
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        for key, value in rows.items():
            table.add_row(str(key), str(value))
        self.console.print(Panel(table, title=title, expand=False))

    def decision(self, step: int, decision: Decision, queue_len: int) -> None:
        rows: dict[str, Any] = {
            "step": step,
            "robot": decision.robot_id,
            "action": decision.action,
            "intent": decision.intent,
            "target_object": decision.target_object_id,
            "queued_after_this": queue_len,
            "reason": decision.reason,
        }
        if decision.metadata:
            rows["high_level_order"] = decision.metadata.get("high_level_order")
            rows["phase"] = (
                f"{decision.metadata.get('primitive_index')}/{decision.metadata.get('primitive_count')}"
            )
        if self.verbose:
            if decision.target_pose is not None:
                rows["target_pose"] = json.dumps(
                    decision.target_pose, ensure_ascii=False
                )
            if decision.metadata:
                rows["task_rationale"] = decision.metadata.get("task_rationale")
                rows["breakdown"] = decision.metadata.get("breakdown")
        self.status("Planned action", rows)


# ---------------------------------------------------------------------------
# Runtime utilities
# ---------------------------------------------------------------------------


def wait_for_ready(node: RosWorldNode, reporter: Reporter) -> None:
    last_report = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        readiness = node.readiness()
        if all(readiness.values()):
            return
        now = time.monotonic()
        if now - last_report >= 3.0:
            waiting = [key for key, ready in readiness.items() if not ready]
            reporter.status(
                "Waiting for ROS2 observations", {"missing": ", ".join(waiting)}
            )
            last_report = now


def save_frame(directory: Path, step: int, image_bytes: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"step_{step:03d}.png").write_bytes(image_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic dual-Franka CLI with explicit handover planning."
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="High-level task description for logs.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=160, help="Maximum primitive-action steps."
    )
    parser.add_argument(
        "--once", action="store_true", help="Execute only one primitive action."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ROS2 service calls and only update internal state.",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for Moving convergence.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.7,
        help="Seconds to wait after grip/release/home.",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="Directory to save top-view frames.",
    )
    parser.add_argument(
        "--plain", action="store_true", help="Disable rich terminal UI."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print target poses and extra debugging details.",
    )

    parser.add_argument(
        "--workspace-radius",
        type=float,
        default=None,
        help="Override per-robot XY reach radius in meters. Default: distance(base, table_center)+center margin.",
    )
    parser.add_argument(
        "--center-reach-margin",
        type=float,
        default=DEFAULT_CENTER_REACH_MARGIN_M,
        help="Additional reach beyond table center for the automatic workspace model.",
    )
    parser.add_argument(
        "--lift-delta",
        type=float,
        default=DEFAULT_LIFT_DELTA_M,
        help="Vertical lift added after grasp.",
    )
    parser.add_argument(
        "--safe-z-offset",
        type=float,
        default=DEFAULT_SAFE_Z_OFFSET_M,
        help="Safe Z above table center.",
    )
    parser.add_argument(
        "--goal-margin",
        type=float,
        default=DEFAULT_GOAL_MARGIN_M,
        help="Inset margin inside goal region.",
    )
    parser.add_argument(
        "--drop-clearance",
        type=float,
        default=DEFAULT_DROP_CLEARANCE_M,
        help="Minimum XY spacing between dropped objects.",
    )
    parser.add_argument(
        "--goal-z-offset",
        type=float,
        default=DEFAULT_GOAL_Z_OFFSET_M,
        help="Additional z offset added to every goal drop pose to compensate for goal thickness.",
    )
    parser.add_argument(
        "--buffer-z-offset",
        type=float,
        default=DEFAULT_BUFFER_Z_OFFSET_M,
        help="Additional z offset added to the object-marker z when placing at the shared table-center buffer.",
    )
    parser.add_argument(
        "--verify-delay",
        type=float,
        default=DEFAULT_VERIFY_DELAY_SEC,
        help="Delay before reading a fresh marker/image state for post-release verification.",
    )
    parser.add_argument(
        "--verify-xy-tolerance",
        type=float,
        default=DEFAULT_VERIFY_XY_TOLERANCE_M,
        help="XY tolerance for verifying that an object was deposited at the shared buffer.",
    )
    parser.add_argument(
        "--verify-z-tolerance",
        type=float,
        default=DEFAULT_VERIFY_Z_TOLERANCE_M,
        help="Z tolerance for verifying that an object was deposited at the shared buffer.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if ROS_IMPORT_ERROR is not None:
        print(
            "ROS2 imports failed. Source the ROS2 workspace before running this CLI, "
            "then run it with the ROS2 Python interpreter, for example:\n"
            "  source install/setup.bash && /usr/bin/python3 web-app/cli.py",
            file=sys.stderr,
        )
        print(f"Import error: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 2

    reporter = Reporter(plain=args.plain, verbose=args.verbose)
    reporter.status(
        "Startup",
        {
            "planner": "hierarchical_task_director_with_handover_decomposer",
            "dry_run": args.dry_run,
            "command": args.command,
            "workspace_radius": args.workspace_radius,
            "center_reach_margin": args.center_reach_margin,
            "goal_z_offset": args.goal_z_offset,
            "buffer_z_offset": args.buffer_z_offset,
            "verify_delay": args.verify_delay,
        },
    )

    rclpy.init()
    node = RosWorldNode()
    planner = HandoverTaskPlanner(args)
    executor = Executor(node, args, reporter)

    try:
        wait_for_ready(node, reporter)
        for step in range(1, int(args.max_steps) + 1):
            observation, image_bytes = node.snapshot()
            if args.save_frames is not None:
                save_frame(args.save_frames, step, image_bytes)

            reporter.status(
                "Observation",
                {
                    "step": step,
                    "unsorted": observation["summary"]["unsorted_count"],
                    "objects": observation["summary"]["object_count"],
                    "held": executor.held_objects,
                    "active_object": planner.active_object_id,
                    "active_route": planner.active_route_type,
                    "high_level_order": planner.task_order,
                },
            )

            try:
                decision = planner.next_decision(observation, executor.held_objects)
                reporter.decision(step, decision, queue_len=len(planner.queue))
                result = executor.execute(decision, observation)
            except Exception as exc:
                planner.clear_active_plan()
                reporter.error(f"Planning/execution error: {exc}")
                return 1

            reporter.status(
                "Execution result",
                {"success": result.success, "message": result.message},
            )
            if decision.done:
                return 0
            if not result.success:
                planner.clear_active_plan()
                reporter.error(
                    "Cleared the active plan because the previous primitive failed."
                )
                if args.once:
                    return 1
            if args.once:
                return 0 if result.success else 1

        reporter.error(f"Stopped after reaching --max-steps={args.max_steps}.")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
