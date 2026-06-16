#!/usr/bin/python3
"""ChatGPT-only multi-agent CLI for the dual-Franka Isaac Sim scene.

This file intentionally does not use any local LLM backend. The agents call the
OpenAI-compatible Chat Completions API and coordinate through explicit JSON
messages, ROS2 snapshots, and per-agent context/log files.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except Exception:
    pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional terminal UI
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Prompt = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    RICH_AVAILABLE = False

ROS_IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from custom_msgs.srv import ControlCommand
    from geometry_msgs.msg import Point, Pose, PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import MarkerArray
except Exception as exc:  # pragma: no cover - depends on sourced ROS2 workspace
    ROS_IMPORT_ERROR = exc
    rclpy = None  # type: ignore[assignment]
    ControlCommand = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    Pose = None  # type: ignore[assignment]
    PoseStamped = None  # type: ignore[assignment]
    RosImage = None  # type: ignore[assignment]
    JointState = None  # type: ignore[assignment]
    MarkerArray = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass


BASE_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = BASE_DIR / "agent_context"
LOG_DIR = BASE_DIR / "logs" / "agents"

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SEC = 120

DEFAULT_COMMAND = "빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라."
USE_DEFAULT_ON_EMPTY_INPUT = True
SEND_IMAGE_TO_AGENTS = True

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
JOINT_STATE_TOPICS = {
    "left": "/franka_left/joint_states",
    "right": "/franka_right/joint_states",
}
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}
ROBOT_SCENE_LABELS = {
    "left": "bottom_robot_in_top_view",
    "right": "top_robot_in_top_view",
}

ACTION_MOVING = "Moving"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Release"
ACTION_HOMING = "Homing"
ALLOWED_ACTIONS = {ACTION_MOVING, ACTION_GRIP, ACTION_RELEASE, ACTION_HOMING}

VERTICAL_EEF_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
DEFAULT_TABLE_CENTER = {"x": 0.6, "y": 0.0, "z": 0.46}
DEFAULT_TABLE_SIZE = {"x": 0.9, "y": 0.7, "z": 0.08}

SERVICE_WAIT_TIMEOUT_SEC = 1.0
SERVICE_CALL_TIMEOUT_SEC = 20.0
ACTION_INTERVAL_SEC = 1.0
MOVING_SETTLE_SEC = 1.0
GRIP_SETTLE_SEC = 1.8
RELEASE_SETTLE_SEC = 1.2
HOMING_SETTLE_SEC = 1.0
PENDING_OBSERVATION_SEC = 1.0
MAX_PENDING_DIAGNOSES = 8
READINESS_REPORT_SEC = 3.0
MAX_MAIN_STEPS = 120
MAX_REPLAN_ATTEMPTS_PER_TASK = 3
OBJECT_APPROACH_XY_TOL = 0.06
OBJECT_APPROACH_Z_TOL = 0.035
PRE_GRIP_OBJECT_XY_TOL = 0.04
PRE_GRIP_OBJECT_Z_TOL = 0.018
EEF_REACHED_RELAXED_XY_TOL = 0.04
EEF_REACHED_RELAXED_Z_TOL = 0.018
GOAL_DROP_EEF_Z_OFFSET = 0.095
GOAL_DROP_Z_TOL = 0.012
TRANSFER_CLEARANCE_Z_OFFSET = 0.12
POST_GRIP_LIFT_MIN_Z_OFFSET = 0.07
POST_GRIP_LOCAL_XY_TOL = 0.07
WORKSPACE_RADIUS_FRACTION_OF_TABLE_LENGTH = 0.65
REACH_BORDERLINE_MARGIN_M = 0.06
TROUBLESHOOTER_STATUSES = {
    "success",
    "pending",
    "retry",
    "replan_task",
    "replan_all",
    "emergency_recover",
    "complete",
}


@dataclass
class PrimitiveAction:
    robot_id: str | None
    action: str
    target_pose: dict[str, Any] | None = None
    target_object_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "action": self.action,
            "target_pose": self.target_pose,
            "target_object_id": self.target_object_id,
        }


@dataclass
class ActionResult:
    success: bool
    message: str
    action: PrimitiveAction
    snapshot_before: dict[str, Any] | None = None
    snapshot_after: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "action": self.action.as_dict(),
        }


@dataclass
class AgentReply:
    parsed: dict[str, Any]
    raw_text: str


class Reporter:
    def __init__(self) -> None:
        self.console = Console() if RICH_AVAILABLE else None

    def print(self, message: str) -> None:
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
        if self.console is None:
            flat = ", ".join(f"{key}={value}" for key, value in rows.items())
            print(f"{title}: {flat}")
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        for key, value in rows.items():
            table.add_row(str(key), str(value))
        self.console.print(Panel(table, title=title, expand=False))


def ensure_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("agent response must be a JSON object")
    return parsed


def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach {url}: {exc.reason}") from exc


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
    return "red" if float(color.get("r", 0.0)) >= float(color.get("b", 0.0)) else "blue"


def shape_from_namespace(namespace: str) -> str:
    prefix = namespace.split("_", 1)[0].lower()
    if prefix in {"cube", "sphere", "capsule"}:
        return prefix
    return "unknown"


def ros_pose_from_dict(pose_dict: dict[str, Any] | None) -> Any:
    pose = Pose()
    if pose_dict is None:
        pose_dict = {"position": DEFAULT_TABLE_CENTER}
    position = pose_dict.get("position", pose_dict)
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


def image_msg_to_png_bytes(message: Any) -> bytes:
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "rgba8", "bgr8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    raw = bytes(message.data)
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channels = 4 if "a" in encoding else 3
    mode = "RGBA" if channels == 4 else "RGB"
    expected_min = height * step
    if len(raw) < expected_min:
        raise ValueError(
            f"Image data shorter than expected: got={len(raw)}, expected_at_least={expected_min}"
        )
    image = Image.frombytes(mode, (width, height), raw, "raw", mode, step, 1)
    if encoding.startswith("bgr"):
        r, g, b = image.convert("RGB").split()
        image = Image.merge("RGB", (b, g, r))
    else:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")


def position_dict(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def pose_with_position(
    position: dict[str, float], orientation: dict[str, float] | None = None
) -> dict[str, Any]:
    return {
        "position": position,
        "orientation": orientation or VERTICAL_EEF_ORIENTATION,
    }


def compact_pose(pose: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(pose, dict):
        return None
    position = pose.get("position") if isinstance(pose.get("position"), dict) else {}
    orientation = (
        pose.get("orientation") if isinstance(pose.get("orientation"), dict) else {}
    )
    return {
        "position": {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        },
        "orientation": {
            "x": float(orientation.get("x", 0.0)),
            "y": float(orientation.get("y", 0.0)),
            "z": float(orientation.get("z", 0.0)),
            "w": float(orientation.get("w", 1.0)),
        },
    }


def gripper_summary(joint_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(joint_state, dict):
        return {
            "state": "unknown",
            "finger_positions": [],
            "mean_finger_position": None,
        }
    names = (
        joint_state.get("names") if isinstance(joint_state.get("names"), list) else []
    )
    positions = (
        joint_state.get("positions")
        if isinstance(joint_state.get("positions"), list)
        else []
    )
    finger_positions: list[float] = []
    for name, position in zip(names, positions):
        if "finger" in str(name):
            try:
                finger_positions.append(float(position))
            except (TypeError, ValueError):
                pass
    if not finger_positions:
        return {
            "state": "unknown",
            "finger_positions": [],
            "mean_finger_position": None,
        }
    mean_position = sum(finger_positions) / len(finger_positions)
    if mean_position >= 0.033:
        state = "open"
    elif mean_position <= 0.012:
        state = "closed"
    else:
        state = "partially_closed_or_holding"
    return {
        "state": state,
        "finger_positions": finger_positions,
        "mean_finger_position": mean_position,
        "interpretation": "larger values are more open; smaller values are more closed",
    }


def scale_value(scale: dict[str, Any] | None, axis: str, default: float) -> float:
    if not isinstance(scale, dict):
        return default
    try:
        return float(scale.get(axis, default))
    except (TypeError, ValueError):
        return default


def point_xy_distance(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> float | None:
    point_a = xyz_from_pose_dict(a)
    point_b = xyz_from_pose_dict(b)
    if point_a is None or point_b is None:
        return None
    return (
        (point_a["x"] - point_b["x"]) ** 2 + (point_a["y"] - point_b["y"]) ** 2
    ) ** 0.5


def reach_summary_for_pose(
    pose: dict[str, Any] | None,
    robots: dict[str, Any],
    workspace_radius: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for robot_id, robot in robots.items():
        if not isinstance(robot, dict):
            continue
        distance = point_xy_distance((robot.get("base_pose") or {}), pose)
        if distance is None:
            result[robot_id] = {"distance_xy": None, "assessment": "unknown"}
            continue
        margin = workspace_radius - distance
        if margin >= 0.0:
            assessment = "reachable"
        elif margin >= -REACH_BORDERLINE_MARGIN_M:
            assessment = "borderline"
        else:
            assessment = "likely_unreachable"
        result[robot_id] = {
            "distance_xy": distance,
            "workspace_radius": workspace_radius,
            "margin": margin,
            "assessment": assessment,
        }
    return result


def nearest_robot_for_pose(
    pose: dict[str, Any] | None,
    robots: dict[str, Any],
    workspace_radius: float,
) -> str | None:
    reach = reach_summary_for_pose(pose, robots, workspace_radius)
    best_robot = None
    best_distance = float("inf")
    for robot_id, info in reach.items():
        distance = info.get("distance_xy")
        if isinstance(distance, (int, float)) and distance < best_distance:
            best_robot = robot_id
            best_distance = float(distance)
    return best_robot


def reach_is_usable(reach_info: Any) -> bool:
    return (
        isinstance(reach_info, dict)
        and reach_info.get("assessment") in {"reachable", "borderline"}
    )


def build_agent_world_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    table = snapshot.get("table") if isinstance(snapshot.get("table"), dict) else {}
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    robots = snapshot.get("robots") if isinstance(snapshot.get("robots"), dict) else {}
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    table_pose = compact_pose(table.get("pose")) or {"position": DEFAULT_TABLE_CENTER}
    table_position = table_pose.get("position", DEFAULT_TABLE_CENTER)
    table_scale = (
        table.get("scale")
        if isinstance(table.get("scale"), dict)
        else DEFAULT_TABLE_SIZE
    )
    table_length = scale_value(table_scale, "x", DEFAULT_TABLE_SIZE["x"])
    workspace_radius = table_length * WORKSPACE_RADIUS_FRACTION_OF_TABLE_LENGTH
    object_z_values = [
        float(
            obj.get("pose", {}).get("position", {}).get("z", DEFAULT_TABLE_CENTER["z"])
        )
        for obj in objects
        if isinstance(obj, dict) and isinstance(obj.get("pose"), dict)
    ]
    suggested_transfer_clearance_z = (
        max(object_z_values) + TRANSFER_CLEARANCE_Z_OFFSET
        if object_z_values
        else DEFAULT_TABLE_CENTER["z"] + TRANSFER_CLEARANCE_Z_OFFSET
    )
    handover_z = (
        max(object_z_values) + GOAL_DROP_EEF_Z_OFFSET
        if object_z_values
        else DEFAULT_TABLE_CENTER["z"] + GOAL_DROP_EEF_Z_OFFSET
    )
    table_center_handover_pose = pose_with_position(
        position_dict(
            float(table_position.get("x", DEFAULT_TABLE_CENTER["x"])),
            float(table_position.get("y", DEFAULT_TABLE_CENTER["y"])),
            handover_z,
        )
    )
    named_locations = {
        "table_center_handover": {
            "id": "table_center_handover",
            "pose": table_center_handover_pose,
            "purpose": "shared buffer/handover location for objects that one robot can pick but the other robot should deliver",
            "reach": reach_summary_for_pose(
                table_center_handover_pose, robots, workspace_radius
            ),
        }
    }
    enriched_goals: dict[str, Any] = {}
    for color, goal in goals.items():
        if not isinstance(goal, dict):
            continue
        goal_position = (goal.get("pose") or {}).get("position", {})
        minimum_safe_release_z = (
            float(goal_position.get("z", 0.0)) + GOAL_DROP_EEF_Z_OFFSET
        )
        enriched_goals[color] = {
            "id": goal.get("id"),
            "color": goal.get("color", color),
            "physical_marker_pose": compact_pose(goal.get("pose")),
            "minimum_safe_release_z": minimum_safe_release_z,
            "eef_drop_pose": pose_with_position(
                position_dict(
                    float(goal_position.get("x", 0.0)),
                    float(goal_position.get("y", 0.0)),
                    minimum_safe_release_z,
                )
            ),
            "scale": goal.get("scale"),
            "reach": reach_summary_for_pose(goal.get("pose"), robots, workspace_radius),
            "nearest_robot": nearest_robot_for_pose(
                goal.get("pose"), robots, workspace_radius
            ),
            "note": "physical_marker_pose is the goal surface marker and is too low for Release. Use eef_drop_pose/minimum_safe_release_z or another safe pose above the goal for end-effector placement",
        }
    enriched_objects: dict[str, Any] = {}
    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        object_id = str(obj.get("id"))
        object_color = str(obj.get("color") or "")
        target_goal = enriched_goals.get(object_color)
        object_reach = reach_summary_for_pose(obj.get("pose"), robots, workspace_radius)
        pickup_robot = nearest_robot_for_pose(obj.get("pose"), robots, workspace_radius)
        destination_robot = (
            target_goal.get("nearest_robot") if isinstance(target_goal, dict) else None
        )
        goal_reach = (
            target_goal.get("reach")
            if isinstance(target_goal, dict) and isinstance(target_goal.get("reach"), dict)
            else {}
        )
        direct_robot_candidates = [
            robot_id
            for robot_id in robots
            if reach_is_usable(object_reach.get(robot_id))
            and reach_is_usable(goal_reach.get(robot_id))
        ]
        preferred_direct_robot = None
        if direct_robot_candidates:
            if destination_robot in direct_robot_candidates:
                preferred_direct_robot = destination_robot
            elif pickup_robot in direct_robot_candidates:
                preferred_direct_robot = pickup_robot
            else:
                preferred_direct_robot = direct_robot_candidates[0]
        handover_needed = (
            not direct_robot_candidates
            and bool(pickup_robot)
            and bool(destination_robot)
            and pickup_robot != destination_robot
        )
        if preferred_direct_robot:
            route_hint = (
                f"direct single-robot task preferred with {preferred_direct_robot}; "
                "do not use table_center_handover unless the direct robot actually fails or the image shows a clear obstruction"
            )
        elif handover_needed:
            route_hint = (
                f"handover may be needed: {pickup_robot} can pick near the object, "
                f"then {destination_robot} can deliver near {target_goal.get('id')}"
            )
        elif pickup_robot and destination_robot:
            route_hint = f"direct single-robot task may be suitable with {pickup_robot}"
        else:
            route_hint = "insufficient reach data; reason from poses and image"
        enriched_objects[object_id] = {
            "id": obj.get("id"),
            "shape": obj.get("shape"),
            "color": obj.get("color"),
            "pose": compact_pose(obj.get("pose")),
            "scale": obj.get("scale"),
            "target_goal_color": object_color,
            "target_goal_id": (
                target_goal.get("id") if isinstance(target_goal, dict) else None
            ),
            "reach": object_reach,
            "nearest_robot": pickup_robot,
            "destination_nearest_robot": destination_robot,
            "direct_robot_candidates": direct_robot_candidates,
            "preferred_direct_robot": preferred_direct_robot,
            "handover_needed": handover_needed,
            "route_hint": route_hint,
        }
    return {
        "table": {
            "id": table.get("id"),
            "pose": table_pose,
            "scale": table_scale,
        },
        "named_locations": named_locations,
        "goals": enriched_goals,
        "objects_by_id": enriched_objects,
        "robots": {
            robot_id: {
                "scene_label": robot.get("scene_label"),
                "base_pose": compact_pose(robot.get("base_pose")),
                "end_effector_pose": compact_pose(robot.get("end_effector_pose")),
                "joint_state": robot.get("joint_state"),
                "gripper": gripper_summary(robot.get("joint_state")),
            }
            for robot_id, robot in robots.items()
            if isinstance(robot, dict)
        },
        "motion_model": {
            "moving_service_behavior": (
                "A Moving command is queued in Isaac as horizontal XY/orientation motion at the current EEF Z, "
                "then vertical Z motion at the target XY."
            ),
            "planning_implication": (
                "A direct Moving command from a low grasp pose to a distant goal can sweep low across the table. "
                "If carrying an object or crossing crowded space, consider explicit lift/clearance and destination-approach Moving waypoints."
            ),
            "suggested_transfer_clearance_z": suggested_transfer_clearance_z,
            "carry_clearance_guidance": (
                "After Grip, treat the gripper and object as one carried body. "
                "If the next motion changes XY meaningfully, a plan that first raises Z at the current/object XY, "
                "then moves horizontally at that higher Z, then descends near the destination is usually safer than a low direct XY sweep. "
                "This is especially important for goal boundaries, table guards, other objects, and handover transfers."
            ),
            "note": "This is guidance for the agents, not a runtime-enforced waypoint rule.",
        },
        "workspace_model": {
            "workspace_radius_assumption": "approximate planar reach radius, table_length * 0.65",
            "workspace_radius": workspace_radius,
            "borderline_margin": REACH_BORDERLINE_MARGIN_M,
            "route_planning_hint": (
                "Prefer a direct single-robot task whenever an object has a non-empty direct_robot_candidates list. "
                "Nearest_robot differences alone are not enough reason for handover. "
                "Use named_locations.table_center_handover only when handover_needed is true or direct execution has actually failed."
            ),
        },
        "critical_notes": [
            "robot_id 'left' is the bottom robot in the top-view image.",
            "robot_id 'right' is the top robot in the top-view image.",
            "Task decomposition should prefer preferred_direct_robot when present. Handover is exceptional, not the default.",
            "Mention robot_id and intermediate handover/buffer location only when one robot cannot reasonably do both pickup and delivery.",
            "For an object-moving task, the first Moving action must target the object's current pose.",
            "Only after Grip may a later Moving action target a goal/drop pose.",
            "The service action spelling is 'Release'. The older typo is accepted only for backward compatibility.",
            "Goal marker poses are physical surface markers. End-effector drop poses must be above those markers.",
            "Never Release at a goal physical_marker_pose. Release near a goal must use eef_drop_pose/minimum_safe_release_z or a safer higher pose.",
            "Because Moving first translates horizontally at current Z, use explicit lifted waypoints when low horizontal transfer could collide with objects.",
            "After Grip, meaningful XY transfer must include Z lift before horizontal movement.",
            "If replanning while a gripper is closed and no object is securely held, open it with Release before descending onto an object.",
        ],
    }


def xyz_from_pose_dict(pose: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(pose, dict):
        return None
    position = pose.get("position", pose)
    if not isinstance(position, dict):
        return None
    try:
        return {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        }
    except (TypeError, ValueError):
        return None


def pose_distance_ok(
    pose_a: dict[str, Any] | None,
    pose_b: dict[str, Any] | None,
    xy_tol: float = OBJECT_APPROACH_XY_TOL,
    z_tol: float = OBJECT_APPROACH_Z_TOL,
) -> bool:
    a = xyz_from_pose_dict(pose_a)
    b = xyz_from_pose_dict(pose_b)
    if a is None or b is None:
        return False
    return (
        abs(a["x"] - b["x"]) <= xy_tol
        and abs(a["y"] - b["y"]) <= xy_tol
        and abs(a["z"] - b["z"]) <= z_tol
    )


def pose_error(
    current_pose: dict[str, Any] | None,
    target_pose: dict[str, Any] | None,
) -> dict[str, float] | None:
    current = xyz_from_pose_dict(current_pose)
    target = xyz_from_pose_dict(target_pose)
    if current is None or target is None:
        return None
    return {
        "dx": current["x"] - target["x"],
        "dy": current["y"] - target["y"],
        "dz": current["z"] - target["z"],
        "xy": ((current["x"] - target["x"]) ** 2 + (current["y"] - target["y"]) ** 2)
        ** 0.5,
        "z_abs": abs(current["z"] - target["z"]),
    }


def pose_reached(
    current_pose: dict[str, Any] | None,
    target_pose: dict[str, Any] | None,
    xy_tol: float = OBJECT_APPROACH_XY_TOL,
    z_tol: float = OBJECT_APPROACH_Z_TOL,
) -> bool:
    error = pose_error(current_pose, target_pose)
    if error is None:
        return False
    return error["xy"] <= xy_tol and error["z_abs"] <= z_tol


def xy_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    point_a = xyz_from_pose_dict(a)
    point_b = xyz_from_pose_dict(b)
    if point_a is None or point_b is None:
        return None
    return (
        (point_a["x"] - point_b["x"]) ** 2 + (point_a["y"] - point_b["y"]) ** 2
    ) ** 0.5


def pose_position(pose: dict[str, Any] | None) -> dict[str, float] | None:
    return xyz_from_pose_dict(pose)


def goal_containing_pose(
    pose: dict[str, Any] | None,
    goals: dict[str, Any],
    margin: float = 0.03,
) -> tuple[str, dict[str, Any]] | None:
    point = pose_position(pose)
    if point is None:
        return None
    for color, goal in goals.items():
        if not isinstance(goal, dict):
            continue
        goal_pose = pose_position(goal.get("pose"))
        scale = goal.get("scale") if isinstance(goal.get("scale"), dict) else {}
        if goal_pose is None:
            continue
        half_x = float(scale.get("x", 0.0)) / 2.0 + margin
        half_y = float(scale.get("y", 0.0)) / 2.0 + margin
        inside_x = abs(point["x"] - goal_pose["x"]) <= half_x
        inside_y = abs(point["y"] - goal_pose["y"]) <= half_y
        if inside_x and inside_y:
            return str(color), goal
    return None


def required_goal_drop_z(goal: dict[str, Any]) -> float | None:
    goal_pose = pose_position(goal.get("pose"))
    if goal_pose is None:
        return None
    return goal_pose["z"] + GOAL_DROP_EEF_Z_OFFSET


def xyz_delta(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, float] | None:
    point_a = xyz_from_pose_dict(a)
    point_b = xyz_from_pose_dict(b)
    if point_a is None or point_b is None:
        return None
    return {
        "dx": point_b["x"] - point_a["x"],
        "dy": point_b["y"] - point_a["y"],
        "dz": point_b["z"] - point_a["z"],
    }


def object_by_id(
    snapshot: dict[str, Any] | None, object_id: str | None
) -> dict[str, Any] | None:
    if not object_id or not isinstance(snapshot, dict):
        return None
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    for obj in objects:
        if isinstance(obj, dict) and str(obj.get("id")) == object_id:
            return obj
    return None


def robot_by_id(
    snapshot: dict[str, Any] | None, robot_id: str | None
) -> dict[str, Any] | None:
    if not robot_id or not isinstance(snapshot, dict):
        return None
    robots = snapshot.get("robots") if isinstance(snapshot.get("robots"), dict) else {}
    robot = robots.get(robot_id)
    return robot if isinstance(robot, dict) else None


def joint_motion_summary(joint_state: dict[str, Any] | None) -> dict[str, Any]:
    velocities = (
        joint_state.get("velocities") if isinstance(joint_state, dict) else None
    )
    if not isinstance(velocities, list) or not velocities:
        return {"max_abs_velocity": None, "appears_moving": "unknown"}
    numeric_velocities: list[float] = []
    for value in velocities:
        try:
            numeric_velocities.append(abs(float(value)))
        except (TypeError, ValueError):
            pass
    if not numeric_velocities:
        return {"max_abs_velocity": None, "appears_moving": "unknown"}
    max_abs_velocity = max(numeric_velocities)
    return {
        "max_abs_velocity": max_abs_velocity,
        "appears_moving": max_abs_velocity > 0.05,
        "note": "Use together with EEF pose error; tiny residual EEF error with low velocity should not stay pending.",
    }


def build_execution_diagnostics(
    result: ActionResult, snapshot_after: dict[str, Any]
) -> dict[str, Any]:
    before = result.snapshot_before or {}
    action = result.action
    robot_before = robot_by_id(before, action.robot_id)
    robot_after = robot_by_id(snapshot_after, action.robot_id)
    object_before = object_by_id(before, action.target_object_id)
    object_after = object_by_id(snapshot_after, action.target_object_id)
    eef_before = (robot_before or {}).get("end_effector_pose")
    eef_after = (robot_after or {}).get("end_effector_pose")
    eef_error_after = pose_error(eef_after, action.target_pose)
    eef_reached_standard = pose_reached(eef_after, action.target_pose)
    eef_reached_relaxed = pose_reached(
        eef_after,
        action.target_pose,
        xy_tol=EEF_REACHED_RELAXED_XY_TOL,
        z_tol=EEF_REACHED_RELAXED_Z_TOL,
    )
    object_to_eef_error_after = pose_error(
        eef_after, (object_after or {}).get("pose") if object_after else None
    )
    grip_alignment_ok = None
    if action.action == ACTION_GRIP and object_to_eef_error_after is not None:
        grip_alignment_ok = (
            object_to_eef_error_after["xy"] <= PRE_GRIP_OBJECT_XY_TOL
            and object_to_eef_error_after["z_abs"] <= PRE_GRIP_OBJECT_Z_TOL
        )
    return {
        "action_name": action.action,
        "robot_id": action.robot_id,
        "target_object_id": action.target_object_id,
        "target_pose": action.target_pose,
        "gripper_before": gripper_summary((robot_before or {}).get("joint_state")),
        "gripper_after": gripper_summary((robot_after or {}).get("joint_state")),
        "joint_motion_after": joint_motion_summary(
            (robot_after or {}).get("joint_state")
        ),
        "eef_before": compact_pose(eef_before),
        "eef_after": compact_pose(eef_after),
        "eef_to_target_pose_error_after": eef_error_after,
        "eef_to_target_xy_after": xy_distance(eef_after, action.target_pose),
        "eef_reach_assessment": {
            "standard_reached": eef_reached_standard,
            "relaxed_reached": eef_reached_relaxed,
            "standard_tolerance": {
                "xy": OBJECT_APPROACH_XY_TOL,
                "z": OBJECT_APPROACH_Z_TOL,
            },
            "relaxed_tolerance": {
                "xy": EEF_REACHED_RELAXED_XY_TOL,
                "z": EEF_REACHED_RELAXED_Z_TOL,
            },
            "guidance": (
                "If relaxed_reached is true and joint_motion_after does not show clear motion, "
                "do not keep the action pending solely because of small residual pose error."
            ),
        },
        "object_before": object_before,
        "object_after": object_after,
        "object_pose_delta": xyz_delta(
            (object_before or {}).get("pose") if object_before else None,
            (object_after or {}).get("pose") if object_after else None,
        ),
        "object_to_eef_xy_after": xy_distance(
            (object_after or {}).get("pose") if object_after else None, eef_after
        ),
        "object_to_eef_pose_error_after": object_to_eef_error_after,
        "grip_alignment_assessment": {
            "applies_to": "Grip",
            "ok": grip_alignment_ok,
            "required_before_calling_grip": {
                "xy_max": PRE_GRIP_OBJECT_XY_TOL,
                "z_abs_max": PRE_GRIP_OBJECT_Z_TOL,
            },
            "guidance": (
                "For a Grip action, ok=false means the EEF was still hovering above or offset from the object. "
                "A closed gripper is not enough to call the grasp successful when this alignment is false."
            ),
        },
        "diagnostic_hints": [
            "For a pre-grasp Moving action before any Grip, an open gripper is expected and must not be diagnosed as failed grasp.",
            "If a Moving action is immediately before Grip, do not mark it successful unless the EEF actually descended to the object pose within the relaxed Z tolerance.",
            "For a Grip action, the immediately preceding Moving must have descended to the object's grasp pose; hovering above the object is not enough.",
            "After Grip, a gripper that remains fully open may mean grasp failed or no contact was made.",
            "A closed gripper alone does not prove grasp success if the EEF was still above the object when Grip was commanded.",
            "After a post-grip Moving/lift, the target object should move with the end effector; if object_pose_delta is tiny, the grasp likely failed or slipped.",
            "If replanning and the gripper is closed without a secure object, Release should occur before descending toward an object.",
            "For goal placement, target_pose z should be above the goal surface marker, not equal to the physical marker z.",
            "If eef_reach_assessment.relaxed_reached is true, prefer success over pending unless task-level evidence contradicts it.",
        ],
    }


def validate_action_sequence_against_snapshot(
    actions: list[PrimitiveAction], snapshot: dict[str, Any]
) -> None:
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    object_map = {
        str(obj.get("id")): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("id")
    }
    goal_ids = {
        str(goal.get("id"))
        for goal in goals.values()
        if isinstance(goal, dict) and goal.get("id")
    }
    for index, action in enumerate(actions):
        object_id = action.target_object_id
        if object_id in goal_ids:
            raise ValueError(
                f"action #{index + 1} uses goal id {object_id!r} as target_object_id; "
                "target_object_id must remain the object id being manipulated"
            )
        if object_id and object_id not in object_map:
            raise ValueError(
                f"action #{index + 1} references unknown object id: {object_id!r}"
            )
        if action.action == ACTION_GRIP:
            if index == 0:
                raise ValueError(
                    "Grip must be preceded by a Moving descent to the object grasp pose"
                )
            previous_action = actions[index - 1]
            if previous_action.action != ACTION_MOVING:
                raise ValueError(
                    "Grip must immediately follow a Moving descent to the object grasp pose"
                )
            if previous_action.target_object_id != object_id:
                raise ValueError(
                    "Grip must follow a Moving action for the same target_object_id"
                )
            obj = object_map.get(str(object_id))
            object_pose = obj.get("pose") if isinstance(obj, dict) else None
            pre_grip_error = pose_error(previous_action.target_pose, object_pose)
            if pre_grip_error is None:
                raise ValueError(
                    "Grip predecessor Moving must have a target_pose matching the object pose"
                )
            if (
                pre_grip_error["xy"] > PRE_GRIP_OBJECT_XY_TOL
                or pre_grip_error["z_abs"] > PRE_GRIP_OBJECT_Z_TOL
            ):
                raise ValueError(
                    "Grip predecessor Moving appears to hover above or away from the object; "
                    "insert a Moving descent to the object's current pose/grasp height immediately before Grip"
                )
            next_moving = next(
                (
                    later_action
                    for later_action in actions[index + 1 :]
                    if later_action.action == ACTION_MOVING
                ),
                None,
            )
            if next_moving is not None:
                object_point = pose_position(object_pose)
                next_point = pose_position(next_moving.target_pose)
                if object_point is not None and next_point is not None:
                    lift_xy = xy_distance(next_moving.target_pose, object_pose)
                    min_lift_z = object_point["z"] + POST_GRIP_LIFT_MIN_Z_OFFSET
                    if (
                        lift_xy is not None
                        and lift_xy > POST_GRIP_LOCAL_XY_TOL
                        and next_point["z"] < min_lift_z
                    ):
                        raise ValueError(
                            "First Moving after Grip starts a meaningful XY transfer without enough Z lift; "
                            f"insert a lift Moving at the object/current XY with z >= {min_lift_z:.3f} "
                            f"before horizontal transfer, got z={next_point['z']:.3f}"
                        )
                    if (
                        lift_xy is not None
                        and lift_xy <= POST_GRIP_LOCAL_XY_TOL
                        and next_point["z"] < min_lift_z
                    ):
                        raise ValueError(
                            "First Moving after Grip must lift the carried object above table/goal guards before transfer; "
                            f"use z >= {min_lift_z:.3f}, got z={next_point['z']:.3f}"
                        )
        if action.action == ACTION_RELEASE:
            if index == 0:
                raise ValueError("Release must be preceded by a safe drop Moving")
            previous_action = actions[index - 1]
            if previous_action.action != ACTION_MOVING:
                raise ValueError("Release must immediately follow a safe drop Moving")
            containing_goal = goal_containing_pose(previous_action.target_pose, goals)
            if containing_goal is not None:
                goal_color, goal = containing_goal
                required_z = required_goal_drop_z(goal)
                release_point = pose_position(previous_action.target_pose)
                if (
                    required_z is not None
                    and release_point is not None
                    and release_point["z"] < required_z - GOAL_DROP_Z_TOL
                ):
                    raise ValueError(
                        f"Release before {goal_color}_goal is too low for the goal surface/rim and held object; "
                        f"use a drop pose at z >= {required_z:.3f}, not the physical marker z"
                    )
    if actions and actions[-1].action != ACTION_HOMING:
        raise ValueError("action sequence must end with Homing")
    for index, action in enumerate(actions):
        if action.action == ACTION_HOMING and index != len(actions) - 1:
            raise ValueError(
                "Homing may appear only as the final action in a task action sequence"
            )
        if (
            action.action == ACTION_HOMING
            and action.robot_id not in CONTROL_SERVICE_TOPICS
        ):
            raise ValueError("final Homing action must include a valid robot_id")
        if action.action == ACTION_HOMING and action.target_pose is not None:
            raise ValueError("final Homing action must use target_pose null")
        if action.action == ACTION_HOMING and action.target_object_id is not None:
            raise ValueError("final Homing action must use target_object_id null")


def infer_task_object_id(task: str, snapshot: dict[str, Any]) -> str | None:
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    object_ids = [
        str(obj.get("id"))
        for obj in objects
        if isinstance(obj, dict) and obj.get("id")
    ]
    matches = [object_id for object_id in object_ids if object_id in task]
    if len(matches) == 1:
        return matches[0]
    return None


def normalize_action_target_object_ids(
    actions: list[PrimitiveAction],
    task: str,
    snapshot: dict[str, Any],
) -> list[PrimitiveAction]:
    task_object_id = infer_task_object_id(task, snapshot)
    if task_object_id is None:
        return actions
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    goal_ids = {
        str(goal.get("id"))
        for goal in goals.values()
        if isinstance(goal, dict) and goal.get("id")
    }
    normalized: list[PrimitiveAction] = []
    for action in actions:
        target_object_id = action.target_object_id
        if action.action != ACTION_HOMING and target_object_id in goal_ids:
            target_object_id = task_object_id
        normalized.append(
            PrimitiveAction(
                robot_id=action.robot_id,
                action=action.action,
                target_pose=action.target_pose,
                target_object_id=target_object_id,
            )
        )
    return normalized


class AgentLogger:
    def __init__(self, agent_name: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{agent_name}.log"
        self.lock = threading.Lock()

    def write(self, event: str, payload: Any) -> None:
        record = {
            "time": time.time(),
            "event": event,
            "payload": payload,
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class Agent:
    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.context_dir = CONTEXT_DIR / name
        self.claude_path = self.context_dir / "CLAUDE.md"
        self.memory_path = self.context_dir / "MEMORY.md"
        self.logger = AgentLogger(name)
        self.session_log: list[dict[str, Any]] = []
        self.ensure_context_files()

    def ensure_context_files(self) -> None:
        self.context_dir.mkdir(parents=True, exist_ok=True)
        if not self.claude_path.exists():
            self.claude_path.write_text(
                f"# {self.name} Agent Context\n\n"
                "Use project ROS2 observations as the source of truth. Return JSON only when asked.\n",
                encoding="utf-8",
            )
        if not self.memory_path.exists():
            self.memory_path.write_text(
                f"# {self.name} Agent Memory\n\n" "- No persistent memories yet.\n",
                encoding="utf-8",
            )

    def context_text(self) -> str:
        return (
            "CLAUDE.md:\n"
            + self.claude_path.read_text(encoding="utf-8")
            + "\n\nMEMORY.md:\n"
            + self.memory_path.read_text(encoding="utf-8")
        )

    def append_memory(self, text: str) -> None:
        with self.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        self.logger.write("memory_append", text)

    def call_json(
        self,
        user_payload: dict[str, Any],
        image_bytes: bytes | None = None,
    ) -> AgentReply:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY or CHATGPT_API_KEY is required")
        user_text = json.dumps(
            {
                "agent_context": self.context_text(),
                "payload": user_payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        content: Any = [{"type": "text", "text": user_text}]
        if SEND_IMAGE_TO_AGENTS and image_bytes is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url(image_bytes)}}
            )

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        self.logger.write(
            "request",
            {
                "system": self.system_prompt,
                "user": user_payload,
                "has_image": image_bytes is not None,
            },
        )
        try:
            response = post_json(
                OPENAI_API_URL, payload, {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            )
            choices = response.get("choices") or []
            message = (
                choices[0].get("message")
                if choices and isinstance(choices[0], dict)
                else {}
            )
            raw_text = str((message or {}).get("content") or "")
            parsed = ensure_json_object(raw_text)
            self.session_log.append({"request": user_payload, "response": parsed})
            self.logger.write("response", {"raw": raw_text, "parsed": parsed})
            return AgentReply(parsed=parsed, raw_text=raw_text)
        except Exception as exc:
            self.logger.write("error", {"error": str(exc), "request": user_payload})
            raise


class TaskAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "task",
            (
                "You decompose a user's high-level robot command into natural-language subtasks. "
                'Return JSON only: {"tasks": [string, ...], "notes": string}. '
                "Tasks must be concrete object-level goals, not ROS primitive actions. "
                "Use exact object ids from world_summary.objects_by_id. "
                "Each task should name the responsible robot_id when the route is clear. "
                "When the user asks to sort red and blue objects, make one task per visible object, "
                "or multiple route-stage tasks when handover is needed. "
                "Prefer a direct single-robot task whenever world_summary.objects_by_id[*].direct_robot_candidates is non-empty. "
                "Nearest_robot differences alone are not a reason to use table_center_handover. "
                "Only split into buffer/handover subtasks when handover_needed is true, no direct robot candidate exists, "
                "or a previous direct attempt actually failed. "
                "Use world_summary.objects_by_id[*].route_hint as a cue, but still reason from the poses and image. "
                "A staged handover task should be explicit about who handles each stage and where the object should be left for the next robot. "
                "For example: 'left robot move cube_1:3 to table_center_handover for right robot handover', "
                "then 'right robot move cube_1:3 from table_center_handover to blue_goal:2'. "
                "Do not emit generic tasks such as 'Move the red object' when exact ids are available."
            ),
        )

    def decompose(
        self, command: str, snapshot: dict[str, Any], image_bytes: bytes | None
    ) -> list[str]:
        reply = self.call_json(
            {
                "user_command": command,
                "world_summary": build_agent_world_summary(snapshot),
                "task_rules": [
                    "Create one task for each object that still needs to be moved.",
                    "Each task must contain the exact object id, responsible robot_id if inferable, and destination.",
                    "Use red_goal for red objects and blue_goal for blue objects.",
                    "Use world_summary.objects_by_id[*].reach, goals[*].reach, and workspace_model to decide whether a direct single-robot task or a staged handover task is more appropriate.",
                    "If objects_by_id[*].preferred_direct_robot is present, create one direct task using that robot and the final color goal.",
                    "Do not split a task only because nearest_robot and destination_nearest_robot differ; direct_robot_candidates has priority.",
                    "When objects_by_id[*].handover_needed is true and direct_robot_candidates is empty, split into two tasks: pickup robot moves object to table_center_handover or another named shared location, then destination robot moves it from that location to the final goal.",
                    "For staged tasks, include the handover source/destination in natural language so the ActionAgent knows whether it is picking from the object's original pose or from a buffer.",
                    "The destination may be a goal, table_center_handover, or another named_locations entry when handover/buffer is needed.",
                    "Do not include primitive action names in tasks.",
                ],
                "required_output": {
                    "tasks": [
                        "<robot_id> robot move <object_id> to <goal_id|named_location>",
                        "<robot_id> robot move <object_id> from <named_location> to <goal_id>",
                    ]
                },
            },
            image_bytes=image_bytes,
        )
        tasks = reply.parsed.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("TaskAgent response must contain non-empty tasks list")
        return [str(task) for task in tasks]


class ActionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "action",
            (
                "You convert one natural-language subtask into directly executable ROS2 primitive actions. "
                'Return JSON only: {"actions": [{"robot_id": "left"|"right", '
                '"action": "Moving"|"Grip"|"Release"|"Homing", '
                '"target_pose": object|null, "target_object_id": string|null}]}. '
                "Do not include reason, metadata, intent, or extra fields. "
                "For moving an object to a goal, output the complete set of primitive actions you believe is needed for that task. "
                "If the task names a robot_id, use that robot_id unless feedback explicitly explains why it is impossible. "
                "If the task destination is a named location such as table_center_handover, treat that named location as the placement destination for this stage. "
                "If the task says 'from table_center_handover to blue_goal', pick the object's current observed pose after the previous stage and deliver it to the goal. "
                "A typical pick-and-place plan may include moving above the object, descending to the object's current grasp pose, gripping, lifting in Z, moving above the destination, descending to a safe drop pose, releasing, and homing. "
                "That example is not a mandatory fixed sequence; choose the sequence that best matches the current state and feedback. "
                "Do not return only one primitive when the task clearly requires multiple primitives. "
                "The first Moving before Grip must never target the goal pose. "
                "If you approach above an object, that is only an approach waypoint; you must add a second Moving that descends to the object's actual pose/grasp height immediately before Grip. "
                "Keep target_object_id equal to the object id for every object manipulation action, including Moving to the goal pose and Release. "
                "Goal ids such as red_goal:1 and blue_goal:2 are destinations encoded only by target_pose; they must never appear in target_object_id. "
                "Every task action sequence must end with Homing. Use the same robot_id, target_pose null, and target_object_id null for the final Homing action. "
                "Use the exact spelling 'Release'. "
                "If replanning with a closed gripper and no securely held object, open the gripper with Release before descending toward an object. "
                "For goal placement, approach from a high clearance Z first, then descend to a target pose above the goal marker surface, preferably world_summary.goals[color].eef_drop_pose, not physical_marker_pose. "
                "Never descend to world_summary.goals[color].physical_marker_pose for Release; that is the goal surface/rim marker and can collide with the held object. "
                "Remember the Moving service moves horizontally at the current EEF Z before vertical descent/ascent; if carrying an object across the table or near other objects, consider adding explicit lift and above-destination Moving waypoints. "
                "After Grip, treat the gripper and object as a single carried body; low XY transfer can drag or collide the object with the table, goal lip, guards, or nearby objects. "
                "When there is any meaningful XY transfer after Grip, include a lifted waypoint at current/object XY and an above-destination waypoint; omit these only for a tiny local adjustment with no obstacles. "
                "robot_id 'left' is the bottom robot in top-view, and robot_id 'right' is the top robot in top-view."
            ),
        )

    def make_actions(
        self,
        task: str,
        snapshot: dict[str, Any],
        image_bytes: bytes | None,
        feedback: str = "",
    ) -> list[PrimitiveAction]:
        reply = self.call_json(
            {
                "task": task,
                "world_summary": build_agent_world_summary(snapshot),
                "feedback": feedback,
                "allowed_actions": sorted(ALLOWED_ACTIONS),
                "action_rules": [
                    "If the task names an object id, use that exact id.",
                    "If the task names a responsible robot_id such as 'left robot' or 'right robot', use that robot_id for the whole primitive sequence unless feedback says to reassign.",
                    "If the task destination is world_summary.named_locations.<name>, use that named location pose as the stage drop target rather than the final color goal.",
                    "If the task says the object is coming from a named handover location, still use the object's current observed marker pose as the pre-grasp Moving target.",
                    "A high Moving above the object is allowed as an approach waypoint, but it is not a grasp pose.",
                    "If you include an above-object approach, the next Moving before Grip must be a distinct lower descent to the object pose; do not duplicate the same hover/approach pose and then Grip.",
                    "The immediate action before Grip must be a Moving descent whose target_pose matches that object's current pose/grasp height from world_summary.objects_by_id.",
                    "Do not call Grip while the EEF is still at clearance Z above the object.",
                    "Do not move to the goal before Grip.",
                    "After Grip, do not go directly to the final low drop pose. First add a Moving waypoint at current/object XY with clearance Z, then a Moving waypoint above the destination at clearance Z, then descend to the drop pose.",
                    "For goal placement, goal guards and already placed objects make Z clearance important. Approach goals from above using clearance Z before descending.",
                    "After Grip, the final drop Moving target_pose should be inside the destination goal and at or above world_summary.goals[color].minimum_safe_release_z.",
                    "Use world_summary.goals[color].eef_drop_pose as the default goal drop EEF target. Never use physical_marker_pose as an EEF target or Release predecessor.",
                    "Moving service behavior: horizontal XY at current EEF Z first, then vertical Z at target XY. A far Moving command from a low grasp pose can sweep low and collide.",
                    "After Grip, the carried object effectively extends the gripper geometry. For meaningful XY transfer, include an explicit lift Moving at the current/object XY before horizontal transfer, then a Moving above the destination, then descend/place.",
                    "If you omit a lifted waypoint after Grip, it must be because the motion is tiny/local and the current EEF Z is already clearly above obstacles and object height.",
                    "Use world_summary.motion_model.suggested_transfer_clearance_z as guidance when choosing lifted/clearance poses; this is not mandatory if the current state makes another route better.",
                    "Return enough actions to accomplish the current task; do not assume the runtime will ask you for the missing primitives of the same task.",
                    "The final primitive must be Homing with target_pose null and target_object_id null.",
                    "If feedback indicates failed grasp/slip/retry and the selected robot gripper is closed or partially closed, Release before Moving down to an object.",
                    "Do not use a goal marker id as target_object_id.",
                    "Even when target_pose is a goal or named destination pose, target_object_id must remain the manipulated object id from the task.",
                    "Use 'Release' exactly for opening the gripper.",
                    "target_pose orientation may be omitted; the runtime enforces the vertical end-effector orientation.",
                ],
                "required_schema": {
                    "actions": [
                        {
                            "robot_id": "left|right",
                            "action": "Moving|Grip|Release|Homing",
                            "target_pose": "pose object or null",
                            "target_object_id": "string or null",
                        }
                    ]
                },
            },
            image_bytes=image_bytes,
        )
        raw_actions = reply.parsed.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("ActionAgent response must contain non-empty actions list")
        actions = [parse_primitive_action(item) for item in raw_actions]
        actions = normalize_action_target_object_ids(actions, task, snapshot)
        validate_action_sequence_against_snapshot(actions, snapshot)
        return actions


class TroubleshooterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "troubleshooter",
            (
                "You diagnose robot primitive execution results using before/after ROS2 snapshots. "
                "You are called after each primitive in a multi-action sequence. "
                "Diagnose the primitive in the context of the current natural-language task and the full planned action sequence; "
                "do not judge only whether the service call succeeded. "
                "A primitive may be locally successful but task-inconsistent, or locally queued but still pending. "
                "If sequence_context.is_final_action is false, never return complete; return success to continue, "
                "pending if more observation time is needed, or retry/replan only if the current primitive failed or clearly made the state worse. "
                "For non-final primitives, success means the current primitive made the expected progress for the task stage and the remaining actions are still sensible. "
                "For pre-grasp Moving before any Grip, success means the end effector reached the object in XY and Z while the gripper remains open; an open gripper is expected and is not a failed grasp. "
                "If the next planned primitive is Grip and execution_diagnostics.eef_reach_assessment.relaxed_reached is false, do not return success for the Moving action. "
                "Only diagnose failed grasp after a Grip action or after a post-grip lift/transfer where the object should have followed the EEF. "
                "For Grip, verify execution_diagnostics.grip_alignment_assessment.ok; a closed gripper above the object is not a successful grasp. "
                "For post-grip Moving or transfer, verify whether the target object moved consistently with the EEF and whether the motion supports the task destination or handover stage. "
                "A ControlCommand service success only means the motion was queued; use execution_diagnostics and action_result.message to decide whether the EEF actually reached the target. "
                "A ControlCommand failure/rejection from world.py is not success and should lead to retry/replan with safer reachable poses. "
                "For Release near a goal, reject or replan if the preceding Moving target was the physical marker z instead of minimum_safe_release_z/eef_drop_pose. "
                "Use execution_diagnostics.eef_reach_assessment: if relaxed_reached is true and joint_motion_after does not show clear motion, do not keep returning pending for small residual EEF error. "
                "If a Moving action still appears in progress and relaxed_reached is false, return pending rather than retry. "
                'Return JSON only: {"status": "success"|"pending"|"retry"|"replan_task"|'
                '"replan_all"|"emergency_recover"|"complete", "message": string, '
                '"memory_update": string|null}.'
            ),
        )

    def diagnose(
        self,
        task: str,
        result: ActionResult,
        snapshot_after: dict[str, Any],
        image_bytes: bytes | None,
        action_index: int,
        total_actions: int,
        previous_actions: list[dict[str, Any]],
        remaining_actions: list[dict[str, Any]],
        pending_check_count: int = 0,
    ) -> dict[str, Any]:
        reply = self.call_json(
            {
                "task": task,
                "action_result": result.as_dict(),
                "snapshot_before": result.snapshot_before,
                "snapshot_after": snapshot_after,
                "world_summary_after": build_agent_world_summary(snapshot_after),
                "execution_diagnostics": build_execution_diagnostics(
                    result, snapshot_after
                ),
                "task_context": {
                    "natural_language_task": task,
                    "current_primitive": result.action.as_dict(),
                    "previous_actions": previous_actions,
                    "current_and_remaining_sequence": [
                        result.action.as_dict(),
                        *remaining_actions,
                    ],
                    "full_planned_sequence": [
                        *previous_actions,
                        result.action.as_dict(),
                        *remaining_actions,
                    ],
                    "diagnosis_goal": (
                        "Judge whether the current primitive made the right progress for this task stage, "
                        "whether the remaining planned actions still make sense, and whether retry/replanning is needed. "
                        "Do not mark task complete from a locally successful intermediate primitive."
                    ),
                },
                "sequence_context": {
                    "action_index": action_index,
                    "total_actions": total_actions,
                    "is_final_action": action_index >= total_actions,
                    "remaining_actions": remaining_actions,
                    "pending_check_count": pending_check_count,
                    "rule": (
                        "Do not return complete before the final action. "
                        "Intermediate successful primitives should return success so runtime continues the sequence. "
                        "Return pending only when the action may still be settling and the relaxed EEF tolerance has not been reached."
                    ),
                },
                "allowed_status": sorted(TROUBLESHOOTER_STATUSES),
            },
            image_bytes=image_bytes,
        )
        status = str(reply.parsed.get("status") or "").strip()
        if status not in TROUBLESHOOTER_STATUSES:
            raise ValueError(f"TroubleshooterAgent returned invalid status: {status!r}")
        return reply.parsed


class MainAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "main",
            (
                "You coordinate Task, Action, DB, and Troubleshooter agents. "
                "You decide whether the entire user command is complete or whether to continue/retry/replan. "
                "Do not treat a primitive success report as task completion unless the current world summary shows the task-level object/goal state is satisfied. "
                "Use Troubleshooter reports as evidence, but verify against object poses, goals, gripper states, and the remaining task list. "
                "Return JSON only when asked."
            ),
        )

    def decide_completion(
        self,
        command: str,
        tasks: list[str],
        completed_tasks: list[str],
        snapshot: dict[str, Any],
        troubleshooter_reports: list[dict[str, Any]],
        image_bytes: bytes | None,
    ) -> dict[str, Any]:
        reply = self.call_json(
            {
                "user_command": command,
                "tasks": tasks,
                "completed_tasks": completed_tasks,
                "world_snapshot": snapshot,
                "world_summary": build_agent_world_summary(snapshot),
                "troubleshooter_reports": troubleshooter_reports[-10:],
                "required_output": {
                    "complete": "boolean",
                    "next_task_index": "integer or null",
                    "decision": "continue|retry|replan_task|replan_all|emergency_recover|complete",
                    "message": "string",
                    "memory_update": "string|null",
                },
            },
            image_bytes=image_bytes,
        )
        return reply.parsed


class MemoryAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "memory",
            (
                "You decide whether agent MEMORY.md files should be updated. "
                'Return JSON only: {"updates": [{"agent": string, "text": string}]}'
            ),
        )

    def propose_updates(self, event: dict[str, Any]) -> list[dict[str, str]]:
        reply = self.call_json(
            {
                "event": event,
                "agents": ["main", "task", "action", "db", "troubleshooter"],
            }
        )
        updates = reply.parsed.get("updates")
        if not isinstance(updates, list):
            return []
        normalized: list[dict[str, str]] = []
        for update in updates:
            if not isinstance(update, dict):
                continue
            agent = str(update.get("agent") or "").strip()
            text = str(update.get("text") or "").strip()
            if agent and text:
                normalized.append({"agent": agent, "text": text})
        return normalized


class DBRosNode(Node):
    def __init__(self) -> None:
        super().__init__("multi_agent_chatgpt_db")
        self._lock = threading.Lock()
        self._table: dict[str, Any] | None = None
        self._goals: dict[str, dict[str, Any]] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._latest_image: Any | None = None
        self._latest_image_time = 0.0
        self._camera_pose: dict[str, Any] | None = None
        self._robot_poses: dict[str, dict[str, Any]] = {}
        self._eef_poses: dict[str, dict[str, Any]] = {}
        self._joint_states: dict[str, dict[str, Any]] = {}

        self.create_subscription(MarkerArray, MARKER_TOPIC, self._on_markers, 10)
        self.create_subscription(RosImage, IMAGE_TOPIC, self._on_image, 2)
        self.create_subscription(
            PoseStamped, CAMERA_POSE_TOPIC, self._on_camera_pose, 10
        )
        for robot_id, topic in ROBOT_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, rid=robot_id: self._on_robot_pose(rid, msg),
                10,
            )
        for robot_id, topic in EEF_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, rid=robot_id: self._on_eef_pose(rid, msg),
                10,
            )
        for robot_id, topic in JOINT_STATE_TOPICS.items():
            self.create_subscription(
                JointState,
                topic,
                lambda msg, rid=robot_id: self._on_joint_state(rid, msg),
                10,
            )

        self.control_clients = {
            robot_id: self.create_client(ControlCommand, topic)
            for robot_id, topic in CONTROL_SERVICE_TOPICS.items()
        }

    def _on_markers(self, message: Any) -> None:
        table = None
        goals: dict[str, dict[str, Any]] = {}
        objects: dict[str, dict[str, Any]] = {}
        for marker in message.markers:
            color = color_to_dict(marker.color)
            entry = {
                "id": f"{marker.ns}:{marker.id}",
                "namespace": str(marker.ns),
                "marker_id": int(marker.id),
                "shape": shape_from_namespace(str(marker.ns)),
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

    def _on_joint_state(self, robot_id: str, message: Any) -> None:
        with self._lock:
            self._joint_states[robot_id] = {
                "names": [str(name) for name in message.name],
                "positions": [float(value) for value in message.position],
                "velocities": [float(value) for value in message.velocity],
                "efforts": [float(value) for value in message.effort],
                "timestamp_unix": time.time(),
            }

    def readiness(self) -> dict[str, bool]:
        with self._lock:
            return {
                "markers": self._table is not None
                and bool(self._objects)
                and bool(self._goals),
                "image": self._latest_image is not None,
                "camera_pose": self._camera_pose is not None,
                "robot_poses": all(
                    robot_id in self._robot_poses for robot_id in ROBOT_POSE_TOPICS
                ),
                "eef_poses": all(
                    robot_id in self._eef_poses for robot_id in EEF_POSE_TOPICS
                ),
                "joint_states": all(
                    robot_id in self._joint_states for robot_id in JOINT_STATE_TOPICS
                ),
            }

    def snapshot(
        self, include_image: bool = True
    ) -> tuple[dict[str, Any], bytes | None]:
        with self._lock:
            latest_image = self._latest_image
            image_age = (
                time.monotonic() - self._latest_image_time
                if self._latest_image_time
                else None
            )
            snapshot = {
                "timestamp_unix": time.time(),
                "topics": {
                    "markers": MARKER_TOPIC,
                    "image": IMAGE_TOPIC,
                    "camera_pose": CAMERA_POSE_TOPIC,
                    "robot_poses": ROBOT_POSE_TOPICS,
                    "end_effector_poses": EEF_POSE_TOPICS,
                    "joint_states": JOINT_STATE_TOPICS,
                },
                "camera": {
                    "pose": self._camera_pose,
                    "image_age_sec": image_age,
                    "image_size": (
                        {
                            "width": int(latest_image.width),
                            "height": int(latest_image.height),
                        }
                        if latest_image is not None
                        else None
                    ),
                },
                "table": self._table or self._default_table(),
                "goals": dict(self._goals),
                "objects": sorted(self._objects.values(), key=lambda item: item["id"]),
                "robots": {
                    robot_id: {
                        "scene_label": ROBOT_SCENE_LABELS[robot_id],
                        "base_pose": self._robot_poses.get(robot_id),
                        "end_effector_pose": self._eef_poses.get(robot_id),
                        "joint_state": self._joint_states.get(robot_id),
                    }
                    for robot_id in ("left", "right")
                },
            }
        image_bytes = (
            image_msg_to_png_bytes(latest_image)
            if include_image and latest_image is not None
            else None
        )
        return snapshot, image_bytes

    @staticmethod
    def _default_table() -> dict[str, Any]:
        return {
            "id": "table:0",
            "namespace": "table",
            "shape": "cube",
            "color": "red",
            "pose": {
                "position": DEFAULT_TABLE_CENTER,
                "orientation": VERTICAL_EEF_ORIENTATION,
            },
            "scale": DEFAULT_TABLE_SIZE,
        }

    def call_control_service(self, action: PrimitiveAction) -> tuple[bool, str]:
        robot_id = str(action.robot_id)
        client = self.control_clients[robot_id]
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            return False, f"Service unavailable: {CONTROL_SERVICE_TOPICS[robot_id]}"
        request = ControlCommand.Request()
        request.action = action.action
        request.target_pose = ros_pose_from_dict(action.target_pose)
        future = client.call_async(request)
        deadline = time.monotonic() + SERVICE_CALL_TIMEOUT_SEC
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return (
                False,
                f"{action.action} timed out after {SERVICE_CALL_TIMEOUT_SEC:.1f}s",
            )
        try:
            response = future.result()
        except Exception as exc:
            return False, f"{action.action} failed: {exc}"
        return bool(response.success), str(response.message)


class DBAgent(Agent):
    def __init__(self, node: DBRosNode) -> None:
        super().__init__(
            "db",
            (
                "You answer questions about the latest ROS2 world snapshot. "
                "Use supplied marker, pose, image, and joint-state data. Return JSON only."
            ),
        )
        self.node = node
        self.spin_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        if self.spin_thread is not None:
            return
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def _spin_loop(self) -> None:
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.05)
            except Exception as exc:
                if exc.__class__.__name__ == "ExternalShutdownException":
                    break
                raise

    def stop(self) -> None:
        self.stop_event.set()
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=1.0)

    def snapshot(self) -> tuple[dict[str, Any], bytes | None]:
        return self.node.snapshot(include_image=True)

    def answer_query(
        self,
        query: str,
        image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        snapshot, current_image = self.snapshot()
        reply = self.call_json(
            {
                "query": query,
                "world_snapshot": snapshot,
                "required_output": {
                    "answer": "any JSON value",
                    "confidence": "0..1",
                    "notes": "string",
                },
            },
            image_bytes=image_bytes if image_bytes is not None else current_image,
        )
        return reply.parsed


def parse_primitive_action(value: Any) -> PrimitiveAction:
    if not isinstance(value, dict):
        raise ValueError(
            f"primitive action must be an object, got {type(value).__name__}"
        )
    allowed_keys = {"robot_id", "action", "target_pose", "target_object_id"}
    extra_keys = set(value) - allowed_keys
    if extra_keys:
        raise ValueError(
            f"primitive action has unsupported fields: {sorted(extra_keys)}"
        )
    action_name = str(value.get("action") or "").strip()
    if action_name in {
        "Release",
        "release",
        "Releasing",
        "releasing",
        "Realease",
        "realease",
    }:
        action_name = ACTION_RELEASE
    if action_name not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported primitive action: {action_name!r}")
    robot_id = value.get("robot_id")
    if action_name != ACTION_HOMING and robot_id not in CONTROL_SERVICE_TOPICS:
        raise ValueError(f"invalid robot_id: {robot_id!r}")
    if robot_id is not None and robot_id not in CONTROL_SERVICE_TOPICS:
        raise ValueError(f"invalid robot_id: {robot_id!r}")
    target_pose = value.get("target_pose")
    if target_pose is not None and not isinstance(target_pose, dict):
        raise ValueError("target_pose must be object or null")
    target_object_id = value.get("target_object_id")
    return PrimitiveAction(
        robot_id=str(robot_id) if robot_id is not None else None,
        action=action_name,
        target_pose=target_pose,
        target_object_id=(
            str(target_object_id) if target_object_id is not None else None
        ),
    )


class ActionExecutor:
    def __init__(self, db_node: DBRosNode, reporter: Reporter) -> None:
        self.db_node = db_node
        self.reporter = reporter

    def execute(self, action: PrimitiveAction) -> ActionResult:
        before, _ = self.db_node.snapshot(include_image=False)
        if action.robot_id not in CONTROL_SERVICE_TOPICS:
            return ActionResult(
                False, f"Invalid robot_id: {action.robot_id}", action, before, before
            )
        success, message = self.db_node.call_control_service(action)
        self._action_interval(action.action)
        after, _ = self.db_node.snapshot(include_image=False)
        return ActionResult(success, message, action, before, after)

    @staticmethod
    def _action_interval(action_name: str) -> None:
        settle_sec = {
            ACTION_MOVING: MOVING_SETTLE_SEC,
            ACTION_GRIP: GRIP_SETTLE_SEC,
            ACTION_RELEASE: RELEASE_SETTLE_SEC,
            ACTION_HOMING: HOMING_SETTLE_SEC,
        }.get(action_name, ACTION_INTERVAL_SEC)
        deadline = time.monotonic() + settle_sec
        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(0.05)


def wait_for_ready(db_node: DBRosNode, reporter: Reporter) -> None:
    last_report = 0.0
    while rclpy.ok():
        readiness = db_node.readiness()
        if all(readiness.values()):
            return
        now = time.monotonic()
        if now - last_report >= READINESS_REPORT_SEC:
            missing = [key for key, ready in readiness.items() if not ready]
            reporter.status(
                "Waiting for ROS2 observations", {"missing": ", ".join(missing)}
            )
            last_report = now
        time.sleep(0.1)


def ask_user_command() -> str:
    if RICH_AVAILABLE and Prompt is not None:
        text = Prompt.ask(
            "Command", default=DEFAULT_COMMAND if USE_DEFAULT_ON_EMPTY_INPUT else ""
        )
    else:
        text = input(f"Command [{DEFAULT_COMMAND}]: ").strip()
    if not text and USE_DEFAULT_ON_EMPTY_INPUT:
        return DEFAULT_COMMAND
    return text


def apply_memory_updates(
    memory_agent: MemoryAgent, agents: dict[str, Agent], event: dict[str, Any]
) -> None:
    try:
        updates = memory_agent.propose_updates(event)
    except Exception as exc:
        memory_agent.logger.write("memory_update_failed", str(exc))
        return
    for update in updates:
        agent = agents.get(update["agent"])
        if agent is not None:
            agent.append_memory(update["text"])


def run() -> int:
    if ROS_IMPORT_ERROR is not None:
        print(
            "ROS2 imports failed. Source the ROS2 workspace before running this CLI, for example:\n"
            "  source install/setup.bash && /usr/bin/python3 web-app/cli_multi_agent_chatgpt.py",
            file=sys.stderr,
        )
        print(f"Import error: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY or CHATGPT_API_KEY is required.", file=sys.stderr)
        return 2

    reporter = Reporter()
    reporter.status(
        "Startup",
        {
            "cli": "multi_agent_chatgpt",
            "model": OPENAI_MODEL,
            "action_interval_sec": ACTION_INTERVAL_SEC,
            "max_steps": MAX_MAIN_STEPS,
        },
    )

    rclpy.init()
    db_node = DBRosNode()
    db_agent = DBAgent(db_node)
    main_agent = MainAgent()
    task_agent = TaskAgent()
    action_agent = ActionAgent()
    troubleshooter = TroubleshooterAgent()
    memory_agent = MemoryAgent()
    agents = {
        "main": main_agent,
        "task": task_agent,
        "action": action_agent,
        "db": db_agent,
        "troubleshooter": troubleshooter,
        "memory": memory_agent,
    }
    executor = ActionExecutor(db_node, reporter)

    try:
        db_agent.start()
        wait_for_ready(db_node, reporter)
        command = ask_user_command()
        snapshot, image_bytes = db_agent.snapshot()
        tasks = task_agent.decompose(command, snapshot, image_bytes)
        completed_tasks: list[str] = []
        trouble_reports: list[dict[str, Any]] = []
        task_index = 0
        replan_attempts: dict[int, int] = {}

        for step in range(1, MAX_MAIN_STEPS + 1):
            snapshot, image_bytes = db_agent.snapshot()
            completion = main_agent.decide_completion(
                command,
                tasks,
                completed_tasks,
                snapshot,
                trouble_reports,
                image_bytes,
            )
            if (
                bool(completion.get("complete"))
                or completion.get("decision") == "complete"
            ):
                reporter.status(
                    "Complete",
                    {"message": completion.get("message", "MainAgent marked complete")},
                )
                return 0

            if isinstance(completion.get("next_task_index"), int):
                task_index = max(
                    0, min(int(completion["next_task_index"]), len(tasks) - 1)
                )
            if task_index >= len(tasks):
                task_index = max(0, len(tasks) - 1)
            task = tasks[task_index]
            feedback = trouble_reports[-1].get("message", "") if trouble_reports else ""
            actions: list[PrimitiveAction] | None = None
            action_feedback = feedback
            for plan_attempt in range(1, MAX_REPLAN_ATTEMPTS_PER_TASK + 1):
                try:
                    actions = action_agent.make_actions(
                        task, snapshot, image_bytes, feedback=action_feedback
                    )
                    break
                except Exception as exc:
                    action_feedback = (
                        f"Previous action plan was rejected before execution: {exc}. "
                        "Return a corrected plan with enough primitives to accomplish the task. "
                        "For object-moving tasks, do not return only one primitive if grasping, moving, releasing, or recovery is still required. "
                        "Use explicit Z lift after Grip and never Release at a goal physical_marker_pose; use eef_drop_pose/minimum_safe_release_z."
                    )
                    trouble_reports.append(
                        {
                            "status": "retry",
                            "message": action_feedback,
                            "source": "runtime_action_plan_validation",
                            "task": task,
                        }
                    )
                    reporter.status(
                        "Action plan rejected",
                        {"task": task, "attempt": plan_attempt, "error": str(exc)},
                    )
            if actions is None:
                reporter.error(
                    f"ActionAgent could not produce a valid plan for task: {task}"
                )
                continue
            reporter.status(
                "Task",
                {
                    "step": step,
                    "task_index": task_index,
                    "task": task,
                    "actions": len(actions),
                },
            )

            task_done = False
            for action_index, action in enumerate(actions, start=1):
                reporter.status("Action", action.as_dict())
                result = executor.execute(action)
                snapshot_after, image_after = db_agent.snapshot()
                if not result.success:
                    diagnosis = {
                        "status": "replan_task",
                        "message": (
                            "ControlCommand was rejected or failed in world.py; "
                            f"the current action sequence is not executable as planned. {result.message}"
                        ),
                        "memory_update": (
                            "When world.py rejects an action, treat it as task/action failure and replan with safer reachable poses."
                        ),
                    }
                else:
                    diagnosis = troubleshooter.diagnose(
                        task,
                        result,
                        snapshot_after,
                        image_after,
                        action_index=action_index,
                        total_actions=len(actions),
                        previous_actions=[
                            item.as_dict() for item in actions[: action_index - 1]
                        ],
                        remaining_actions=[
                            item.as_dict() for item in actions[action_index:]
                        ],
                    )
                pending_checks = 0
                while (
                    diagnosis.get("status") == "pending"
                    and pending_checks < MAX_PENDING_DIAGNOSES
                ):
                    pending_checks += 1
                    reporter.status(
                        "Troubleshooter pending",
                        {
                            "check": pending_checks,
                            "max": MAX_PENDING_DIAGNOSES,
                            "message": diagnosis.get("message", ""),
                        },
                    )
                    time.sleep(PENDING_OBSERVATION_SEC)
                    snapshot_after, image_after = db_agent.snapshot()
                    diagnosis = troubleshooter.diagnose(
                        task,
                        result,
                        snapshot_after,
                        image_after,
                        action_index=action_index,
                        total_actions=len(actions),
                        previous_actions=[
                            item.as_dict() for item in actions[: action_index - 1]
                        ],
                        remaining_actions=[
                            item.as_dict() for item in actions[action_index:]
                        ],
                        pending_check_count=pending_checks,
                    )
                if diagnosis.get("status") == "pending":
                    diagnosis = {
                        **diagnosis,
                        "status": "replan_task",
                        "message": (
                            "Troubleshooter kept the action pending until the observation limit was reached; "
                            "runtime requests task replanning. "
                            + str(diagnosis.get("message", ""))
                        ),
                    }
                if diagnosis.get("status") == "complete" and action_index < len(
                    actions
                ):
                    diagnosis = {
                        **diagnosis,
                        "status": "success",
                        "message": (
                            "Troubleshooter returned complete for an intermediate primitive; "
                            "runtime downgraded it to success and will continue the planned action sequence. "
                            + str(diagnosis.get("message", ""))
                        ),
                    }
                trouble_reports.append(diagnosis)
                reporter.status("Troubleshooter", diagnosis)

                memory_event = {
                    "task": task,
                    "action_result": result.as_dict(),
                    "diagnosis": diagnosis,
                    "recent_reports": trouble_reports[-5:],
                }
                apply_memory_updates(memory_agent, agents, memory_event)

                status = diagnosis.get("status")
                if status == "success":
                    continue
                if status == "complete":
                    task_done = True
                    break
                if status == "retry":
                    replan_attempts[task_index] = replan_attempts.get(task_index, 0) + 1
                    if replan_attempts[task_index] > MAX_REPLAN_ATTEMPTS_PER_TASK:
                        task_done = False
                        break
                    break
                if status in {"replan_task", "replan_all", "emergency_recover"}:
                    break

            latest_status = (
                trouble_reports[-1].get("status") if trouble_reports else "success"
            )
            if task_done or latest_status in {"success", "complete"}:
                if task not in completed_tasks:
                    completed_tasks.append(task)
                task_index += 1
                continue
            if latest_status == "replan_all":
                snapshot, image_bytes = db_agent.snapshot()
                tasks = task_agent.decompose(command, snapshot, image_bytes)
                completed_tasks = []
                task_index = 0
                continue
            if latest_status == "emergency_recover":
                for robot_id in ("left", "right"):
                    executor.execute(
                        PrimitiveAction(robot_id=robot_id, action=ACTION_RELEASE)
                    )
                    executor.execute(
                        PrimitiveAction(robot_id=robot_id, action=ACTION_HOMING)
                    )
                continue

        reporter.error(f"Stopped after MAX_MAIN_STEPS={MAX_MAIN_STEPS}")
        return 1
    finally:
        db_agent.stop()
        db_node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
