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
ACTION_CENTERING = "Centering"
ACTION_PLACING = "Placing"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Release"
ACTION_HOMING = "Homing"
MOVEMENT_ACTIONS = {ACTION_MOVING, ACTION_CENTERING, ACTION_PLACING}
ALLOWED_ACTIONS = {
    ACTION_MOVING,
    ACTION_CENTERING,
    ACTION_PLACING,
    ACTION_GRIP,
    ACTION_RELEASE,
    ACTION_HOMING,
}

VERTICAL_EEF_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
DEFAULT_TABLE_CENTER = {"x": 0.6, "y": 0.0, "z": 0.46}
DEFAULT_TABLE_SIZE = {"x": 0.9, "y": 0.7, "z": 0.08}
CONTROL_TABLE_CENTER_TARGET = {"x": 0.6, "y": 0.0, "z": 0.46}
CONTROL_GOAL_TARGETS = {
    "red": {"x": 0.272, "y": 0.228, "z": 0.466},
    "blue": {"x": 0.928, "y": -0.228, "z": 0.466},
}
DROP_SLOT_STEP_M = 0.032
DROP_SLOT_OFFSETS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (0.0, 0.0),
)
DROP_SLOT_TABLE_INSET_M = 0.04

SERVICE_WAIT_TIMEOUT_SEC = 1.0
SERVICE_CALL_TIMEOUT_SEC = 20.0
ACTION_INTERVAL_SEC = 1.0
MOVING_SETTLE_SEC = 1.0
GRIP_SETTLE_SEC = 2.2
RELEASE_SETTLE_SEC = 1.2
PLACING_SETTLE_SEC = 2.5
HOMING_SETTLE_SEC = 1.0
PRE_GRIP_SETTLE_TIMEOUT_SEC = 8.0
PRE_GRIP_ARM_VELOCITY_TOL = 0.05
GRIPPER_OPEN_POSITION_MIN = 0.033
GRIPPER_EMPTY_CLOSED_POSITION_MAX = 0.012
GRIP_CONTACT_MIN_CLOSING_DELTA = 0.001
GRIP_SETTLED_FINGER_VELOCITY_TOL = 0.02
GRIP_ACTIVE_FINGER_VELOCITY_TOL = 0.05
GRIP_EEF_STATIONARY_XY_TOL = 0.006
GRIP_EEF_STATIONARY_Z_TOL = 0.006
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
MOVEMENT_NO_PROGRESS_XY_TOL = 0.01
MOVEMENT_NO_PROGRESS_Z_TOL = 0.006
MOVEMENT_CLEAR_MISS_XY_TOL = 0.12
CARRIED_OBJECT_TARGET_XY_TOL = 0.05
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


def stable_slot_index(identifier: str, slot_count: int) -> int:
    if slot_count <= 0:
        return 0
    return sum(ord(char) for char in identifier) % slot_count


def clamp_table_drop_position(position: dict[str, float]) -> dict[str, float]:
    table_x = DEFAULT_TABLE_CENTER["x"]
    table_y = DEFAULT_TABLE_CENTER["y"]
    half_x = DEFAULT_TABLE_SIZE["x"] / 2.0
    half_y = DEFAULT_TABLE_SIZE["y"] / 2.0
    x_min = table_x - half_x + DROP_SLOT_TABLE_INSET_M
    x_max = table_x + half_x - DROP_SLOT_TABLE_INSET_M
    y_min = table_y - half_y + DROP_SLOT_TABLE_INSET_M
    y_max = table_y + half_y - DROP_SLOT_TABLE_INSET_M
    return position_dict(
        min(max(float(position["x"]), x_min), x_max),
        min(max(float(position["y"]), y_min), y_max),
        float(position["z"]),
    )


def drop_slot_poses(base_position: dict[str, float]) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    for dx, dy in DROP_SLOT_OFFSETS:
        slot_position = {
            "x": float(base_position["x"]) + dx * DROP_SLOT_STEP_M,
            "y": float(base_position["y"]) + dy * DROP_SLOT_STEP_M,
            "z": float(base_position["z"]),
        }
        poses.append(pose_with_position(clamp_table_drop_position(slot_position)))
    return poses


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


def control_pose_from_pose(pose: dict[str, Any] | None) -> dict[str, Any] | None:
    compact = compact_pose(pose)
    if compact is None:
        return None
    return pose_with_position(compact["position"])


def gripper_summary(joint_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(joint_state, dict):
        return {
            "state": "unknown",
            "finger_positions": [],
            "mean_finger_position": None,
            "mean_finger_velocity": None,
        }
    names = (
        joint_state.get("names") if isinstance(joint_state.get("names"), list) else []
    )
    positions = (
        joint_state.get("positions")
        if isinstance(joint_state.get("positions"), list)
        else []
    )
    velocities = (
        joint_state.get("velocities")
        if isinstance(joint_state.get("velocities"), list)
        else []
    )
    finger_positions: list[float] = []
    finger_velocities: list[float] = []
    for name, position, velocity in zip(names, positions, velocities):
        if "finger" in str(name):
            try:
                finger_positions.append(float(position))
                finger_velocities.append(abs(float(velocity)))
            except (TypeError, ValueError):
                pass
    if not finger_positions:
        return {
            "state": "unknown",
            "finger_positions": [],
            "mean_finger_position": None,
            "mean_finger_velocity": None,
        }
    mean_position = sum(finger_positions) / len(finger_positions)
    mean_velocity = (
        sum(finger_velocities) / len(finger_velocities) if finger_velocities else None
    )
    if mean_position >= GRIPPER_OPEN_POSITION_MIN:
        state = "open"
    elif mean_position <= GRIPPER_EMPTY_CLOSED_POSITION_MAX:
        state = "closed"
    else:
        state = "partially_closed_or_holding"
    return {
        "state": state,
        "finger_positions": finger_positions,
        "mean_finger_position": mean_position,
        "mean_finger_velocity": mean_velocity,
        "position_reference": {
            "fully_open_approx": 0.04,
            "empty_fully_closed_approx": 0.0,
            "open_min": GRIPPER_OPEN_POSITION_MIN,
            "empty_closed_max": GRIPPER_EMPTY_CLOSED_POSITION_MAX,
        },
        "interpretation": "larger values are more open; smaller values are more closed",
        "grasp_note": (
            "partially_closed_or_holding is a normal successful grasp state when an object prevents full closure"
        ),
    }


def robot_gripper_state(snapshot: dict[str, Any], robot_id: str | None) -> str:
    if not robot_id:
        return "unknown"
    robot = robot_by_id(snapshot, robot_id)
    return str(gripper_summary((robot or {}).get("joint_state")).get("state"))


def robot_appears_to_hold_object(
    snapshot: dict[str, Any], robot_id: str | None, object_id: str | None
) -> bool:
    robot = robot_by_id(snapshot, robot_id)
    obj = object_by_id(snapshot, object_id)
    if not isinstance(robot, dict) or not isinstance(obj, dict):
        return False
    gripper = gripper_summary(robot.get("joint_state"))
    mean_position = gripper.get("mean_finger_position")
    if (
        isinstance(mean_position, (int, float))
        and float(mean_position) <= GRIPPER_EMPTY_CLOSED_POSITION_MAX
    ):
        return False
    error = pose_error(robot.get("end_effector_pose"), obj.get("pose"))
    return (
        isinstance(error, dict)
        and error["xy"] <= PRE_GRIP_OBJECT_XY_TOL
        and error["z_abs"] <= PRE_GRIP_OBJECT_Z_TOL
    )


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
    return isinstance(reach_info, dict) and reach_info.get("assessment") in {
        "reachable",
        "borderline",
    }


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
            "drop_slots": drop_slot_poses(table_center_handover_pose["position"]),
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
        gui_goal_target = CONTROL_GOAL_TARGETS.get(color)
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
            "gui_target_pose": (
                pose_with_position(gui_goal_target)
                if isinstance(gui_goal_target, dict)
                else None
            ),
            "placement_slots": (
                drop_slot_poses(gui_goal_target)
                if isinstance(gui_goal_target, dict)
                else []
            ),
            "note": (
                "For ordinary execution prefer the object-specific recommended_goal_drop_pose "
                "from objects_by_id when available; otherwise use control_service_contract.goal_targets[color]. "
                "physical_marker_pose and eef_drop_pose are diagnostic/recovery references."
            ),
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
            if isinstance(target_goal, dict)
            and isinstance(target_goal.get("reach"), dict)
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
        handover_candidate = (
            not direct_robot_candidates
            and bool(pickup_robot)
            and bool(destination_robot)
            and pickup_robot != destination_robot
        )
        if preferred_direct_robot:
            route_hint = (
                f"direct single-robot task preferred with {preferred_direct_robot}; "
                "direct_robot_candidates is non-empty, so table_center_handover is only a fallback if this direct attempt fails"
            )
        elif handover_candidate and pickup_robot and destination_robot:
            route_hint = (
                f"handover required by reach model: {pickup_robot} should move the object to table_center_handover, "
                f"then {destination_robot} should deliver near {target_goal.get('id')}"
            )
        elif pickup_robot and destination_robot:
            preferred_direct_robot = pickup_robot or destination_robot
            route_hint = f"direct single-robot task may be suitable with {preferred_direct_robot}"
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
            "handover_candidate": handover_candidate,
            "handover_needed": False,
            "recommended_goal_drop_pose": (
                target_goal.get("placement_slots", [])[
                    stable_slot_index(
                        object_id, len(target_goal.get("placement_slots", []))
                    )
                ]
                if isinstance(target_goal, dict)
                and isinstance(target_goal.get("placement_slots"), list)
                and target_goal.get("placement_slots")
                else None
            ),
            "recommended_handover_drop_pose": (
                named_locations["table_center_handover"]["drop_slots"][
                    stable_slot_index(
                        object_id,
                        len(named_locations["table_center_handover"]["drop_slots"]),
                    )
                ]
            ),
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
                "world.py owns action-level motion shaping for Moving, Centering, and Placing. "
                "Each queued movement rises vertically to home-level Z, moves horizontally at that high Z, then descends vertically."
            ),
            "target_z_policy": (
                "Moving descends slightly below the object marker for grasping. "
                "Centering and Placing descend to poses slightly above their target point to avoid table/goal collision, "
                "then world.py applies a small final XY drop variation."
            ),
            "orientation_policy": (
                "world.py ignores arbitrary movement-command orientation and uses the robot home end-effector orientation, "
                "preventing left-arm tip flips caused by mismatched requested orientations."
            ),
            "planning_implication": (
                "Agents should provide semantic target poses only. Do not add extra lift/descent waypoints for normal object transport; "
                "world.py already enforces the safe vertical-horizontal-vertical path."
            ),
            "note": "This is runtime behavior implemented by world.py, not a prompt-only convention.",
        },
        "workspace_model": {
            "workspace_radius_assumption": "approximate planar reach radius, table_length * 0.65",
            "workspace_radius": workspace_radius,
            "borderline_margin": REACH_BORDERLINE_MARGIN_M,
            "route_planning_hint": (
                "Use a direct single-robot task only when objects_by_id[*].direct_robot_candidates is non-empty. "
                "If direct_robot_candidates is empty and nearest_robot differs from destination_nearest_robot, use named_locations.table_center_handover."
            ),
        },
        "control_service_contract": {
                "canonical_object_transport_sequence": [
                    "Moving(object marker pose)",
                    "Grip",
                    "Placing(goal pose) for color goals, or Centering(table_center_handover pose) for handover/table-center stages",
                    "Release",
                    "Homing",
                ],
            "table_center_target": pose_with_position(CONTROL_TABLE_CENTER_TARGET),
            "table_center_drop_slots": drop_slot_poses(CONTROL_TABLE_CENTER_TARGET),
            "goal_targets": {
                color: pose_with_position(position)
                for color, position in CONTROL_GOAL_TARGETS.items()
            },
            "goal_drop_slots": {
                color: drop_slot_poses(position)
                for color, position in CONTROL_GOAL_TARGETS.items()
            },
            "note": (
                "This is the baseline ControlCommand service sequence implemented by world.py. "
                "Agents provide semantic target poses; world.py enforces vertical rise, high-Z horizontal travel, vertical descent, Z offsets, home-orientation motion, and a small final XY drop variation for Centering/Placing."
            ),
        },
        "critical_notes": [
            "robot_id 'left' is the bottom robot in the top-view image.",
            "robot_id 'right' is the top robot in the top-view image.",
            "Task decomposition should prefer preferred_direct_robot when present. Handover is exceptional, not the default.",
            "Mention robot_id and intermediate handover/buffer location whenever direct_robot_candidates is empty and pickup/destination robots differ.",
            "For an object-moving task, the first Moving action must target the object's current pose.",
            "Only after Grip may a later movement action target a goal/drop pose.",
            "The service action spelling is 'Release'. The older typo is accepted only for backward compatibility.",
            "For ordinary goal placement, prefer objects_by_id[object_id].recommended_goal_drop_pose; if absent, use control_service_contract.goal_targets[color].",
            "Goal physical_marker_pose/eef_drop_pose are recovery references; do not prefer them over the baseline service target unless the baseline sequence actually failed.",
            "world.py already performs vertical rise, high-Z horizontal travel, vertical descent, and small final XY variation for Moving/Centering/Placing; agents should not duplicate those waypoints.",
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


def arm_joint_motion_summary(joint_state: dict[str, Any] | None) -> dict[str, Any]:
    names = joint_state.get("names") if isinstance(joint_state, dict) else None
    velocities = (
        joint_state.get("velocities") if isinstance(joint_state, dict) else None
    )
    if not isinstance(names, list) or not isinstance(velocities, list):
        return joint_motion_summary(joint_state)
    arm_velocities: list[float] = []
    for name, velocity in zip(names, velocities):
        # Filter out gripper fingers and wrist roll (reaction force from gripper closing)
        if "finger" in str(name) or "panda_joint7" in str(name):
            continue
        try:
            arm_velocities.append(abs(float(velocity)))
        except (TypeError, ValueError):
            pass
    if not arm_velocities:
        return {"max_abs_velocity": None, "appears_moving": "unknown"}
    max_abs_velocity = max(arm_velocities)
    return {
        "max_abs_velocity": max_abs_velocity,
        "appears_moving": max_abs_velocity > PRE_GRIP_ARM_VELOCITY_TOL,
        "note": "Finger joint and wrist roll (panda_joint7) velocities are ignored; this reports whether the main arm links (joints 1-6) are still moving.",
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
    object_to_target_error_after = pose_error(
        (object_after or {}).get("pose") if object_after else None,
        action.target_pose,
    )
    eef_pose_delta = xyz_delta(eef_before, eef_after)
    eef_stationary_after_action = False
    if isinstance(eef_pose_delta, dict):
        eef_stationary_after_action = (
            abs(float(eef_pose_delta.get("dx", 0.0))) <= GRIP_EEF_STATIONARY_XY_TOL
            and abs(float(eef_pose_delta.get("dy", 0.0))) <= GRIP_EEF_STATIONARY_XY_TOL
            and abs(float(eef_pose_delta.get("dz", 0.0))) <= GRIP_EEF_STATIONARY_Z_TOL
        )
    grip_alignment_ok = None
    if action.action == ACTION_GRIP and object_to_eef_error_after is not None:
        grip_alignment_ok = (
            object_to_eef_error_after["xy"] <= PRE_GRIP_OBJECT_XY_TOL
            and object_to_eef_error_after["z_abs"] <= PRE_GRIP_OBJECT_Z_TOL
        )
    joint_motion_after = joint_motion_summary((robot_after or {}).get("joint_state"))
    arm_motion_after = arm_joint_motion_summary((robot_after or {}).get("joint_state"))
    moving_still_active = (
        action.action in MOVEMENT_ACTIONS
        and not eef_reached_relaxed
        and arm_motion_after.get("appears_moving") is True
    )
    gripper_after = gripper_summary((robot_after or {}).get("joint_state"))
    gripper_before = gripper_summary((robot_before or {}).get("joint_state"))
    mean_before = gripper_before.get("mean_finger_position")
    mean_after = gripper_after.get("mean_finger_position")
    gripper_closing_delta = None
    if isinstance(mean_before, (int, float)) and isinstance(mean_after, (int, float)):
        gripper_closing_delta = float(mean_before) - float(mean_after)
    grip_success_assessment = None
    if action.action == ACTION_GRIP:
        empty_fully_closed = (
            isinstance(mean_after, (int, float))
            and float(mean_after) <= GRIPPER_EMPTY_CLOSED_POSITION_MAX
        )
        grip_success_assessment = {
            "applies_to": "Grip",
            "single_runtime_rule": (
                "After Grip, only mean_finger_position is used. "
                "Near empty fully-closed means failed empty grasp; every other value is treated as success."
            ),
            "mean_finger_position": mean_after,
            "empty_fully_closed_threshold": GRIPPER_EMPTY_CLOSED_POSITION_MAX,
            "empty_fully_closed": empty_fully_closed,
            "runtime_status": "retry" if empty_fully_closed else "success",
            "gripper_closing_delta": gripper_closing_delta,
            "do_not_use": (
                "Do not use finger velocity, arm motion, EEF motion, alignment, or open/partially-closed state to keep Grip pending."
            ),
            "normal_contact_behavior": (
                "A grasped object prevents empty full closure; any non-empty-closed finger position is accepted as holding/contact."
            ),
        }
    return {
        "action_name": action.action,
        "robot_id": action.robot_id,
        "target_object_id": action.target_object_id,
        "target_pose": action.target_pose,
        "gripper_before": gripper_before,
        "gripper_after": gripper_after,
        "gripper_closing_delta": gripper_closing_delta,
        "joint_motion_after": joint_motion_after,
        "arm_motion_after": arm_motion_after,
        "eef_before": compact_pose(eef_before),
        "eef_after": compact_pose(eef_after),
        "eef_pose_delta": eef_pose_delta,
        "eef_to_target_pose_error_after": eef_error_after,
        "eef_to_target_xy_after": xy_distance(eef_after, action.target_pose),
        "eef_reach_assessment": {
            "standard_reached": eef_reached_standard,
            "relaxed_reached": eef_reached_relaxed,
            "applies_to_current_action": action.action in MOVEMENT_ACTIONS,
            "standard_tolerance": {
                "xy": OBJECT_APPROACH_XY_TOL,
                "z": OBJECT_APPROACH_Z_TOL,
            },
            "relaxed_tolerance": {
                "xy": EEF_REACHED_RELAXED_XY_TOL,
                "z": EEF_REACHED_RELAXED_Z_TOL,
            },
            "guidance": (
                "Use this only for Moving/Centering/Placing target poses. For Grip target_pose is null, so this assessment is not a Grip success criterion."
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
        "object_to_target_pose_error_after": object_to_target_error_after,
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
        "grip_success_assessment": grip_success_assessment,
        "moving_temporal_assessment": {
            "applies_to": "Moving|Centering|Placing",
            "still_active": moving_still_active,
            "guidance": (
                "If still_active is true, the motion has not settled yet. Diagnose pending rather than retry/replan. "
                "Judge failed arrival only after the robot is no longer clearly moving or the pending observation limit is reached."
            ),
        },
        "diagnostic_hints": [
            "For a pre-grasp Moving action before any Grip, an open gripper is expected and must not be diagnosed as failed grasp.",
            "If a Moving action is still active, return pending; do not return retry just because the current sample is not yet at the target.",
            "If a Moving action is immediately before Grip and its target_pose is the object's marker pose, do not diagnose retry from EEF-object z difference alone. EEF frame z may remain above the object marker center because the marker is not the gripper frame.",
            "For pre-grip Moving, prefer success when the service accepted the command, the motion has settled, and XY is near the object; use pending while still moving and retry only for clear XY miss, service failure, or settled wrong target.",
            "For a Grip action, the immediately preceding Moving must have settled at the grasp pose; a high clearance waypoint is not enough.",
            "After Grip, use exactly one runtime check: if mean_finger_position is near empty fully-closed, the grasp failed; otherwise treat Grip as successful.",
            "After a post-grip Moving/lift, the target object should move with the end effector; if object_pose_delta is tiny, the grasp likely failed or slipped.",
            "For Grip, the pre-grip wait already checks descent settling before sending Grip.",
            "For Grip, do not use joint motion, finger velocity, EEF motion, alignment, or open/partially-closed labels to keep the action pending.",
            "For Grip, fully open, open-range, and partially closed finger positions are all accepted unless the fingers are near empty fully-closed.",
            "For Moving/Centering/Placing, a service success only means the target was queued. If arm_motion_after is settled, EEF is still far from target, and eef_pose_delta is near zero, the queued move did not actually execute and should be replanned rather than kept pending.",
            "If replanning and the gripper is closed without a secure object, Release should occur before descending toward an object.",
            "For goal placement, target_pose z should be above the goal surface marker, not equal to the physical marker z.",
            "If eef_reach_assessment.relaxed_reached is true, prefer success over pending unless task-level evidence contradicts it.",
        ],
    }


def enforce_runtime_safety_diagnosis(
    diagnosis: dict[str, Any],
    result: ActionResult,
    snapshot_after: dict[str, Any],
) -> dict[str, Any]:
    action = result.action
    diagnostics = build_execution_diagnostics(result, snapshot_after)
    if not result.success:
        return {
            **diagnosis,
            "execution_diagnostics": diagnostics,
        }
    if action.action == ACTION_GRIP:
        gripper_after = diagnostics.get("gripper_after")
        mean_position = (
            gripper_after.get("mean_finger_position")
            if isinstance(gripper_after, dict)
            else None
        )
        empty_closed_grip = (
            isinstance(mean_position, (int, float))
            and float(mean_position) <= GRIPPER_EMPTY_CLOSED_POSITION_MAX
        )
        if empty_closed_grip:
            return {
                **diagnosis,
                "status": "retry",
                "message": (
                    "Runtime Grip assessment failed: finger position is near the empty fully-closed value "
                    f"({mean_position:.5f}), so the gripper likely closed without an object. "
                    + str(diagnosis.get("message", ""))
                ),
                "execution_diagnostics": diagnostics,
            }
    return {
        **diagnosis,
        "execution_diagnostics": diagnostics,
    }


def deterministic_grip_diagnosis(
    result: ActionResult, snapshot_after: dict[str, Any]
) -> dict[str, Any] | None:
    if result.action.action != ACTION_GRIP or not result.success:
        return None
    diagnostics = build_execution_diagnostics(result, snapshot_after)
    gripper_after = diagnostics.get("gripper_after")
    mean_position = (
        gripper_after.get("mean_finger_position")
        if isinstance(gripper_after, dict)
        else None
    )
    if isinstance(mean_position, (int, float)) and float(mean_position) <= GRIPPER_EMPTY_CLOSED_POSITION_MAX:
        return {
            "status": "retry",
            "message": (
                "Runtime Grip assessment failed: finger position is near the empty fully-closed value "
                f"({float(mean_position):.5f}), so the gripper likely closed without an object."
            ),
            "source": "runtime_grip_assessment",
            "execution_diagnostics": diagnostics,
        }
    return {
        "status": "success",
        "message": (
            "Runtime Grip assessment passed: finger position is not confirmed near empty fully-closed"
            + (
                f" ({float(mean_position):.5f})"
                if isinstance(mean_position, (int, float))
                else ""
            )
            + ", so Grip is treated as successful."
        ),
        "source": "runtime_grip_assessment",
        "execution_diagnostics": diagnostics,
    }


def deterministic_movement_diagnosis(
    result: ActionResult, snapshot_after: dict[str, Any]
) -> dict[str, Any] | None:
    if result.action.action not in MOVEMENT_ACTIONS or not result.success:
        return None
    diagnostics = build_execution_diagnostics(result, snapshot_after)
    eef_reach = diagnostics.get("eef_reach_assessment")
    arm_motion = diagnostics.get("arm_motion_after")
    eef_delta = diagnostics.get("eef_pose_delta")
    eef_error = diagnostics.get("eef_to_target_pose_error_after")
    object_target_error = diagnostics.get("object_to_target_pose_error_after")
    if not isinstance(eef_reach, dict) or not isinstance(arm_motion, dict):
        return None
    if eef_reach.get("relaxed_reached") is True:
        if result.action.action in {ACTION_CENTERING, ACTION_PLACING}:
            if (
                isinstance(object_target_error, dict)
                and float(object_target_error.get("xy", 0.0))
                > CARRIED_OBJECT_TARGET_XY_TOL
            ):
                return {
                    "status": "replan_task",
                    "message": (
                        f"Runtime {result.action.action} assessment failed: the EEF reached the target, "
                        "but the carried object is still not near the destination. Do not Release; replan the transfer."
                    ),
                    "source": "runtime_movement_assessment",
                    "execution_diagnostics": diagnostics,
                }
        return {
            "status": "success",
            "message": (
                f"Runtime {result.action.action} assessment passed: the EEF reached the target within relaxed tolerance."
            ),
            "source": "runtime_movement_assessment",
            "execution_diagnostics": diagnostics,
        }
    if arm_motion.get("appears_moving") is True:
        return {
            "status": "pending",
            "message": (
                f"Runtime {result.action.action} assessment is waiting: arm joints are still moving toward the target."
            ),
            "source": "runtime_movement_assessment",
            "execution_diagnostics": diagnostics,
        }
    if isinstance(eef_error, dict):
        no_progress = False
        clear_miss = float(eef_error.get("xy", 0.0)) >= MOVEMENT_CLEAR_MISS_XY_TOL
        if isinstance(eef_delta, dict):
            no_progress = (
                abs(float(eef_delta.get("dx", 0.0))) <= MOVEMENT_NO_PROGRESS_XY_TOL
                and abs(float(eef_delta.get("dy", 0.0))) <= MOVEMENT_NO_PROGRESS_XY_TOL
                and abs(float(eef_delta.get("dz", 0.0))) <= MOVEMENT_NO_PROGRESS_Z_TOL
            )
        if clear_miss and (
            no_progress or arm_motion.get("appears_moving") is not True
        ):
            return {
                "status": "replan_task",
                "message": (
                    f"Runtime {result.action.action} assessment failed: world.py accepted the command, "
                    "but the EEF is still far from the target and no active progress is visible. "
                    "This usually means the queued IK target is not executable for this robot/pose; replan using a reachable transfer or table_center_handover with the other robot."
                ),
                "source": "runtime_movement_assessment",
                "execution_diagnostics": diagnostics,
            }
        return {
            "status": "pending",
            "message": (
                f"Runtime {result.action.action} assessment is waiting: target has not been reached yet "
                f"(xy_error={float(eef_error.get('xy', 0.0)):.3f}, z_error={float(eef_error.get('z_abs', 0.0)):.3f}). "
                "Do not continue to Release until this movement succeeds."
            ),
            "source": "runtime_movement_assessment",
            "execution_diagnostics": diagnostics,
        }
    return None


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
        if action.action == ACTION_RELEASE:
            if index == 0:
                continue
            previous_action = actions[index - 1]
            if previous_action.action == ACTION_RELEASE:
                continue
            if previous_action.action not in MOVEMENT_ACTIONS:
                raise ValueError("Release must immediately follow a safe drop movement")
            containing_goal = goal_containing_pose(previous_action.target_pose, goals)
            if containing_goal is not None and previous_action.action != ACTION_PLACING:
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


def action_is_object_descent_before_grip(
    actions: list[PrimitiveAction],
    index: int,
    snapshot: dict[str, Any],
) -> bool:
    action = actions[index]
    if action.action != ACTION_MOVING or not action.target_object_id:
        return False
    if index + 1 >= len(actions):
        return False
    next_action = actions[index + 1]
    if (
        next_action.action != ACTION_GRIP
        or next_action.target_object_id != action.target_object_id
    ):
        return False
    obj = object_by_id(snapshot, action.target_object_id)
    object_pose = obj.get("pose") if isinstance(obj, dict) else None
    error = pose_error(action.target_pose, object_pose)
    if error is None:
        return False
    return (
        error["xy"] <= PRE_GRIP_OBJECT_XY_TOL
        and error["z_abs"] <= PRE_GRIP_OBJECT_Z_TOL
    )


def ensure_release_before_closed_gripper_descent(
    actions: list[PrimitiveAction],
    snapshot: dict[str, Any],
) -> list[PrimitiveAction]:
    if not actions:
        return actions
    if actions[0].action == ACTION_RELEASE:
        return actions
    for index, action in enumerate(actions):
        if action.action in {ACTION_RELEASE, ACTION_GRIP}:
            return actions
        if not action_is_object_descent_before_grip(actions, index, snapshot):
            continue
        if robot_gripper_state(snapshot, action.robot_id) not in {
            "closed",
            "partially_closed_or_holding",
        }:
            return actions
        return [
            PrimitiveAction(
                robot_id=action.robot_id,
                action=ACTION_RELEASE,
                target_pose=None,
                target_object_id=action.target_object_id,
            ),
            *actions,
        ]
    return actions


def infer_task_object_id(task: str, snapshot: dict[str, Any]) -> str | None:
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    object_ids = [
        str(obj.get("id")) for obj in objects if isinstance(obj, dict) and obj.get("id")
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


def infer_task_robot_id(task: str, snapshot: dict[str, Any]) -> str | None:
    lowered = task.lower()
    for robot_id in CONTROL_SERVICE_TOPICS:
        if f"{robot_id} robot" in lowered or f"robot {robot_id}" in lowered:
            return robot_id
    object_id = infer_task_object_id(task, snapshot)
    if object_id is not None:
        summary = build_agent_world_summary(snapshot)
        obj = summary.get("objects_by_id", {}).get(object_id)
        if isinstance(obj, dict):
            preferred = obj.get("preferred_direct_robot") or obj.get("nearest_robot")
            if preferred in CONTROL_SERVICE_TOPICS:
                return str(preferred)
    return None


def infer_task_destination(
    task: str, snapshot: dict[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    lowered = task.lower()
    object_id = infer_task_object_id(task, snapshot)
    summary = build_agent_world_summary(snapshot) if object_id is not None else {}
    summary_obj = (
        summary.get("objects_by_id", {}).get(object_id)
        if isinstance(summary.get("objects_by_id"), dict)
        else None
    )
    for color, target in CONTROL_GOAL_TARGETS.items():
        if f"{color}_goal" in lowered or f"{color} goal" in lowered:
            if (
                isinstance(summary_obj, dict)
                and summary_obj.get("target_goal_color") == color
                and isinstance(summary_obj.get("recommended_goal_drop_pose"), dict)
            ):
                return ACTION_PLACING, summary_obj["recommended_goal_drop_pose"]
            return ACTION_PLACING, pose_with_position(target)
    if "center" in lowered or "table_center" in lowered or "handover" in lowered:
        if (
            isinstance(summary_obj, dict)
            and isinstance(summary_obj.get("recommended_handover_drop_pose"), dict)
            and "goal" not in lowered
        ):
            return ACTION_CENTERING, summary_obj["recommended_handover_drop_pose"]
        return ACTION_CENTERING, pose_with_position(CONTROL_TABLE_CENTER_TARGET)
    if object_id is not None and isinstance(summary_obj, dict):
        color = str(summary_obj.get("target_goal_color") or "")
        if color in CONTROL_GOAL_TARGETS:
            recommended_pose = summary_obj.get("recommended_goal_drop_pose")
            if isinstance(recommended_pose, dict):
                return ACTION_PLACING, recommended_pose
            return ACTION_PLACING, pose_with_position(CONTROL_GOAL_TARGETS[color])
    return None


def canonical_control_action_plan(
    task: str,
    snapshot: dict[str, Any],
) -> list[PrimitiveAction] | None:
    object_id = infer_task_object_id(task, snapshot)
    robot_id = infer_task_robot_id(task, snapshot)
    destination = infer_task_destination(task, snapshot)
    if object_id is None or robot_id is None or destination is None:
        return None
    obj = object_by_id(snapshot, object_id)
    object_pose = (
        control_pose_from_pose(obj.get("pose")) if isinstance(obj, dict) else None
    )
    if object_pose is None:
        return None
    destination_action, destination_pose = destination
    if robot_appears_to_hold_object(snapshot, robot_id, object_id):
        return [
            PrimitiveAction(robot_id, destination_action, destination_pose, object_id),
            PrimitiveAction(robot_id, ACTION_RELEASE, None, object_id),
            PrimitiveAction(robot_id, ACTION_HOMING, None, None),
        ]
    return ensure_release_before_closed_gripper_descent(
        [
            PrimitiveAction(robot_id, ACTION_MOVING, object_pose, object_id),
            PrimitiveAction(robot_id, ACTION_GRIP, None, object_id),
            PrimitiveAction(robot_id, destination_action, destination_pose, object_id),
            PrimitiveAction(robot_id, ACTION_RELEASE, None, object_id),
            PrimitiveAction(robot_id, ACTION_HOMING, None, None),
        ],
        snapshot,
    )


def action_sequence_label(actions: list[PrimitiveAction]) -> str:
    return " -> ".join(action.action for action in actions)


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

    def context_text(self) -> str:
        return (
            "Runtime prompt structure:\n"
            "- system prompt: stable agent role and JSON output contract.\n"
            "- CLAUDE.md: domain/context guidance and durable project lessons for this robot simulation.\n"
            "- MEMORY.md is no longer loaded or updated by this CLI.\n\n"
            "CLAUDE.md domain guidance:\n"
            + self.claude_path.read_text(encoding="utf-8")
        )

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
                "You are the Task agent. Decompose the user's high-level robot command into concrete natural-language object-level tasks. "
                'Return JSON only: {"tasks": [string, ...], "notes": string}. '
                "Do not emit ROS primitive actions. Use the supplied world_summary, task_rules, and agent_context for domain rules."
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
                    "Use world_summary.objects_by_id[*].preferred_direct_robot for ordinary direct tasks.",
                    "If an object is already inside its matching color goal region, do not create a task for that object.",
                    "If direct_robot_candidates is non-empty or preferred_direct_robot is present, create a single direct task to the final color goal; do not insert table_center_handover.",
                    "Create one direct task using the preferred robot and the final color goal unless feedback says a direct attempt already failed.",
                    "If preferred_direct_robot is absent, direct_robot_candidates is empty, and nearest_robot differs from destination_nearest_robot, split into two tasks: pickup robot moves object to table_center_handover, then destination robot moves it from that location to the final goal.",
                    "When a direct attempt actually failed, also split into the same table_center_handover route instead of repeating the unreachable direct placement.",
                    "For recovery staged tasks, include the handover source/destination in natural language so the ActionAgent knows whether it is picking from the object's original pose or from a buffer.",
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
                "You are the Action agent. Convert one natural-language subtask into directly executable ROS2 primitive actions. "
                'Return JSON only: {"actions": [{"robot_id": "left"|"right", '
                '"action": "Moving"|"Centering"|"Placing"|"Grip"|"Release"|"Homing", '
                '"target_pose": object|null, "target_object_id": string|null}]}. '
                "Do not include reason, metadata, intent, or extra fields. "
                "Use the supplied action_rules, world_summary, feedback, and agent_context for the robot-specific policy."
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
                    "Default to the baseline ControlCommand workflow: Moving(object pose), Grip, then Placing for a color goal or Centering for table_center_handover, Release, Homing.",
                    "Use Placing for red_goal or blue_goal. Use Centering for table_center or table_center_handover.",
                    "For Placing, prefer world_summary.objects_by_id[object_id].recommended_goal_drop_pose so repeated same-color objects do not all target the exact same XY; if absent, use world_summary.control_service_contract.goal_targets[color].",
                    "For Centering to table_center_handover, prefer world_summary.objects_by_id[object_id].recommended_handover_drop_pose or the named location drop_slots; if absent, use control_service_contract.table_center_target.",
                    "If the task destination is world_summary.named_locations.<name>, use Centering or that named location pose as the stage drop target rather than the final color goal.",
                    "If the task says the object is coming from a named handover location, still use the object's current observed marker pose as the pre-grasp Moving target.",
                    "The immediate action before Grip must be a Moving descent whose target_pose matches that object's current pose/grasp height from world_summary.objects_by_id, and the runtime will block Grip until both XY and Z are aligned and the arm is settled.",
                    "Do not move to the goal before Grip.",
                    "Do not add extra lift, approach, clearance, or descent waypoints. world.py enforces vertical rise, high-Z horizontal travel, vertical descent, and small final XY variation for Moving, Centering, and Placing.",
                    "Return enough actions to accomplish the current task; do not assume the runtime will ask you for the missing primitives of the same task.",
                    "The final primitive must be Homing with target_pose null and target_object_id null.",
                    "If feedback indicates failed grasp/slip/retry and the selected robot gripper is closed or partially closed, Release before Moving down to an object.",
                    "Do not use a goal marker id as target_object_id.",
                    "Even when target_pose is a goal or named destination pose, target_object_id must remain the manipulated object id from the task.",
                    "Use 'Release' exactly for opening the gripper.",
                    "target_pose orientation may be omitted; world.py uses each robot's home end-effector orientation for movement to avoid left-arm tip flips.",
                ],
                "required_schema": {
                    "actions": [
                        {
                            "robot_id": "left|right",
                            "action": "Moving|Centering|Placing|Grip|Release|Homing",
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
        actions = ensure_release_before_closed_gripper_descent(actions, snapshot)
        validate_action_sequence_against_snapshot(actions, snapshot)
        return actions


class TroubleshooterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            "troubleshooter",
            (
                "You are the Troubleshooter agent. Diagnose one primitive using before/after ROS2 snapshots and the planned sequence. "
                "Return only one allowed status from the payload. For non-final primitives, never return complete. "
                "Use execution_diagnostics, sequence_context, and agent_context for domain rules; do not rely on service success alone. "
                'Return JSON only: {"status": "success"|"pending"|"retry"|"replan_task"|'
                '"replan_all"|"emergency_recover"|"complete", "message": string}.'
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
        execution_diagnostics = build_execution_diagnostics(result, snapshot_after)
        motion_pending = (
            execution_diagnostics.get("moving_temporal_assessment", {}).get(
                "still_active"
            )
            is True
        )
        allowed_statuses = sorted(TROUBLESHOOTER_STATUSES)
        if motion_pending:
            allowed_statuses = ["pending"]
        reply = self.call_json(
            {
                "task": task,
                "action_result": result.as_dict(),
                "snapshot_before": result.snapshot_before,
                "snapshot_after": snapshot_after,
                "world_summary_after": build_agent_world_summary(snapshot_after),
                "execution_diagnostics": execution_diagnostics,
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
                        "Return pending when the action may still be settling and the relaxed EEF tolerance has not been reached. "
                        "If moving_temporal_assessment.still_active is true, pending is the only valid diagnosis."
                    ),
                },
                "allowed_status": allowed_statuses,
            },
            image_bytes=image_bytes,
        )
        status = str(reply.parsed.get("status") or "").strip()
        if motion_pending and status != "pending":
            reply.parsed["status"] = "pending"
            reply.parsed["message"] = (
                "Runtime corrected Troubleshooter status to pending because the Moving action is still active. "
                + str(reply.parsed.get("message", ""))
            )
            status = "pending"
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
                },
            },
            image_bytes=image_bytes,
        )
        return reply.parsed


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
    if action_name in {"Center", "center", "Centering", "centering"}:
        action_name = ACTION_CENTERING
    if action_name in {"Place", "place", "Placing", "placing"}:
        action_name = ACTION_PLACING
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
    target_pose = control_pose_from_pose(target_pose)
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
        if action.action == ACTION_GRIP:
            ready, message, before = self._wait_for_grip_precondition(action, before)
            if not ready:
                return ActionResult(False, message, action, before, before)
        success, message = self.db_node.call_control_service(action)
        self._action_interval(action.action)
        after, _ = self.db_node.snapshot(include_image=False)
        return ActionResult(success, message, action, before, after)

    def _wait_for_grip_precondition(
        self, action: PrimitiveAction, initial_snapshot: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any]]:
        if not action.target_object_id:
            return (
                False,
                "Grip requires target_object_id before closing.",
                initial_snapshot,
            )
        deadline = time.monotonic() + PRE_GRIP_SETTLE_TIMEOUT_SEC
        last_snapshot = initial_snapshot
        last_reason = "waiting for pre-grip descent to settle"
        reported = False
        while rclpy.ok() and time.monotonic() <= deadline:
            snapshot, _ = self.db_node.snapshot(include_image=False)
            last_snapshot = snapshot
            robot = robot_by_id(snapshot, action.robot_id)
            obj = object_by_id(snapshot, action.target_object_id)
            eef_pose = (robot or {}).get("end_effector_pose")
            object_pose = (obj or {}).get("pose") if isinstance(obj, dict) else None
            alignment_error = pose_error(eef_pose, object_pose)
            arm_motion = arm_joint_motion_summary((robot or {}).get("joint_state"))
            gripper = gripper_summary((robot or {}).get("joint_state"))
            gripper_state = str(gripper.get("state"))
            xy_ready = (
                alignment_error is not None
                and alignment_error["xy"] <= PRE_GRIP_OBJECT_XY_TOL
            )
            z_ready = (
                alignment_error is not None
                and alignment_error["z_abs"] <= PRE_GRIP_OBJECT_Z_TOL
            )
            arm_settled = arm_motion.get("appears_moving") is not True
            gripper_open = gripper_state == "open"
            if xy_ready and z_ready and arm_settled and gripper_open:
                return (
                    True,
                    (
                        "Pre-grip descent settled: EEF XY/Z are near target object, "
                        "arm motion is settled, and gripper is open."
                    ),
                    snapshot,
                )
            last_reason = (
                "Grip blocked until descent completes: "
                f"xy_error={(alignment_error or {}).get('xy')}, z_error={(alignment_error or {}).get('z_abs')}, "
                f"arm_motion={arm_motion.get('appears_moving')}, "
                f"gripper_state={gripper_state}."
            )
            if not reported:
                self.reporter.status(
                    "Pre-grip wait",
                    {
                        "robot_id": action.robot_id,
                        "target_object_id": action.target_object_id,
                        "timeout_sec": PRE_GRIP_SETTLE_TIMEOUT_SEC,
                    },
                )
                reported = True
            time.sleep(0.1)
        return (
            False,
            (
                "Grip was not sent because the preceding descent did not settle. "
                + last_reason
            ),
            last_snapshot,
        )

    @staticmethod
    def _action_interval(action_name: str) -> None:
        settle_sec = {
            ACTION_MOVING: MOVING_SETTLE_SEC,
            ACTION_CENTERING: MOVING_SETTLE_SEC,
            ACTION_PLACING: PLACING_SETTLE_SEC,
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


def feedback_requires_recovery(feedback: str) -> bool:
    lowered = feedback.lower()
    recovery_markers = (
        "failed",
        "failure",
        "rejected",
        "retry",
        "not executable",
        "slip",
        "collision",
        "grasp",
        "world.py",
        "controlcommand",
        "replan",
        "emergency",
        "실패",
        "거절",
        "충돌",
        "재계획",
    )
    return any(marker in lowered for marker in recovery_markers)


def requested_sort_colors(command: str) -> set[str]:
    lowered = command.lower()
    colors: set[str] = set()
    if "red" in lowered or "빨간" in lowered or "빨강" in lowered:
        colors.add("red")
    if "blue" in lowered or "파란" in lowered or "파랑" in lowered:
        colors.add("blue")
    return colors


def color_sort_completion_status(
    command: str, snapshot: dict[str, Any]
) -> tuple[bool | None, list[str]]:
    colors = requested_sort_colors(command)
    if not colors:
        return None, []
    objects = (
        snapshot.get("objects") if isinstance(snapshot.get("objects"), list) else []
    )
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    missing: list[str] = []
    checked = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        color = str(obj.get("color") or "")
        if color not in colors:
            continue
        checked += 1
        goal = goals.get(color)
        if not isinstance(goal, dict) or not goal_containing_pose(
            obj.get("pose"), {color: goal}
        ):
            missing.append(str(obj.get("id") or f"{color}_object"))
    if checked == 0:
        return None, []
    return not missing, missing


def object_in_own_goal(snapshot: dict[str, Any], object_id: str | None) -> bool:
    obj = object_by_id(snapshot, object_id)
    if not isinstance(obj, dict):
        return False
    color = str(obj.get("color") or "")
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    goal = goals.get(color)
    return isinstance(goal, dict) and bool(
        goal_containing_pose(obj.get("pose"), {color: goal})
    )


def task_satisfied_by_snapshot(task: str, snapshot: dict[str, Any]) -> bool:
    object_id = infer_task_object_id(task, snapshot)
    if object_id is None:
        return False
    return object_in_own_goal(snapshot, object_id)


def baseline_sort_tasks_from_snapshot(
    command: str, snapshot: dict[str, Any]
) -> list[str]:
    colors = requested_sort_colors(command)
    if not colors:
        return []
    summary = build_agent_world_summary(snapshot)
    objects_by_id = summary.get("objects_by_id")
    if not isinstance(objects_by_id, dict):
        return []
    goals = snapshot.get("goals") if isinstance(snapshot.get("goals"), dict) else {}
    tasks: list[str] = []
    for object_id in sorted(objects_by_id):
        obj = objects_by_id[object_id]
        if not isinstance(obj, dict):
            continue
        color = str(obj.get("color") or "")
        if color not in colors:
            continue
        goal = goals.get(color)
        if isinstance(goal, dict) and goal_containing_pose(
            obj.get("pose"), {color: goal}
        ):
            continue
        goal_id = obj.get("target_goal_id")
        if not goal_id and isinstance(goal, dict):
            goal_id = goal.get("id")
        if not goal_id:
            continue
        direct_candidates = (
            obj.get("direct_robot_candidates")
            if isinstance(obj.get("direct_robot_candidates"), list)
            else []
        )
        pickup_robot = obj.get("nearest_robot")
        destination_robot = obj.get("destination_nearest_robot")
        if (
            not direct_candidates
            and pickup_robot in CONTROL_SERVICE_TOPICS
            and destination_robot in CONTROL_SERVICE_TOPICS
            and pickup_robot != destination_robot
        ):
            tasks.append(
                f"{pickup_robot} robot move {object_id} to table_center_handover for {destination_robot} robot handover"
            )
            tasks.append(
                f"{destination_robot} robot move {object_id} from table_center_handover to {goal_id}"
            )
            continue
        robot_id = obj.get("preferred_direct_robot") or pickup_robot
        if robot_id not in CONTROL_SERVICE_TOPICS:
            continue
        tasks.append(f"{robot_id} robot move {object_id} to {goal_id}")
    return tasks


def handover_tasks_for_task(task: str, snapshot: dict[str, Any]) -> list[str] | None:
    if "table_center_handover" in task:
        return None
    object_id = infer_task_object_id(task, snapshot)
    if object_id is None:
        return None
    if object_in_own_goal(snapshot, object_id):
        return None
    summary = build_agent_world_summary(snapshot)
    obj = summary.get("objects_by_id", {}).get(object_id)
    if not isinstance(obj, dict):
        return None
    goal_id = obj.get("target_goal_id")
    pickup_robot = obj.get("nearest_robot")
    destination_robot = obj.get("destination_nearest_robot")
    direct_candidates = (
        obj.get("direct_robot_candidates")
        if isinstance(obj.get("direct_robot_candidates"), list)
        else []
    )
    if (
        direct_candidates
        or pickup_robot not in CONTROL_SERVICE_TOPICS
        or destination_robot not in CONTROL_SERVICE_TOPICS
        or pickup_robot == destination_robot
        or not goal_id
    ):
        return None
    return [
        f"{pickup_robot} robot move {object_id} to table_center_handover for {destination_robot} robot handover",
        f"{destination_robot} robot move {object_id} from table_center_handover to {goal_id}",
    ]


def choose_task_list(
    command: str,
    snapshot: dict[str, Any],
    llm_tasks: list[str],
    reporter: Reporter,
) -> list[str]:
    sort_complete, _ = color_sort_completion_status(command, snapshot)
    if sort_complete is True:
        reporter.status(
            "Task baseline",
            {
                "source": "world_state_color_sort",
                "tasks": 0,
                "note": "All requested objects are already in their target goal regions.",
            },
        )
        return []
    baseline_tasks = baseline_sort_tasks_from_snapshot(command, snapshot)
    if baseline_tasks:
        reporter.status(
            "Task baseline",
            {
                "source": "world_state_color_sort",
                "tasks": len(baseline_tasks),
                "note": "Using direct object-level tasks from ROS state for the standard color sorting command.",
            },
        )
        return baseline_tasks
    filtered_tasks = [
        task for task in llm_tasks if not task_satisfied_by_snapshot(task, snapshot)
    ]
    if len(filtered_tasks) != len(llm_tasks):
        reporter.status(
            "Task filter",
            {
                "source": "world_state_goal_membership",
                "removed": len(llm_tasks) - len(filtered_tasks),
                "remaining": len(filtered_tasks),
            },
        )
    return filtered_tasks


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
    executor = ActionExecutor(db_node, reporter)

    try:
        db_agent.start()
        wait_for_ready(db_node, reporter)
        command = ask_user_command()
        snapshot, image_bytes = db_agent.snapshot()
        llm_tasks = task_agent.decompose(command, snapshot, image_bytes)
        tasks = choose_task_list(command, snapshot, llm_tasks, reporter)
        sort_complete, missing_objects = color_sort_completion_status(command, snapshot)
        if sort_complete is True:
            reporter.status(
                "Complete",
                {
                    "source": "world_state_color_sort",
                    "message": "All requested objects are already in their target goal regions.",
                },
            )
            return 0
        if not tasks:
            if missing_objects:
                reporter.error(
                    "No executable tasks were produced, but some requested objects are still outside their target goals: "
                    + ", ".join(missing_objects)
                )
                return 1
            reporter.status(
                "No tasks",
                {
                    "message": "No executable tasks were produced for the current world state.",
                    "missing_objects": missing_objects,
                },
            )
            return 0
        completed_tasks: list[str] = []
        trouble_reports: list[dict[str, Any]] = []
        task_index = 0
        replan_attempts: dict[int, int] = {}

        for step in range(1, MAX_MAIN_STEPS + 1):
            snapshot, image_bytes = db_agent.snapshot()
            sort_complete, missing_objects = color_sort_completion_status(
                command, snapshot
            )
            if sort_complete is True:
                reporter.status(
                    "Complete",
                    {
                        "source": "world_state_color_sort",
                        "message": "All requested objects are in their target goal regions.",
                    },
                )
                return 0
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
                sort_complete, missing_objects = color_sort_completion_status(
                    command, snapshot
                )
                if sort_complete is False:
                    reporter.status(
                        "Completion blocked",
                        {
                            "reason": "MainAgent claimed completion, but object-goal state is not satisfied.",
                            "missing_objects": missing_objects,
                            "message": completion.get("message", ""),
                        },
                    )
                else:
                    if sort_complete is None and task_index < len(tasks):
                        reporter.status(
                            "Completion blocked",
                            {
                                "reason": "MainAgent claimed completion before all planned tasks were consumed.",
                                "task_index": task_index,
                                "tasks": len(tasks),
                                "message": completion.get("message", ""),
                            },
                        )
                    else:
                        reporter.status(
                            "Complete",
                            {
                                "message": completion.get(
                                    "message", "MainAgent marked complete"
                                )
                            },
                        )
                        return 0

            if isinstance(completion.get("next_task_index"), int):
                task_index = max(
                    0, min(int(completion["next_task_index"]), len(tasks) - 1)
                )
            if task_index >= len(tasks):
                task_index = max(0, len(tasks) - 1)
            task = tasks[task_index]
            if task_satisfied_by_snapshot(task, snapshot):
                if task not in completed_tasks:
                    completed_tasks.append(task)
                reporter.status(
                    "Task skipped",
                    {
                        "source": "world_state_goal_membership",
                        "task": task,
                        "reason": "The target object is already inside its own goal region.",
                    },
                )
                task_index += 1
                continue
            if trouble_reports and trouble_reports[-1].get("task") == task:
                feedback = str(trouble_reports[-1].get("message", ""))
            else:
                feedback = ""
            actions: list[PrimitiveAction] | None = None
            plan_source = "action_agent"
            canonical_actions = canonical_control_action_plan(task, snapshot)
            if canonical_actions is not None and not feedback_requires_recovery(
                feedback
            ):
                try:
                    validate_action_sequence_against_snapshot(
                        canonical_actions, snapshot
                    )
                    actions = canonical_actions
                    plan_source = "control_service_baseline"
                    reporter.status(
                        "ControlCommand baseline plan",
                        {
                            "task": task,
                            "actions": len(actions),
                            "sequence": action_sequence_label(actions),
                        },
                    )
                except Exception as exc:
                    feedback = (
                        f"Baseline ControlCommand plan was invalid before execution: {exc}. "
                        "Use recovery planning."
                    )
            if actions is None:
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
                            "For object-moving tasks, prefer the baseline ControlCommand flow Moving -> Grip -> destination action -> Release -> Homing, where destination action is Placing for color goals or Centering for table_center_handover. "
                            "Use recovery waypoints only if the simple flow is not executable."
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
                fallback_actions = canonical_control_action_plan(task, snapshot)
                if fallback_actions is None:
                    reporter.error(
                        f"ActionAgent could not produce a valid plan for task: {task}"
                    )
                    continue
                try:
                    validate_action_sequence_against_snapshot(
                        fallback_actions, snapshot
                    )
                except Exception as exc:
                    reporter.error(
                        f"Baseline ControlCommand fallback plan was invalid for task {task}: {exc}"
                    )
                    continue
                actions = fallback_actions
                plan_source = "control_service_fallback"
                reporter.status(
                    "ControlCommand baseline fallback",
                    {
                        "task": task,
                        "actions": len(actions),
                        "sequence": action_sequence_label(actions),
                    },
                )
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
                    }
                elif plan_source in {
                    "control_service_baseline",
                    "control_service_fallback",
                } and (
                    (
                        action.action == ACTION_MOVING
                        and action_index < len(actions)
                        and actions[action_index].action == ACTION_GRIP
                    )
                    or action.action == ACTION_HOMING
                ):
                    diagnosis = {
                        "status": "success",
                        "message": (
                            f"Baseline ControlCommand fast-path step accepted by world.py: {result.message}"
                        ),
                        "source": plan_source,
                    }
                else:
                    diagnosis = deterministic_grip_diagnosis(result, snapshot_after)
                    if diagnosis is None:
                        diagnosis = deterministic_movement_diagnosis(
                            result, snapshot_after
                        )
                    if diagnosis is None:
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
                    diagnosis = deterministic_grip_diagnosis(result, snapshot_after)
                    if diagnosis is None:
                        diagnosis = deterministic_movement_diagnosis(
                            result, snapshot_after
                        )
                    if diagnosis is None:
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
                diagnosis = enforce_runtime_safety_diagnosis(
                    diagnosis, result, snapshot_after
                )
                diagnosis = {
                    **diagnosis,
                    "task": task,
                    "action": action.as_dict(),
                    "action_index": action_index,
                    "total_actions": len(actions),
                }
                trouble_reports.append(diagnosis)
                status_panel = (
                    "ControlCommand step"
                    if diagnosis.get("source")
                    in {"control_service_baseline", "control_service_fallback"}
                    else "Troubleshooter"
                )
                reporter.status(status_panel, diagnosis)

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
                llm_tasks = task_agent.decompose(command, snapshot, image_bytes)
                tasks = choose_task_list(command, snapshot, llm_tasks, reporter)
                sort_complete, missing_objects = color_sort_completion_status(
                    command, snapshot
                )
                if sort_complete is True:
                    reporter.status(
                        "Complete",
                        {
                            "source": "world_state_color_sort",
                            "message": "All requested objects are in their target goal regions after replanning.",
                        },
                    )
                    return 0
                if not tasks:
                    if missing_objects:
                        reporter.error(
                            "Replanning produced no executable tasks, but some requested objects are still outside their target goals: "
                            + ", ".join(missing_objects)
                        )
                        return 1
                    reporter.status(
                        "No tasks",
                        {
                            "message": "Replanning produced no remaining executable tasks.",
                            "missing_objects": missing_objects,
                        },
                    )
                    return 0
                completed_tasks = []
                task_index = 0
                continue
            if latest_status == "replan_task":
                snapshot, image_bytes = db_agent.snapshot()
                if task_satisfied_by_snapshot(task, snapshot):
                    if task not in completed_tasks:
                        completed_tasks.append(task)
                    task_index += 1
                    reporter.status(
                        "Task replan skipped",
                        {
                            "source": "world_state_goal_membership",
                            "task": task,
                            "reason": "The target object reached its own goal region before replanning.",
                        },
                    )
                    continue
                staged_tasks = handover_tasks_for_task(task, snapshot)
                if staged_tasks:
                    tasks = tasks[:task_index] + staged_tasks + tasks[task_index + 1 :]
                    reporter.status(
                        "Task replan",
                        {
                            "source": "runtime_handover_replan",
                            "old_task": task,
                            "new_tasks": staged_tasks,
                            "reason": "direct destination move failed or is unreachable; using table_center_handover",
                        },
                    )
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
