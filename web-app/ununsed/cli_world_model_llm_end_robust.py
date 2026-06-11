#!/usr/bin/python3
"""World-model + LLM semantic planner CLI for the dual-Franka Isaac Sim scene.

Design goals
------------
- User command is entered interactively inside the CLI, not as an argument.
- Runtime constants are configured as globals, not argparse fields.
- ROS2 observations are converted into a symbolic world model before calling the LLM.
- The LLM performs natural-language understanding and high-level task planning.
- The LLM outputs MCP-like semantic tool calls, never raw ROS primitive commands.
- The CLI validates semantic plans, decomposes them into Moving/Grip/Realease/Homing,
  executes them through ROS2 services, and verifies results with strict geometry plus optional multimodal image checks.
- Top-view image verification never infers whether an object is grasped/held, because grasp state is not observable from the top-down camera.

The ROS service still uses the existing action strings exposed by world.py:
Moving, Grip, Realease, Homing.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

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


# =============================================================================
# Global configuration
# =============================================================================

# ROS topics and services.
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
ROBOT_SCENE_LABELS = {
    "left": "bottom_robot_in_top_view",
    "right": "top_robot_in_top_view",
}

# Existing simulator action strings. Keep the service typo "Realease".
ACTION_MOVING = "Moving"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Realease"
ACTION_HOMING = "Homing"
ALLOWED_PRIMITIVE_ACTIONS = {ACTION_MOVING, ACTION_GRIP, ACTION_RELEASE, ACTION_HOMING}

# Interactive command behavior.
DEFAULT_COMMAND = "빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라."
USE_DEFAULT_ON_EMPTY_INPUT = True

# LLM backend. "auto" tries OpenAI first when an API key exists, then Ollama.
LLM_BACKEND = os.getenv("SEMANTIC_BACKEND", "auto").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SEC = 120
SEND_IMAGE_TO_LLM = True
MAX_LLM_REPLAN_ATTEMPTS = 3

# Pose conventions.
VERTICAL_EEF_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
DEFAULT_TABLE_CENTER = {"x": 0.6, "y": 0.0, "z": 0.46}
DEFAULT_TABLE_SIZE = {"x": 0.9, "y": 0.7, "z": 0.08}

# Inferred reachability rule requested by the project.
# A pose is reachable when its XY distance from robot base is <= 70% of the table major length.
REACH_RADIUS_TABLE_SCALE = 0.70
CENTER_REACH_MARGIN_M = 0.06

# Motion and placement safety parameters.
SERVICE_WAIT_TIMEOUT_SEC = 1.0
EEF_POSITION_TOLERANCE_M = 0.035
MOTION_TIMEOUT_SEC = 20.0
SETTLE_SEC = 0.7
LIFT_DELTA_M = 0.12
SAFE_Z_OFFSET_M = 0.16
GOAL_MARGIN_M = 0.014
DROP_CLEARANCE_M = 0.052
BUFFER_CLEARANCE_M = 0.035
GOAL_Z_OFFSET_M = 0.03
BUFFER_Z_OFFSET_M = 0.0
OUTSIDE_GOAL_Z_OFFSET_M = 0.0

# Verification.
VERIFY_DELAY_SEC = 0.8
VERIFY_XY_TOLERANCE_M = 0.06
VERIFY_Z_TOLERANCE_M = 0.07

# Goal placement must be stricter than "object center is inside the goal".
# The object footprint must be fully inside the goal rectangle with this extra inward margin.
GOAL_INTERIOR_MARGIN_M = 0.030

# Goal drop pose scoring. Prefer robust interior placement over the shortest travel path.
GOAL_CENTER_BIAS_WEIGHT = 2.0
GOAL_TRAVEL_BIAS_WEIGHT = 0.03
GOAL_CLEARANCE_BONUS_WEIGHT = 0.35

FINISH_VERIFY_CHECKS = 2
FINISH_VERIFY_INTERVAL_SEC = 0.4

# Final verification must be performed only after both robots have been explicitly homed.
# This prevents the top-view verifier from accepting a task while an arm is still occluding,
# hovering over, or interacting with the final object/goal region.
FINAL_VERIFY_REQUIRES_ALL_ROBOTS_HOMED = True
FINAL_HOMING_ROBOT_ORDER = ("left", "right")

# Emergency recovery. If primitive/finish failures repeat, force all grippers open and home both arms.
CONSECUTIVE_FAILURE_EMERGENCY_THRESHOLD = 3
EMERGENCY_ROBOT_ORDER = ("left", "right")

# Multimodal visual verification uses the same image sent to the semantic planner, but
# only after strict marker/geometry checks have passed. The camera is top-view only,
# so visual verification must never decide whether an object is held by a gripper.
USE_MULTIMODAL_VISUAL_VERIFICATION = True
REQUIRE_MULTIMODAL_VISUAL_VERIFICATION = False
VISUAL_VERIFY_CONFIDENCE_THRESHOLD = 0.60

# MarkerArray geometry is the metric source of truth. Top-down visual verification is
# advisory only: it may add warnings, but it must not overturn a strict marker/footprint
# predicate that already passed. This prevents false negatives caused by perspective,
# occlusion, or the LLM treating compact symbolic coordinates incorrectly.
VISUAL_VERIFICATION_CAN_VETO_MARKER_SUCCESS = False
FINAL_VISUAL_VERIFICATION_CAN_VETO_MARKER_SUCCESS = False

# Execution loop.
MAX_STEPS = 120
SAVE_LLM_FRAMES_DIR: Path | None = None
VERBOSE = True
PLAIN_OUTPUT = False
DRY_RUN = False

# Safety policy.
ALLOW_OFF_TABLE_DESTINATION = False
SUPPORTED_COLORS = {"red", "blue"}
SUPPORTED_SHAPES = {"cube", "sphere", "capsule"}
SUPPORTED_CURRENT_REGIONS = {
    "anywhere",
    "inside_any_goal",
    "inside_red_goal",
    "inside_blue_goal",
    "outside_goals",
    "handover_buffer",
    "table_center",
}
SUPPORTED_DESTINATION_TYPES = {
    "goal",
    "outside_goals",
    "safe_free_space",
    "table_center",
    "handover_buffer",
}


# =============================================================================
# Data models
# =============================================================================


@dataclass
class PrimitiveDecision:
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
    decision: PrimitiveDecision


@dataclass
class SemanticGoalCondition:
    condition_type: str
    object_ids: list[str]
    required_region: str | None = None
    required_goal_color: str | None = None
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_type": self.condition_type,
            "object_ids": list(self.object_ids),
            "required_region": self.required_region,
            "required_goal_color": self.required_goal_color,
            "description": self.description,
        }


@dataclass
class SemanticToolCall:
    tool: str
    arguments: dict[str, Any]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments, "reason": self.reason}


@dataclass
class LLMPlan:
    status: str
    reasoning_summary: str
    tool_calls: list[SemanticToolCall]
    goal_conditions: list[SemanticGoalCondition] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasoning_summary": self.reasoning_summary,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "goal_conditions": [
                condition.as_dict() for condition in self.goal_conditions
            ],
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class RouteCandidate:
    object_id: str
    route_type: str  # direct, direct_held, or handover
    source_robot_id: str
    destination_robot_id: str
    drop_pose: dict[str, Any]
    score: float
    buffer_pose: dict[str, Any] | None = None
    reason: str = ""


@dataclass
class ObjectTask:
    object_id: str
    route: RouteCandidate
    source_tool_call: SemanticToolCall
    destination: dict[str, Any]
    rank: int
    rationale: str


# =============================================================================
# Basic helpers
# =============================================================================


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = LLM_TIMEOUT_SEC,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request_object = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_object, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
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


def marker_type_name(marker_type: int) -> str:
    return {
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
    }.get(int(marker_type), f"type_{marker_type}")


def shape_from_marker(namespace: str, marker_type: int) -> str:
    prefix = namespace.split("_", 1)[0].lower()
    if prefix in SUPPORTED_SHAPES:
        return prefix
    return marker_type_name(marker_type)


def xy_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def xyz_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2
        + (float(a["y"]) - float(b["y"])) ** 2
        + (float(a["z"]) - float(b["z"])) ** 2
    )


def point_in_box_region(
    point: dict[str, float], region: dict[str, Any], margin: float = 0.0
) -> bool:
    center = region["pose"]["position"]
    scale = region.get("scale") or region.get("size") or DEFAULT_TABLE_SIZE
    half_x = max(0.0, float(scale["x"]) * 0.5 - margin)
    half_y = max(0.0, float(scale["y"]) * 0.5 - margin)
    return (
        abs(float(point["x"]) - float(center["x"])) <= half_x
        and abs(float(point["y"]) - float(center["y"])) <= half_y
    )


def object_xy_half_extents(obj: dict[str, Any]) -> tuple[float, float]:
    """Return a conservative top-view half footprint for a marker object."""
    scale = obj.get("scale") or {}
    try:
        half_x = max(0.0, float(scale.get("x", 0.0)) * 0.5)
        half_y = max(0.0, float(scale.get("y", 0.0)) * 0.5)
    except Exception:
        half_x = half_y = 0.0
    return half_x, half_y


def object_pose_fully_inside_box_region(
    obj: dict[str, Any],
    pose_or_position: dict[str, Any],
    region: dict[str, Any],
    margin: float = 0.0,
) -> bool:
    """Check whether the object's XY footprint is fully inside a rectangular region.

    A previous implementation checked only the marker center. That allowed an elongated
    capsule to be accepted when its center was inside the goal but the body was on the
    boundary. This function shrinks the valid goal box by the object's half footprint
    and by an explicit inward margin.
    """
    position = pose_or_position.get("position", pose_or_position)
    center = region["pose"]["position"]
    scale = region.get("scale") or region.get("size") or DEFAULT_TABLE_SIZE
    obj_half_x, obj_half_y = object_xy_half_extents(obj)
    allowed_half_x = float(scale["x"]) * 0.5 - obj_half_x - margin
    allowed_half_y = float(scale["y"]) * 0.5 - obj_half_y - margin
    if allowed_half_x < 0.0 or allowed_half_y < 0.0:
        return False
    return (
        abs(float(position["x"]) - float(center["x"])) <= allowed_half_x
        and abs(float(position["y"]) - float(center["y"])) <= allowed_half_y
    )


def object_fully_inside_box_region(
    obj: dict[str, Any], region: dict[str, Any], margin: float = 0.0
) -> bool:
    return object_pose_fully_inside_box_region(
        obj, obj["pose"]["position"], region, margin=margin
    )


def make_pose(
    position: dict[str, Any], orientation: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "position": {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        },
        "orientation": dict(orientation or VERTICAL_EEF_ORIENTATION),
    }


def ros_pose_from_dict(pose_dict: dict[str, Any] | None) -> Any:
    pose = Pose()
    if pose_dict is None:
        pose_dict = {
            "position": DEFAULT_TABLE_CENTER,
            "orientation": VERTICAL_EEF_ORIENTATION,
        }
    position = pose_dict.get("position", {})
    pose.position = Point(
        x=float(position.get("x", 0.0)),
        y=float(position.get("y", 0.0)),
        z=float(position.get("z", 0.0)),
    )
    # Keep the known stable vertical EEF quaternion.
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
    expected_min = height * step
    if len(raw) < expected_min:
        raise ValueError(
            f"Image data shorter than expected: got={len(raw)}, expected_at_least={expected_min}"
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


def image_data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


# =============================================================================
# Reporter and interactive input
# =============================================================================


class Reporter:
    def __init__(self) -> None:
        self.plain = PLAIN_OUTPUT or not RICH_AVAILABLE
        self.console = Console() if not self.plain else None

    def info(self, message: str) -> None:
        if self.console:
            self.console.print(message)
        else:
            print(message)

    def error(self, message: str) -> None:
        if self.console:
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

    def primitive(
        self, step: int, decision: PrimitiveDecision, queued_after: int
    ) -> None:
        rows: dict[str, Any] = {
            "step": step,
            "robot": decision.robot_id,
            "action": decision.action,
            "intent": decision.intent,
            "target_object": decision.target_object_id,
            "queued_after_this": queued_after,
            "reason": decision.reason,
        }
        if VERBOSE and decision.target_pose is not None:
            rows["target_pose"] = json.dumps(decision.target_pose, ensure_ascii=False)
        if VERBOSE and decision.metadata:
            rows["metadata"] = json.dumps(decision.metadata, ensure_ascii=False)
        self.status("Primitive action", rows)


def read_interactive_command(reporter: Reporter) -> str:
    examples = [
        "빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라",
        "파란 물체를 목표점에서 전부 밖으로 빼내라",
        "빨간 캡슐만 파란 목표점에 넣어라",
        "가장 가까운 파란 물체를 테이블 중앙으로 옮겨라",
        "모든 물체를 목표점 밖 안전한 빈 공간으로 옮겨라",
    ]
    reporter.info("\nEnter the task command for this run.")
    reporter.info("Examples:")
    for idx, example in enumerate(examples, start=1):
        reporter.info(f"  {idx}) {example}")
    reporter.info(
        "Type 'q' or 'quit' to exit. Press Enter to use the default command.\n"
    )

    while True:
        if RICH_AVAILABLE and not reporter.plain:
            command = Prompt.ask("Task command", default="")  # type: ignore[union-attr]
        else:
            command = input("Task command> ")
        command = command.strip()
        if command.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if command:
            return command
        if USE_DEFAULT_ON_EMPTY_INPUT:
            return DEFAULT_COMMAND
        reporter.error("Command cannot be empty.")


# =============================================================================
# ROS world node
# =============================================================================


class RosWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("llm_world_model_franka_cli")
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
            for goal_color, goal in goals.items():
                if point_in_box_region(obj["pose"]["position"], goal, margin=0.0):
                    obj["inside_goal"] = goal_color
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

    def snapshot_raw(self) -> tuple[dict[str, Any], bytes]:
        with self._lock:
            table = (
                dict(self._table)
                if self._table is not None
                else self._default_table_marker()
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
            raise RuntimeError("No top camera image has been received yet")
        image_bytes = image_msg_to_png_bytes(latest_image)
        raw = {
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
            "table_marker": table,
            "goal_markers": goals,
            "object_markers": sorted(objects.values(), key=lambda obj: obj["id"]),
            "robot_poses": robot_poses,
            "eef_poses": eef_poses,
        }
        return raw, image_bytes

    @staticmethod
    def _default_table_marker() -> dict[str, Any]:
        return {
            "id": "table:0",
            "namespace": "table",
            "marker_id": 0,
            "marker_type": "cube",
            "shape": "cube",
            "color": "red",
            "rgba": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
            "pose": {
                "position": DEFAULT_TABLE_CENTER,
                "orientation": VERTICAL_EEF_ORIENTATION,
            },
            "scale": DEFAULT_TABLE_SIZE,
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
            eef_position = eef_pose["position"]
            if xy_distance(eef_position, target_position) <= EEF_POSITION_TOLERANCE_M:
                if (
                    abs(float(eef_position["z"]) - float(target_position["z"]))
                    <= EEF_POSITION_TOLERANCE_M
                ):
                    return True
        return False


# =============================================================================
# State memory and symbolic world model
# =============================================================================


class ExecutionMemory:
    def __init__(self) -> None:
        self.held_objects: dict[str, str | None] = {"left": None, "right": None}
        self.pending_grip_targets: dict[str, str | None] = {"left": None, "right": None}
        # False by default on purpose: before final verification, the CLI must explicitly
        # command Homing for every robot in this run.
        self.robots_homed: dict[str, bool] = {"left": False, "right": False}
        self.history: list[dict[str, Any]] = []
        self.failed_targets: list[dict[str, Any]] = []
        self.last_verification: dict[str, Any] | None = None

    def object_held_by(self, object_id: str) -> str | None:
        for robot_id, held_id in self.held_objects.items():
            if held_id == object_id:
                return robot_id
        return None

    def all_robots_homed(self) -> bool:
        return all(
            bool(self.robots_homed.get(robot_id, False))
            for robot_id in CONTROL_SERVICE_TOPICS
        )

    def robots_not_homed(self) -> list[str]:
        return [
            robot_id
            for robot_id in FINAL_HOMING_ROBOT_ORDER
            if not self.robots_homed.get(robot_id, False)
        ]

    def append_result(self, result: ExecutionResult) -> None:
        decision = result.decision
        self.history.append(
            {
                "time": time.time(),
                "success": result.success,
                "message": result.message,
                "robot_id": decision.robot_id,
                "action": decision.action,
                "intent": decision.intent,
                "target_object_id": decision.target_object_id,
                "target_pose": decision.target_pose,
                "metadata": decision.metadata,
                "held_objects": dict(self.held_objects),
                "pending_grip_targets": dict(self.pending_grip_targets),
                "robots_homed": dict(self.robots_homed),
            }
        )
        if not result.success:
            self.failed_targets.append(
                {
                    "time": time.time(),
                    "robot_id": decision.robot_id,
                    "action": decision.action,
                    "intent": decision.intent,
                    "target_object_id": decision.target_object_id,
                    "target_pose": decision.target_pose,
                    "message": result.message,
                }
            )

    def as_llm_context(self) -> dict[str, Any]:
        return {
            "held_objects": dict(self.held_objects),
            "pending_grip_targets": dict(self.pending_grip_targets),
            "robots_homed": dict(self.robots_homed),
            "recent_history": self.history[-20:],
            "failed_targets": self.failed_targets[-10:],
            "last_verification": self.last_verification,
        }


class WorldModelBuilder:
    def __init__(self, memory: ExecutionMemory) -> None:
        self.memory = memory

    def build(self, raw: dict[str, Any]) -> dict[str, Any]:
        table_marker = raw["table_marker"]
        table_pose = table_marker.get("pose") or {
            "position": DEFAULT_TABLE_CENTER,
            "orientation": VERTICAL_EEF_ORIENTATION,
        }
        table_size = table_marker.get("scale") or DEFAULT_TABLE_SIZE
        table_center = dict(table_pose["position"])
        table_length = max(
            float(table_size.get("x", DEFAULT_TABLE_SIZE["x"])),
            float(table_size.get("y", DEFAULT_TABLE_SIZE["y"])),
        )
        reach_radius = REACH_RADIUS_TABLE_SCALE * table_length

        regions = self._build_regions(
            table_marker, raw["goal_markers"], raw["object_markers"]
        )
        safe_pose_candidates = self._build_safe_pose_candidates(
            regions, raw["object_markers"]
        )
        objects = self._build_objects(raw["object_markers"], regions)
        robots = self._build_robots(
            raw, objects, regions, safe_pose_candidates, reach_radius
        )

        world = {
            "timestamp_unix": raw["timestamp_unix"],
            "camera": raw.get("camera", {}),
            "inference_rules": {
                "can_reach_pose_rule": (
                    "A pose is reachable when XY distance from robot base to pose <= "
                    f"{REACH_RADIUS_TABLE_SCALE:.2f} * max(table.scale.x, table.scale.y). Z is ignored."
                ),
                "currently_holding_rule": "Holding state is inferred from successful Moving(pick), Grip, Realease history plus marker verification.",
                "coordinate_source_rule": "Use ROS2 MarkerArray geometry as metric ground truth; use image for qualitative layout only.",
            },
            "regions": regions,
            "safe_pose_candidates": safe_pose_candidates,
            "objects": objects,
            "robots": robots,
            "runtime_state": self.memory.as_llm_context(),
            "summary": self._summary(objects, robots),
        }
        return world

    def _build_regions(
        self,
        table_marker: dict[str, Any],
        goals: dict[str, Any],
        objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        table_pose = table_marker.get("pose") or {
            "position": DEFAULT_TABLE_CENTER,
            "orientation": VERTICAL_EEF_ORIENTATION,
        }
        table_size = table_marker.get("scale") or DEFAULT_TABLE_SIZE
        regions: dict[str, Any] = {
            "table": {
                "id": table_marker.get("id", "table:0"),
                "type": "support_surface",
                "pose": table_pose,
                "center": dict(table_pose["position"]),
                "scale": table_size,
                "safe_drop": True,
            }
        }
        for color, goal in goals.items():
            regions[f"{color}_goal"] = {
                "id": goal.get("id", f"{color}_goal"),
                "type": "goal",
                "color": color,
                "pose": goal["pose"],
                "center": dict(goal["pose"]["position"]),
                "scale": goal.get("scale", DEFAULT_TABLE_SIZE),
                "safe_drop": True,
            }
        # These computed regions are represented by pose candidates.
        regions["outside_goals"] = {
            "id": "outside_goals",
            "type": "computed_region",
            "description": "table area excluding red_goal and blue_goal",
            "safe_drop": True,
        }
        regions["safe_free_space"] = {
            "id": "safe_free_space",
            "type": "computed_region",
            "description": "collision-free table pose not inside any goal and not close to any object",
            "safe_drop": True,
        }
        object_z = self._representative_object_z(objects)
        center = dict(table_pose["position"])
        center["z"] = object_z + BUFFER_Z_OFFSET_M
        regions["table_center"] = {
            "id": "table_center",
            "type": "computed_point_region",
            "pose": make_pose(center),
            "center": center,
            "safe_drop": True,
        }
        regions["handover_buffer"] = {
            "id": "handover_buffer",
            "type": "shared_buffer",
            "description": "mutually reachable table-center buffer, with object-level z height",
            "pose": make_pose(center),
            "center": center,
            "safe_drop": True,
        }
        return regions

    def _build_objects(
        self, markers: list[dict[str, Any]], regions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for marker in markers:
            obj = dict(marker)
            current_region = self._current_region(marker, regions)
            obj["current_region"] = current_region
            obj["inside_goal"] = (
                current_region[:-5] if current_region.endswith("_goal") else None
            )
            obj["held_by"] = self.memory.object_held_by(obj["id"])
            obj["graspable"] = obj.get("shape") in SUPPORTED_SHAPES
            obj["stable_on_table"] = obj["held_by"] is None
            obj["distance_to_each_goal"] = {}
            for region_name, region in regions.items():
                if region.get("type") == "goal":
                    obj["distance_to_each_goal"][region["color"]] = round(
                        xy_distance(obj["pose"]["position"], region["center"]), 4
                    )
            objects.append(obj)
        return sorted(objects, key=lambda item: item["id"])

    def _build_robots(
        self,
        raw: dict[str, Any],
        objects: list[dict[str, Any]],
        regions: dict[str, Any],
        safe_pose_candidates: dict[str, list[dict[str, Any]]],
        reach_radius: float,
    ) -> dict[str, Any]:
        robots: dict[str, Any] = {}
        for robot_id in CONTROL_SERVICE_TOPICS:
            base_pose = raw.get("robot_poses", {}).get(robot_id)
            eef_pose = raw.get("eef_poses", {}).get(robot_id)
            reachable_objects: list[str] = []
            object_xy_distances: dict[str, float] = {}
            reachable_goals: list[str] = []
            reachable_regions: list[str] = []
            if base_pose:
                base_position = base_pose["position"]
                for obj in objects:
                    distance = xy_distance(base_position, obj["pose"]["position"])
                    object_xy_distances[obj["id"]] = round(distance, 4)
                    if distance <= reach_radius:
                        reachable_objects.append(obj["id"])
                for region_name, region in regions.items():
                    if region.get("type") == "goal":
                        if self._robot_can_reach_region_candidate(
                            base_position,
                            safe_pose_candidates.get(f"{region['color']}_goal", []),
                            reach_radius,
                        ):
                            reachable_goals.append(region_name)
                            reachable_regions.append(region_name)
                    elif region_name in {
                        "outside_goals",
                        "safe_free_space",
                        "table_center",
                        "handover_buffer",
                    }:
                        if self._robot_can_reach_region_candidate(
                            base_position,
                            safe_pose_candidates.get(region_name, []),
                            reach_radius,
                        ):
                            reachable_regions.append(region_name)
            robots[robot_id] = {
                "scene_label": ROBOT_SCENE_LABELS[robot_id],
                "base_pose": base_pose,
                "end_effector_pose": eef_pose,
                "reach_radius": reach_radius,
                "reachable_objects": reachable_objects,
                "reachable_goals": sorted_unique(reachable_goals),
                "reachable_regions": sorted_unique(reachable_regions),
                "object_xy_distances": object_xy_distances,
                "currently_holding": self.memory.held_objects.get(robot_id),
                "pending_grip_target": self.memory.pending_grip_targets.get(robot_id),
                "capabilities": {
                    "primitive_actions": sorted(ALLOWED_PRIMITIVE_ACTIONS),
                    "gripper": "parallel_gripper",
                    "can_grasp_shapes": sorted(SUPPORTED_SHAPES),
                },
            }
        return robots

    def _build_safe_pose_candidates(
        self, regions: dict[str, Any], objects: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        candidates: dict[str, list[dict[str, Any]]] = {}
        for region_name, region in regions.items():
            if region.get("type") == "goal":
                candidates[f"{region['color']}_goal"] = self._goal_drop_candidates(
                    region, objects
                )
        outside = self._outside_goal_candidates(regions, objects)
        candidates["outside_goals"] = outside
        candidates["safe_free_space"] = list(outside)
        table_center_pose = regions.get("table_center", {}).get("pose")
        handover_pose = regions.get("handover_buffer", {}).get("pose")
        candidates["table_center"] = [table_center_pose] if table_center_pose else []
        candidates["handover_buffer"] = [handover_pose] if handover_pose else []
        return candidates

    def _goal_drop_candidates(
        self, goal: dict[str, Any], objects: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        center = goal["center"]
        scale = goal.get("scale", DEFAULT_TABLE_SIZE)
        half_x = max(0.0, float(scale.get("x", 0.1)) * 0.5 - GOAL_MARGIN_M)
        half_y = max(0.0, float(scale.get("y", 0.1)) * 0.5 - GOAL_MARGIN_M)
        z = float(center["z"]) + GOAL_Z_OFFSET_M
        points: list[dict[str, float]] = []
        for ix in range(5):
            for iy in range(5):
                points.append(
                    {
                        "x": float(center["x"]) - half_x + 2.0 * half_x * ix / 4.0,
                        "y": float(center["y"]) - half_y + 2.0 * half_y * iy / 4.0,
                        "z": z,
                    }
                )
        points.append({"x": float(center["x"]), "y": float(center["y"]), "z": z})
        occupied = [
            obj
            for obj in objects
            if point_in_box_region(obj["pose"]["position"], goal, margin=0.0)
        ]
        points.sort(
            key=lambda p: (
                self._min_dist_to_objects(p, occupied),
                -xy_distance(p, center),
            ),
            reverse=True,
        )
        return [
            make_pose(point)
            for point in points
            if self._clear_of_objects(point, occupied, DROP_CLEARANCE_M)
        ]

    def _outside_goal_candidates(
        self, regions: dict[str, Any], objects: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        table = regions["table"]
        center = table["center"]
        scale = table.get("scale", DEFAULT_TABLE_SIZE)
        half_x = max(
            0.0,
            float(scale.get("x", DEFAULT_TABLE_SIZE["x"])) * 0.5 - BUFFER_CLEARANCE_M,
        )
        half_y = max(
            0.0,
            float(scale.get("y", DEFAULT_TABLE_SIZE["y"])) * 0.5 - BUFFER_CLEARANCE_M,
        )
        object_z = self._representative_object_z(objects)
        z = object_z + OUTSIDE_GOAL_Z_OFFSET_M
        points: list[dict[str, float]] = []
        for ix in range(9):
            for iy in range(9):
                point = {
                    "x": float(center["x"]) - half_x + 2.0 * half_x * ix / 8.0,
                    "y": float(center["y"]) - half_y + 2.0 * half_y * iy / 8.0,
                    "z": z,
                }
                if not self._point_inside_any_goal(point, regions):
                    points.append(point)
        points.sort(key=lambda p: self._min_dist_to_objects(p, objects), reverse=True)
        return [
            make_pose(point)
            for point in points
            if self._clear_of_objects(point, objects, DROP_CLEARANCE_M)
        ]

    def _current_region(self, obj: dict[str, Any], regions: dict[str, Any]) -> str:
        position = obj["pose"]["position"]
        for region_name, region in regions.items():
            if region.get("type") == "goal" and object_fully_inside_box_region(
                obj, region, margin=GOAL_INTERIOR_MARGIN_M
            ):
                return region_name
        handover = regions.get("handover_buffer", {}).get("pose")
        if (
            handover
            and xy_distance(position, handover["position"]) <= VERIFY_XY_TOLERANCE_M
        ):
            return "handover_buffer"
        return "outside_goals"

    def _point_inside_any_goal(
        self, point: dict[str, float], regions: dict[str, Any]
    ) -> bool:
        return any(
            region.get("type") == "goal"
            and point_in_box_region(point, region, margin=GOAL_MARGIN_M)
            for region in regions.values()
        )

    def _representative_object_z(self, objects: list[dict[str, Any]]) -> float:
        z_values = [
            float(obj["pose"]["position"]["z"]) for obj in objects if obj.get("pose")
        ]
        if z_values:
            return sorted(z_values)[len(z_values) // 2]
        return DEFAULT_TABLE_CENTER["z"]

    @staticmethod
    def _min_dist_to_objects(
        point: dict[str, float], objects: list[dict[str, Any]]
    ) -> float:
        if not objects:
            return float("inf")
        return min(xy_distance(point, obj["pose"]["position"]) for obj in objects)

    @staticmethod
    def _clear_of_objects(
        point: dict[str, float], objects: list[dict[str, Any]], clearance: float
    ) -> bool:
        return all(
            xy_distance(point, obj["pose"]["position"]) >= clearance for obj in objects
        )

    @staticmethod
    def _robot_can_reach_region_candidate(
        base_position: dict[str, float],
        candidates: list[dict[str, Any]],
        reach_radius: float,
    ) -> bool:
        return any(
            xy_distance(base_position, candidate["position"]) <= reach_radius
            for candidate in candidates
            if candidate
        )

    @staticmethod
    def _summary(
        objects: list[dict[str, Any]], robots: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "object_count": len(objects),
            "objects_by_region": {
                region: [
                    obj["id"] for obj in objects if obj.get("current_region") == region
                ]
                for region in sorted_unique(
                    obj.get("current_region", "unknown") for obj in objects
                )
            },
            "held_objects": {
                robot_id: robot.get("currently_holding")
                for robot_id, robot in robots.items()
            },
            "reachable_objects": {
                robot_id: robot.get("reachable_objects", [])
                for robot_id, robot in robots.items()
            },
            "reachable_regions": {
                robot_id: robot.get("reachable_regions", [])
                for robot_id, robot in robots.items()
            },
        }


# =============================================================================
# LLM semantic planner
# =============================================================================


class LLMSemanticPlanner:
    def __init__(self, reporter: Reporter) -> None:
        self.reporter = reporter

    def plan(
        self,
        command: str,
        world_model: dict[str, Any],
        image_bytes: bytes | None,
        feedback: str = "",
    ) -> LLMPlan:
        errors: list[str] = []
        modes = [LLM_BACKEND] if LLM_BACKEND != "auto" else ["openai", "ollama"]
        for mode in modes:
            try:
                if mode == "openai":
                    return self._call_openai(
                        command, world_model, image_bytes, feedback
                    )
                if mode == "ollama":
                    return self._call_ollama(
                        command, world_model, image_bytes, feedback
                    )
                errors.append(f"unsupported LLM_BACKEND={mode!r}")
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
                if LLM_BACKEND != "auto":
                    break
        raise RuntimeError("LLM semantic planning failed. " + " | ".join(errors))

    def _prompt(
        self, command: str, world_model: dict[str, Any], feedback: str
    ) -> tuple[str, str]:
        system = (
            "You are an expert multi-agent robotic task planner for two Franka arms in Isaac Sim. "
            "Your job is to understand a natural-language command and convert it into MCP-like semantic tool calls. "
            "You must use the symbolic world model as metric ground truth. The image, if provided, is only for qualitative spatial confirmation. "
            "The top-view image cannot reliably show gripper open/closed state or whether an object is held; do not infer grasp state from the image. "
            "Do not output ROS primitives such as Moving, Grip, Realease, or Homing. The runtime will decompose semantic tools into primitives. "
            "Return only one valid JSON object.\n\n"
            "Available semantic tools:\n"
            "1) move_objects\n"
            "   Use this for moving selected objects to a supported destination region.\n"
            "   arguments schema:\n"
            "   {\n"
            '     "object_selector": {\n'
            '       "ids": [string] optional,\n'
            '       "color": "red"|"blue"|"any",\n'
            '       "shape": "cube"|"sphere"|"capsule"|"any",\n'
            '       "current_region": "anywhere"|"inside_any_goal"|"inside_red_goal"|"inside_blue_goal"|"outside_goals"|"handover_buffer"\n'
            "     },\n"
            '     "destination": {\n'
            '       "type": "goal"|"outside_goals"|"safe_free_space"|"table_center"|"handover_buffer",\n'
            '       "goal_color": "red"|"blue" only when type is goal\n'
            "     },\n"
            '     "ordering": "nearest_first"|"left_to_right"|"right_to_left"|"as_listed",\n'
            '     "handover_policy": "use_if_needed"\n'
            "   }\n"
            "2) reject_task\n"
            "   Use this only when the command is unsafe, ambiguous, references unknown objects/regions/colors/shapes, or asks to move outside the safe table workspace.\n"
            "   Do NOT use reject_task for reachability limitations, because the runtime computes direct and handover routes after your semantic extraction.\n"
            '   arguments schema: {"reason": string}\n\n'
            "Supported colors: red, blue. Supported shapes: cube, sphere, capsule.\n"
            "Supported destination types: goal, outside_goals, safe_free_space, table_center, handover_buffer.\n"
            "Division of responsibility:\n"
            "- Your job: extract the semantic intent: which objects, optional source region, destination region, ordering, and constraints.\n"
            "- Runtime job: compute source robot, destination robot, direct route, handover route, exact poses, and feasibility.\n"
            "- Never reject because one robot cannot reach both the object and destination. That is exactly when handover may be required.\n"
            "- If the object is on the left side and the goal is on the right side, still output move_objects with handover_policy=use_if_needed.\n"
            "Safety rules:\n"
            "- Do not allow off-table drops. If user asks to drop outside the table, reject_task.\n"
            "- Do not invent object IDs, regions, colors, or shapes.\n"
            "- If the command says 'goal 밖', '목표점 밖', '밖으로 빼내라', use destination.type=outside_goals, not a colored goal.\n"
            "- If the command says '테이블 중앙', use destination.type=table_center.\n"
            "- If the command says '빈 공간', '안전한 공간', use destination.type=safe_free_space.\n"
            "- If the command specifies color+shape, put both in object_selector.\n\n"
            "Examples:\n"
            "- '빨간 캡슐만 파란 목표점에 넣어라' => accepted move_objects with object_selector.color=red, shape=capsule, current_region=anywhere, destination.type=goal, destination.goal_color=blue. Do not reject for reachability; runtime will handover if needed.\n"
            "- '빨간 물체를 전부 파란 목표점에 넣어라' => accepted move_objects with object_selector.color=red, shape=any, destination.type=goal, destination.goal_color=blue.\n"
            "- '파란 물체를 목표점에서 전부 밖으로 빼내라' => accepted move_objects with object_selector.color=blue, current_region=inside_any_goal, destination.type=outside_goals.\n"
            "- '빨간 캡슐을 테이블 밖으로 떨어트려라' => rejected because off-table drop is unsafe.\n\n"
            "Output schema:\n"
            "{\n"
            '  "status": "accepted"|"rejected",\n'
            '  "reasoning_summary": string,\n'
            '  "plan": [ {"tool": "move_objects", "arguments": {...}, "reason": string} ],\n'
            '  "goal_conditions": [ optional high-level success conditions ],\n'
            '  "rejection_reason": string optional\n'
            "}\n"
        )
        compact_world = self._compact_world_for_llm(world_model)
        user = json.dumps(
            {
                "user_command": command,
                "world_model": compact_world,
                "execution_feedback": feedback,
                "required_response": "Return one JSON object using the output schema. Extract semantic intent with move_objects whenever possible. Use reject_task only for unsafe, unknown, or ambiguous commands; never reject because a handover may be required.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return system, user

    def _compact_world_for_llm(self, world: dict[str, Any]) -> dict[str, Any]:
        return {
            "inference_rules": world.get("inference_rules"),
            "regions": {
                name: {
                    "type": region.get("type"),
                    "color": region.get("color"),
                    "center": region.get("center"),
                    "scale": region.get("scale"),
                    "description": region.get("description"),
                    "safe_drop": region.get("safe_drop"),
                }
                for name, region in world.get("regions", {}).items()
            },
            "objects": [
                {
                    "id": obj.get("id"),
                    "color": obj.get("color"),
                    "shape": obj.get("shape"),
                    "current_region": obj.get("current_region"),
                    "held_by": obj.get("held_by"),
                    "graspable": obj.get("graspable"),
                    "position": obj.get("pose", {}).get("position"),
                }
                for obj in world.get("objects", [])
            ],
            "robots": {
                robot_id: {
                    "scene_label": robot.get("scene_label"),
                    "currently_holding": robot.get("currently_holding"),
                    "reachable_objects": robot.get("reachable_objects"),
                    "reachable_regions": robot.get("reachable_regions"),
                    "reachable_goals": robot.get("reachable_goals"),
                    "reach_radius": robot.get("reach_radius"),
                }
                for robot_id, robot in world.get("robots", {}).items()
            },
            "route_capabilities": self._route_capabilities_for_llm(world),
            "summary": world.get("summary"),
            "runtime_state": world.get("runtime_state"),
        }

    def _route_capabilities_for_llm(
        self, world: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Summarize direct/handover route options so the LLM does not guess feasibility.

        This is advisory context only. The runtime recomputes route feasibility before
        executing, but the summary helps the LLM avoid rejecting cross-workspace tasks.
        """
        destinations = [
            "red_goal",
            "blue_goal",
            "outside_goals",
            "safe_free_space",
            "table_center",
            "handover_buffer",
        ]
        rows: list[dict[str, Any]] = []
        for obj in world.get("objects", []):
            source_robots = []
            held_by = obj.get("held_by")
            if held_by:
                source_robots = [held_by]
            else:
                for robot_id, robot in world.get("robots", {}).items():
                    if obj.get("id") in set(robot.get("reachable_objects") or []):
                        source_robots.append(robot_id)
            dest_rows: dict[str, Any] = {}
            for dest in destinations:
                candidates = world.get("safe_pose_candidates", {}).get(dest, [])
                if not candidates:
                    continue
                destination_robots = [
                    robot_id
                    for robot_id in world.get("robots", {})
                    if any(
                        self._robot_can_reach_pose_for_llm(world, robot_id, pose)
                        for pose in candidates
                    )
                ]
                direct_pairs = [
                    robot_id
                    for robot_id in source_robots
                    if robot_id in destination_robots
                ]
                handover_pairs: list[dict[str, str]] = []
                if not direct_pairs and source_robots and destination_robots:
                    for src in source_robots:
                        for dst in destination_robots:
                            if src == dst:
                                continue
                            if self._robots_share_handover_for_llm(world, src, dst):
                                handover_pairs.append(
                                    {
                                        "source_robot": src,
                                        "destination_robot": dst,
                                        "handover_region": "handover_buffer",
                                    }
                                )
                dest_rows[dest] = {
                    "destination_robots": destination_robots,
                    "direct_robots": direct_pairs,
                    "handover_pairs": handover_pairs,
                    "route_possible": bool(direct_pairs or handover_pairs),
                    "note": "If route_possible is true, output move_objects; do not reject because one robot alone cannot do the whole route.",
                }
            rows.append(
                {
                    "object_id": obj.get("id"),
                    "color": obj.get("color"),
                    "shape": obj.get("shape"),
                    "current_region": obj.get("current_region"),
                    "source_robots": source_robots,
                    "destinations": dest_rows,
                }
            )
        return rows

    @staticmethod
    def _robot_can_reach_pose_for_llm(
        world: dict[str, Any], robot_id: str, pose: dict[str, Any]
    ) -> bool:
        robot = world.get("robots", {}).get(robot_id, {})
        base_pose = robot.get("base_pose")
        if not base_pose or not pose:
            return False
        return xy_distance(base_pose["position"], pose["position"]) <= float(
            robot.get("reach_radius") or 0.0
        )

    def _robots_share_handover_for_llm(
        self, world: dict[str, Any], src: str, dst: str
    ) -> bool:
        candidates = world.get("safe_pose_candidates", {}).get("handover_buffer", [])
        if not candidates:
            candidates = world.get("safe_pose_candidates", {}).get("table_center", [])
        return any(
            self._robot_can_reach_pose_for_llm(world, src, pose)
            and self._robot_can_reach_pose_for_llm(world, dst, pose)
            for pose in candidates
        )

    def _call_openai(
        self,
        command: str,
        world_model: dict[str, Any],
        image_bytes: bytes | None,
        feedback: str,
    ) -> LLMPlan:
        api_key = (
            os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
        ).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or CHATGPT_API_KEY is not set")
        system, user = self._prompt(command, world_model, feedback)
        content: Any = [{"type": "text", "text": user}]
        if SEND_IMAGE_TO_LLM and image_bytes:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url(image_bytes)}}
            )
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        response = post_json(
            OPENAI_API_URL, payload, {"Authorization": f"Bearer {api_key}"}
        )
        choices = response.get("choices") or []
        message = (
            choices[0].get("message")
            if choices and isinstance(choices[0], dict)
            else {}
        )
        parsed = extract_json_object(str((message or {}).get("content") or ""))
        return self._parse_plan(parsed)

    def _call_ollama(
        self,
        command: str,
        world_model: dict[str, Any],
        image_bytes: bytes | None,
        feedback: str,
    ) -> LLMPlan:
        system, user = self._prompt(command, world_model, feedback)
        message: dict[str, Any] = {"role": "user", "content": user}
        if SEND_IMAGE_TO_LLM and image_bytes:
            message["images"] = [base64.b64encode(image_bytes).decode("utf-8")]
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                message,
            ],
            "format": "json",
            "options": {"temperature": LLM_TEMPERATURE},
        }
        response = post_json(OLLAMA_URL, payload, {})
        msg = response.get("message") or {}
        parsed = extract_json_object(
            str(msg.get("content") or response.get("response") or "")
        )
        return self._parse_plan(parsed)

    def _parse_plan(self, parsed: dict[str, Any]) -> LLMPlan:
        status = str(parsed.get("status") or "").strip().lower()
        if status not in {"accepted", "rejected"}:
            # Accept a single tool-call shorthand.
            if str(parsed.get("tool") or "") == "reject_task":
                status = "rejected"
            elif parsed.get("tool"):
                status = "accepted"
            else:
                raise ValueError(f"LLM plan lacks valid status: {parsed}")
        rejection_reason = str(parsed.get("rejection_reason") or "").strip()
        reasoning = str(
            parsed.get("reasoning_summary") or parsed.get("reason") or ""
        ).strip()
        if status == "rejected":
            if not rejection_reason:
                args = (
                    parsed.get("arguments")
                    if isinstance(parsed.get("arguments"), dict)
                    else {}
                )
                rejection_reason = str(
                    (args or {}).get("reason") or reasoning or "task rejected by LLM"
                ).strip()
            return LLMPlan("rejected", reasoning, [], [], parsed, rejection_reason)

        raw_plan = parsed.get("plan")
        if raw_plan is None and parsed.get("tool"):
            raw_plan = [parsed]
        if not isinstance(raw_plan, list) or not raw_plan:
            raise ValueError(
                f"accepted LLM plan must include non-empty plan list: {parsed}"
            )
        tool_calls: list[SemanticToolCall] = []
        for item in raw_plan:
            if not isinstance(item, dict):
                raise ValueError(f"plan item must be object: {item}")
            tool = str(item.get("tool") or item.get("name") or "").strip()
            arguments = item.get("arguments") or item.get("args") or {}
            if not isinstance(arguments, dict):
                raise ValueError(f"tool arguments must be object: {item}")
            reason = str(item.get("reason") or "").strip()
            tool_calls.append(SemanticToolCall(tool, arguments, reason))
        return LLMPlan("accepted", reasoning, tool_calls, [], parsed)


# =============================================================================
# Plan validation and semantic runtime
# =============================================================================


class PlanValidator:
    def validate(self, plan: LLMPlan, world: dict[str, Any]) -> LLMPlan:
        if plan.status == "rejected":
            raise ValueError(plan.rejection_reason or "Task was rejected by LLM")
        if not plan.tool_calls:
            raise ValueError("LLM plan has no tool calls")
        for call in plan.tool_calls:
            if call.tool == "reject_task":
                reason = str(
                    call.arguments.get("reason") or call.reason or "task rejected"
                )
                raise ValueError(reason)
            if call.tool != "move_objects":
                raise ValueError(f"Unsupported semantic tool: {call.tool!r}")
            self._validate_move_objects(call, world)
        return plan

    def _validate_move_objects(
        self, call: SemanticToolCall, world: dict[str, Any]
    ) -> None:
        selector = self._selector(call)
        destination = self._destination(call)

        if selector.get("color", "any") not in SUPPORTED_COLORS | {"any"}:
            raise ValueError(
                f"Unsupported object color selector: {selector.get('color')!r}"
            )
        if selector.get("shape", "any") not in SUPPORTED_SHAPES | {"any"}:
            raise ValueError(
                f"Unsupported object shape selector: {selector.get('shape')!r}"
            )
        if selector.get("current_region", "anywhere") not in SUPPORTED_CURRENT_REGIONS:
            raise ValueError(
                f"Unsupported current_region selector: {selector.get('current_region')!r}"
            )

        ids = selector.get("ids") or []
        if ids:
            known_ids = {obj["id"] for obj in world.get("objects", [])}
            missing = sorted(set(str(item) for item in ids) - known_ids)
            if missing:
                raise ValueError(f"Selector references unknown object ids: {missing}")

        dest_type = destination.get("type")
        if dest_type == "off_table" and not ALLOW_OFF_TABLE_DESTINATION:
            raise ValueError("Off-table destination is unsafe and disabled")
        if dest_type not in SUPPORTED_DESTINATION_TYPES:
            raise ValueError(f"Unsupported destination type: {dest_type!r}")
        if dest_type == "goal":
            goal_color = destination.get("goal_color")
            if goal_color not in SUPPORTED_COLORS:
                raise ValueError(
                    f"goal destination requires goal_color red|blue, got {goal_color!r}"
                )
            if f"{goal_color}_goal" not in world.get("regions", {}):
                raise ValueError(
                    f"Requested goal does not exist in world: {goal_color}_goal"
                )

        matched = select_objects(world, selector)
        # It is not an error for source-filtered tasks to match none; verifier may immediately finish.
        if not matched:
            if not ids:
                return
            raise ValueError(f"Object selector matched no objects: {selector}")

    @staticmethod
    def _selector(call: SemanticToolCall) -> dict[str, Any]:
        selector = call.arguments.get("object_selector") or {}
        if not isinstance(selector, dict):
            raise ValueError("object_selector must be an object")
        return normalize_selector(selector)

    @staticmethod
    def _destination(call: SemanticToolCall) -> dict[str, Any]:
        destination = call.arguments.get("destination") or {}
        if not isinstance(destination, dict):
            raise ValueError("destination must be an object")
        return normalize_destination(destination)


def normalize_selector(selector: dict[str, Any]) -> dict[str, Any]:
    ids_raw = selector.get("ids") or selector.get("object_ids") or []
    if isinstance(ids_raw, str):
        ids = [ids_raw]
    elif isinstance(ids_raw, list):
        ids = [str(item) for item in ids_raw if str(item).strip()]
    else:
        ids = []
    color = str(selector.get("color") or "any").strip().lower()
    shape = str(selector.get("shape") or "any").strip().lower()
    current_region = str(selector.get("current_region") or "anywhere").strip().lower()
    # Normalize common variants.
    if current_region in {"any", "all"}:
        current_region = "anywhere"
    if current_region in {"inside_goal", "inside_goals"}:
        current_region = "inside_any_goal"
    if current_region == "red_goal":
        current_region = "inside_red_goal"
    if current_region == "blue_goal":
        current_region = "inside_blue_goal"
    return {
        "ids": ids,
        "color": color,
        "shape": shape,
        "current_region": current_region,
    }


def normalize_destination(destination: dict[str, Any]) -> dict[str, Any]:
    dest_type = (
        str(destination.get("type") or destination.get("region") or "").strip().lower()
    )
    if dest_type in {"red_goal", "blue_goal"}:
        goal_color = dest_type.split("_", 1)[0]
        dest_type = "goal"
    else:
        goal_color = (
            str(destination.get("goal_color") or destination.get("color") or "")
            .strip()
            .lower()
            or None
        )
    if dest_type in {"outside_goal", "outside_goal_regions", "outside"}:
        dest_type = "outside_goals"
    if dest_type in {"free_space", "empty_space", "safe_space"}:
        dest_type = "safe_free_space"
    if dest_type in {"center", "table centre", "table_middle", "table_mid"}:
        dest_type = "table_center"
    return {"type": dest_type, "goal_color": goal_color}


def select_objects(
    world: dict[str, Any], selector: dict[str, Any]
) -> list[dict[str, Any]]:
    selector = normalize_selector(selector)
    ids = set(selector.get("ids") or [])
    color = selector.get("color", "any")
    shape = selector.get("shape", "any")
    current_region = selector.get("current_region", "anywhere")
    selected: list[dict[str, Any]] = []
    for obj in world.get("objects", []):
        if ids and obj["id"] not in ids:
            continue
        if color != "any" and obj.get("color") != color:
            continue
        if shape != "any" and obj.get("shape") != shape:
            continue
        if not object_matches_region_selector(obj, current_region):
            continue
        selected.append(obj)
    return selected


def object_matches_region_selector(obj: dict[str, Any], current_region: str) -> bool:
    region = obj.get("current_region")
    if current_region == "anywhere":
        return True
    if current_region == "inside_any_goal":
        return region in {"red_goal", "blue_goal"}
    if current_region == "inside_red_goal":
        return region == "red_goal"
    if current_region == "inside_blue_goal":
        return region == "blue_goal"
    return region == current_region


class SemanticRuntime:
    def __init__(self, memory: ExecutionMemory, reporter: Reporter) -> None:
        self.memory = memory
        self.reporter = reporter
        self.plan: LLMPlan | None = None
        self.primitive_queue: list[PrimitiveDecision] = []
        self.active_object_id: str | None = None
        self.concrete_goal_conditions: list[SemanticGoalCondition] = []

    def load_plan(self, plan: LLMPlan, world: dict[str, Any]) -> None:
        self.plan = plan
        self.primitive_queue.clear()
        self.active_object_id = None
        self.concrete_goal_conditions = self._derive_goal_conditions(plan, world)

    def clear_plan(self) -> None:
        self.primitive_queue.clear()
        self.active_object_id = None

    def _next_required_final_homing_decision(self) -> PrimitiveDecision | None:
        if not FINAL_VERIFY_REQUIRES_ALL_ROBOTS_HOMED:
            return None
        for robot_id in FINAL_HOMING_ROBOT_ORDER:
            if not self.memory.robots_homed.get(robot_id, False):
                return PrimitiveDecision(
                    robot_id=robot_id,
                    action=ACTION_HOMING,
                    target_pose=None,
                    target_object_id=None,
                    intent="final_home",
                    reason="Home this robot before final task verification.",
                    metadata={"final_homing_before_verification": True},
                )
        return None

    def next_decision(self, world: dict[str, Any]) -> PrimitiveDecision:
        if self.primitive_queue:
            return self.primitive_queue.pop(0)
        if self.plan is None:
            raise RuntimeError("No semantic plan loaded")
        task = self._select_next_object_task(world)
        if task is None:
            final_homing = self._next_required_final_homing_decision()
            if final_homing is not None:
                return final_homing
            return PrimitiveDecision(
                robot_id=None,
                action=ACTION_HOMING,
                intent="finish",
                done=True,
                reason="All semantic goal conditions appear satisfied and all robots are homed; final verifier will confirm.",
            )
        self.active_object_id = task.object_id
        self.primitive_queue = self._decompose_task(world, task)
        if not self.primitive_queue:
            raise RuntimeError(
                f"Generated empty primitive queue for object {task.object_id}"
            )
        return self.primitive_queue.pop(0)

    def _derive_goal_conditions(
        self, plan: LLMPlan, world: dict[str, Any]
    ) -> list[SemanticGoalCondition]:
        conditions: list[SemanticGoalCondition] = []
        for call in plan.tool_calls:
            if call.tool != "move_objects":
                continue
            selector = normalize_selector(call.arguments.get("object_selector") or {})
            destination = normalize_destination(call.arguments.get("destination") or {})
            selected = select_objects(world, selector)
            ids = [obj["id"] for obj in selected]
            if not ids:
                # For no-match commands, derive a broad condition from the selector so finish can still be meaningful.
                continue
            dest_type = destination["type"]
            if dest_type == "goal":
                conditions.append(
                    SemanticGoalCondition(
                        "objects_in_goal",
                        ids,
                        required_region=f"{destination['goal_color']}_goal",
                        required_goal_color=destination["goal_color"],
                        description=f"selected objects must be inside {destination['goal_color']}_goal",
                    )
                )
            elif dest_type in {"outside_goals", "safe_free_space"}:
                conditions.append(
                    SemanticGoalCondition(
                        "objects_outside_goals",
                        ids,
                        required_region="outside_goals",
                        description="selected objects must not be inside red_goal or blue_goal",
                    )
                )
            elif dest_type in {"table_center", "handover_buffer"}:
                conditions.append(
                    SemanticGoalCondition(
                        "objects_near_region_pose",
                        ids,
                        required_region=dest_type,
                        description=f"selected objects must be near {dest_type}",
                    )
                )
        return conditions

    def _select_next_object_task(self, world: dict[str, Any]) -> ObjectTask | None:
        if self.plan is None:
            return None
        candidates: list[ObjectTask] = []
        rank = 0
        for call in self.plan.tool_calls:
            selector = normalize_selector(call.arguments.get("object_selector") or {})
            destination = normalize_destination(call.arguments.get("destination") or {})
            objects = self._order_objects(
                select_objects(world, selector),
                call.arguments.get("ordering") or "nearest_first",
                world,
            )
            for obj in objects:
                if self._object_satisfies_destination(obj, destination, world):
                    continue
                route = self._choose_route(obj, destination, world)
                if route is None:
                    continue
                rank += 1
                candidates.append(
                    ObjectTask(
                        object_id=obj["id"],
                        route=route,
                        source_tool_call=call,
                        destination=destination,
                        rank=rank,
                        rationale=f"Process {obj['id']} using {route.route_type}: {route.reason}",
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda task: (task.route.score, task.rank, task.object_id))
        return candidates[0]

    def _order_objects(
        self, objects: list[dict[str, Any]], ordering: str, world: dict[str, Any]
    ) -> list[dict[str, Any]]:
        ordering = str(ordering or "nearest_first").strip().lower()
        if ordering == "left_to_right":
            return sorted(objects, key=lambda obj: float(obj["pose"]["position"]["x"]))
        if ordering == "right_to_left":
            return sorted(objects, key=lambda obj: -float(obj["pose"]["position"]["x"]))
        if ordering == "as_listed":
            return list(objects)

        # nearest_first by nearest robot base distance.
        def nearest_robot_distance(obj: dict[str, Any]) -> float:
            distances: list[float] = []
            for robot in world.get("robots", {}).values():
                base_pose = robot.get("base_pose")
                if base_pose:
                    distances.append(
                        xy_distance(base_pose["position"], obj["pose"]["position"])
                    )
            return min(distances) if distances else float("inf")

        return sorted(objects, key=lambda obj: (nearest_robot_distance(obj), obj["id"]))

    def _object_satisfies_destination(
        self, obj: dict[str, Any], destination: dict[str, Any], world: dict[str, Any]
    ) -> bool:
        dest_type = destination["type"]
        if dest_type == "goal":
            goal = world.get("regions", {}).get(f"{destination['goal_color']}_goal")
            return bool(
                goal
                and object_fully_inside_box_region(
                    obj, goal, margin=GOAL_INTERIOR_MARGIN_M
                )
            )
        if dest_type in {"outside_goals", "safe_free_space"}:
            return obj.get("current_region") == "outside_goals"
        if dest_type in {"table_center", "handover_buffer"}:
            pose = world.get("regions", {}).get(dest_type, {}).get("pose")
            return bool(
                pose
                and xy_distance(obj["pose"]["position"], pose["position"])
                <= VERIFY_XY_TOLERANCE_M
            )
        return False

    def _choose_route(
        self, obj: dict[str, Any], destination: dict[str, Any], world: dict[str, Any]
    ) -> RouteCandidate | None:
        drop_pose = self._select_drop_pose(obj, destination, world)
        if drop_pose is None:
            return None

        held_by = obj.get("held_by") or self.memory.object_held_by(obj["id"])
        source_robots = (
            [held_by]
            if held_by
            else self._robots_that_can_reach_pose(world, obj["pose"])
        )
        destination_robots = self._robots_that_can_reach_pose(world, drop_pose)
        if not source_robots or not destination_robots:
            return None

        # If already held, the source robot is fixed.
        for src in source_robots:
            for dst in destination_robots:
                if src == dst:
                    score = self._route_score(world, obj, drop_pose, src, dst)
                    return RouteCandidate(
                        obj["id"],
                        "direct_held" if held_by else "direct",
                        src,
                        dst,
                        drop_pose,
                        score,
                        reason=f"robot {src} can reach both source and destination",
                    )

        buffer_pose = self._select_handover_buffer_pose(
            obj, world, source_robots, destination_robots
        )
        if buffer_pose is None:
            return None
        best: RouteCandidate | None = None
        for src in source_robots:
            if not self._robot_can_reach_pose(world, src, buffer_pose):
                continue
            for dst in destination_robots:
                if not self._robot_can_reach_pose(world, dst, buffer_pose):
                    continue
                score = self._route_score(world, obj, drop_pose, src, dst) + 1.0
                candidate = RouteCandidate(
                    obj["id"],
                    "handover",
                    src,
                    dst,
                    drop_pose,
                    score,
                    buffer_pose=buffer_pose,
                    reason=f"handover via shared buffer because source robot {src} and destination robot {dst} differ",
                )
                if best is None or candidate.score < best.score:
                    best = candidate
        return best

    def _select_drop_pose(
        self, obj: dict[str, Any], destination: dict[str, Any], world: dict[str, Any]
    ) -> dict[str, Any] | None:
        dest_type = destination["type"]
        if dest_type == "goal":
            goal_region = world.get("regions", {}).get(
                f"{destination['goal_color']}_goal"
            )
            if not goal_region:
                return None
            candidates = self._goal_drop_candidates_for_object(obj, goal_region, world)
            score_fn = lambda candidate: self._goal_drop_score(
                obj, candidate, goal_region, world
            )
        elif dest_type in {
            "outside_goals",
            "safe_free_space",
            "table_center",
            "handover_buffer",
        }:
            candidates = world.get("safe_pose_candidates", {}).get(dest_type, [])
            score_fn = lambda candidate: 0.1 * xy_distance(
                obj["pose"]["position"], candidate["position"]
            )
        else:
            return None
        if not candidates:
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for candidate in candidates:
            reachable_count = sum(
                1
                for robot_id in CONTROL_SERVICE_TOPICS
                if self._robot_can_reach_pose(world, robot_id, candidate)
            )
            if reachable_count <= 0:
                continue
            score = score_fn(candidate) - float(reachable_count)
            if best is None or score < best[0]:
                best = (score, candidate)
        return best[1] if best else None

    def _goal_drop_candidates_for_object(
        self, obj: dict[str, Any], goal: dict[str, Any], world: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate object-aware goal poses that keep the full object footprint inside the goal."""
        center = goal["center"]
        scale = goal.get("scale", DEFAULT_TABLE_SIZE)
        obj_half_x, obj_half_y = object_xy_half_extents(obj)
        half_x = float(scale.get("x", 0.1)) * 0.5 - obj_half_x - GOAL_INTERIOR_MARGIN_M
        half_y = float(scale.get("y", 0.1)) * 0.5 - obj_half_y - GOAL_INTERIOR_MARGIN_M
        if half_x < 0.0 or half_y < 0.0:
            return []
        z = float(center["z"]) + GOAL_Z_OFFSET_M
        raw_points: list[dict[str, float]] = []
        grid_n = 7
        for ix in range(grid_n):
            for iy in range(grid_n):
                raw_points.append(
                    {
                        "x": float(center["x"])
                        - half_x
                        + 2.0 * half_x * ix / max(1, grid_n - 1),
                        "y": float(center["y"])
                        - half_y
                        + 2.0 * half_y * iy / max(1, grid_n - 1),
                        "z": z,
                    }
                )
        raw_points.append({"x": float(center["x"]), "y": float(center["y"]), "z": z})
        occupied = [
            candidate
            for candidate in world.get("objects", [])
            if candidate.get("id") != obj.get("id")
            and object_fully_inside_box_region(candidate, goal, margin=0.0)
        ]
        poses = []
        for point in raw_points:
            if not object_pose_fully_inside_box_region(
                obj, point, goal, margin=GOAL_INTERIOR_MARGIN_M
            ):
                continue
            if not self._clear_of_objects(point, occupied, DROP_CLEARANCE_M):
                continue
            poses.append(make_pose(point))
        # Stable deterministic order: central, then clearance-aware if occupied.
        poses.sort(key=lambda pose: self._goal_drop_score(obj, pose, goal, world))
        return poses

    @staticmethod
    def _clear_of_objects(
        point: dict[str, float], objects: list[dict[str, Any]], clearance: float
    ) -> bool:
        """Return True when point is not too close to any currently occupied object.

        SemanticRuntime also needs this helper because it generates object-aware
        goal-drop candidates after the symbolic world has been built. The same
        geometry rule is used by WorldModelBuilder when it precomputes safe pose
        candidates.
        """
        return all(
            xy_distance(point, obj["pose"]["position"]) >= clearance for obj in objects
        )

    def _goal_drop_score(
        self,
        obj: dict[str, Any],
        candidate: dict[str, Any],
        goal: dict[str, Any],
        world: dict[str, Any],
    ) -> float:
        p = candidate["position"]
        center = goal["center"]
        center_dist = xy_distance(p, center)
        travel_dist = xy_distance(obj["pose"]["position"], p)
        occupied = [
            other
            for other in world.get("objects", [])
            if other.get("id") != obj.get("id")
            and object_fully_inside_box_region(other, goal, margin=0.0)
        ]
        clearance = (
            0.0
            if not occupied
            else min(xy_distance(p, other["pose"]["position"]) for other in occupied)
        )
        return (
            GOAL_CENTER_BIAS_WEIGHT * center_dist
            + GOAL_TRAVEL_BIAS_WEIGHT * travel_dist
            - GOAL_CLEARANCE_BONUS_WEIGHT * clearance
        )

    def _select_handover_buffer_pose(
        self,
        obj: dict[str, Any],
        world: dict[str, Any],
        source_robots: list[str],
        destination_robots: list[str],
    ) -> dict[str, Any] | None:
        candidates = list(
            world.get("safe_pose_candidates", {}).get("handover_buffer", [])
        )
        # Add outside candidates as fallback handover buffers if the table center is not mutually reachable.
        candidates.extend(
            world.get("safe_pose_candidates", {}).get("outside_goals", [])[:10]
        )
        if not candidates:
            return None
        object_z = float(obj["pose"]["position"]["z"]) + BUFFER_Z_OFFSET_M
        for candidate in candidates:
            candidate = make_pose({**candidate["position"], "z": object_z})
            if any(
                self._robot_can_reach_pose(world, src, candidate)
                for src in source_robots
            ) and any(
                self._robot_can_reach_pose(world, dst, candidate)
                for dst in destination_robots
            ):
                return candidate
        return None

    def _decompose_task(
        self, world: dict[str, Any], task: ObjectTask
    ) -> list[PrimitiveDecision]:
        obj = get_object(world, task.object_id)
        if obj is None:
            raise RuntimeError(
                f"Object disappeared before decomposition: {task.object_id}"
            )
        route = task.route
        if route.route_type in {"direct", "direct_held"}:
            return self._decompose_direct(world, obj, route, task)
        if route.route_type == "handover":
            return self._decompose_handover(world, obj, route, task)
        raise RuntimeError(f"Unsupported route type: {route.route_type}")

    def _decompose_direct(
        self,
        world: dict[str, Any],
        obj: dict[str, Any],
        route: RouteCandidate,
        task: ObjectTask,
    ) -> list[PrimitiveDecision]:
        robot_id = route.source_robot_id
        object_pose = make_pose(obj["pose"]["position"])
        lift_pose = self._lift_pose(object_pose)
        approach_pose = self._approach_pose(route.drop_pose)
        decisions: list[PrimitiveDecision] = []
        if (
            obj.get("held_by") != robot_id
            and self.memory.held_objects.get(robot_id) != obj["id"]
        ):
            decisions.extend(
                [
                    self._decision(
                        robot_id,
                        ACTION_MOVING,
                        object_pose,
                        obj["id"],
                        "pick",
                        "Move to the selected object pose.",
                        task,
                    ),
                    self._decision(
                        robot_id,
                        ACTION_GRIP,
                        None,
                        obj["id"],
                        "grip",
                        "Grip the selected object.",
                        task,
                    ),
                ]
            )
        decisions.extend(
            [
                self._decision(
                    robot_id,
                    ACTION_MOVING,
                    lift_pose,
                    obj["id"],
                    "lift",
                    "Lift before horizontal transfer.",
                    task,
                ),
                self._decision(
                    robot_id,
                    ACTION_MOVING,
                    approach_pose,
                    obj["id"],
                    "drop_approach",
                    "Move above destination pose.",
                    task,
                ),
                self._decision(
                    robot_id,
                    ACTION_MOVING,
                    route.drop_pose,
                    obj["id"],
                    "drop",
                    "Lower to destination pose.",
                    task,
                ),
                self._decision(
                    robot_id,
                    ACTION_RELEASE,
                    None,
                    obj["id"],
                    "release",
                    "Release object at destination and verify marker state.",
                    task,
                    verify=self._release_verify_metadata(task, route.drop_pose),
                ),
                self._decision(
                    robot_id,
                    ACTION_HOMING,
                    None,
                    obj["id"],
                    "home",
                    "Return robot home after release.",
                    task,
                ),
            ]
        )
        return decisions

    def _decompose_handover(
        self,
        world: dict[str, Any],
        obj: dict[str, Any],
        route: RouteCandidate,
        task: ObjectTask,
    ) -> list[PrimitiveDecision]:
        if route.buffer_pose is None:
            raise RuntimeError("Handover route lacks buffer_pose")
        src = route.source_robot_id
        dst = route.destination_robot_id
        object_pose = make_pose(obj["pose"]["position"])
        src_lift = self._lift_pose(object_pose)
        buffer_approach = self._approach_pose(route.buffer_pose)
        buffer_drop = route.buffer_pose
        dst_lift = self._lift_pose(buffer_drop)
        drop_approach = self._approach_pose(route.drop_pose)
        return [
            self._decision(
                src,
                ACTION_MOVING,
                object_pose,
                obj["id"],
                "pick",
                "Source robot moves to object.",
                task,
            ),
            self._decision(
                src,
                ACTION_GRIP,
                None,
                obj["id"],
                "grip",
                "Source robot grips object.",
                task,
            ),
            self._decision(
                src,
                ACTION_MOVING,
                src_lift,
                obj["id"],
                "lift",
                "Source robot lifts object.",
                task,
            ),
            self._decision(
                src,
                ACTION_MOVING,
                buffer_approach,
                obj["id"],
                "buffer_approach",
                "Source robot moves above handover buffer.",
                task,
            ),
            self._decision(
                src,
                ACTION_MOVING,
                buffer_drop,
                obj["id"],
                "buffer_drop",
                "Source robot lowers to handover buffer.",
                task,
            ),
            self._decision(
                src,
                ACTION_RELEASE,
                None,
                obj["id"],
                "buffer_release",
                "Source robot releases object at handover buffer and verifies marker position.",
                task,
                verify={
                    "type": "object_near_pose",
                    "pose": buffer_drop,
                    "object_id": obj["id"],
                    "stage": "handover_buffer",
                },
            ),
            self._decision(
                src,
                ACTION_HOMING,
                None,
                obj["id"],
                "home",
                "Source robot homes after buffer release.",
                task,
            ),
            self._decision(
                dst,
                ACTION_MOVING,
                buffer_drop,
                obj["id"],
                "pick",
                "Destination robot moves to handover buffer object.",
                task,
            ),
            self._decision(
                dst,
                ACTION_GRIP,
                None,
                obj["id"],
                "grip",
                "Destination robot grips handover object.",
                task,
            ),
            self._decision(
                dst,
                ACTION_MOVING,
                dst_lift,
                obj["id"],
                "lift",
                "Destination robot lifts object.",
                task,
            ),
            self._decision(
                dst,
                ACTION_MOVING,
                drop_approach,
                obj["id"],
                "drop_approach",
                "Destination robot moves above final destination.",
                task,
            ),
            self._decision(
                dst,
                ACTION_MOVING,
                route.drop_pose,
                obj["id"],
                "drop",
                "Destination robot lowers to final destination.",
                task,
            ),
            self._decision(
                dst,
                ACTION_RELEASE,
                None,
                obj["id"],
                "release",
                "Destination robot releases object at final destination and verifies marker state.",
                task,
                verify=self._release_verify_metadata(task, route.drop_pose),
            ),
            self._decision(
                dst,
                ACTION_HOMING,
                None,
                obj["id"],
                "home",
                "Destination robot homes after release.",
                task,
            ),
        ]

    def _release_verify_metadata(
        self, task: ObjectTask, pose: dict[str, Any]
    ) -> dict[str, Any]:
        destination = task.destination
        dest_type = destination["type"]
        if dest_type == "goal":
            return {
                "type": "object_in_region",
                "object_id": task.object_id,
                "region": f"{destination['goal_color']}_goal",
            }
        if dest_type in {"outside_goals", "safe_free_space"}:
            return {"type": "object_outside_goals", "object_id": task.object_id}
        if dest_type in {"table_center", "handover_buffer"}:
            return {
                "type": "object_near_pose",
                "object_id": task.object_id,
                "pose": pose,
                "stage": dest_type,
            }
        return {"type": "object_near_pose", "object_id": task.object_id, "pose": pose}

    def _decision(
        self,
        robot_id: str,
        action: str,
        pose: dict[str, Any] | None,
        object_id: str,
        intent: str,
        reason: str,
        task: ObjectTask,
        verify: dict[str, Any] | None = None,
    ) -> PrimitiveDecision:
        metadata = {
            "semantic_tool": task.source_tool_call.as_dict(),
            "object_task_rank": task.rank,
            "route_type": task.route.route_type,
            "route_reason": task.route.reason,
            "task_rationale": task.rationale,
        }
        if verify:
            metadata["verify_after_action"] = verify
        return PrimitiveDecision(
            robot_id, action, pose, object_id, intent, reason, metadata=metadata
        )

    def _lift_pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        position = dict(pose["position"])
        position["z"] = float(position["z"]) + LIFT_DELTA_M
        return make_pose(position)

    def _approach_pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        position = dict(pose["position"])
        position["z"] = float(position["z"]) + SAFE_Z_OFFSET_M
        return make_pose(position)

    def _robots_that_can_reach_pose(
        self, world: dict[str, Any], pose: dict[str, Any]
    ) -> list[str]:
        candidates = [
            robot_id
            for robot_id in CONTROL_SERVICE_TOPICS
            if self._robot_can_reach_pose(world, robot_id, pose)
        ]
        candidates.sort(key=lambda rid: self._robot_pose_distance(world, rid, pose))
        return candidates

    def _robot_can_reach_pose(
        self, world: dict[str, Any], robot_id: str, pose: dict[str, Any]
    ) -> bool:
        robot = world.get("robots", {}).get(robot_id, {})
        base_pose = robot.get("base_pose")
        if not base_pose:
            return False
        return xy_distance(base_pose["position"], pose["position"]) <= float(
            robot.get("reach_radius", 0.0)
        )

    def _robot_pose_distance(
        self, world: dict[str, Any], robot_id: str, pose: dict[str, Any]
    ) -> float:
        robot = world.get("robots", {}).get(robot_id, {})
        base_pose = robot.get("base_pose")
        if not base_pose:
            return float("inf")
        return xy_distance(base_pose["position"], pose["position"])

    def _route_score(
        self,
        world: dict[str, Any],
        obj: dict[str, Any],
        drop_pose: dict[str, Any],
        src: str,
        dst: str,
    ) -> float:
        return self._robot_pose_distance(
            world, src, obj["pose"]
        ) + self._robot_pose_distance(world, dst, drop_pose)


def get_object(world: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in world.get("objects", []):
        if obj.get("id") == object_id:
            return obj
    return None


# =============================================================================
# Execution and verification
# =============================================================================


def is_topdown_grasp_state_rejection(message: str) -> bool:
    """Return True when a visual-verification rejection is based on grasp/held state.

    A single top-view image cannot reliably determine whether the gripper is open,
    closed, or still holding an object. Such a visual rejection should not override
    marker/geometry predicates that already passed.
    """
    text = str(message or "").lower()
    grasp_terms = (
        "held",
        "holding",
        "grasp",
        "gripper",
        "attached",
        "carried",
        "잡",
        "쥐",
        "들고",
        "그리퍼",
        "파지",
    )
    return any(term in text for term in grasp_terms)


class VisualVerifier:
    def __init__(self, reporter: Reporter) -> None:
        self.reporter = reporter

    def verify_predicate(
        self, world: dict[str, Any], image_bytes: bytes, verify: dict[str, Any]
    ) -> tuple[bool, str]:
        if not USE_MULTIMODAL_VISUAL_VERIFICATION:
            return True, "visual verification disabled"
        prompt = self._predicate_prompt(world, verify)
        return self._call(prompt, image_bytes)

    def verify_finish(
        self,
        world: dict[str, Any],
        image_bytes: bytes,
        conditions: list[SemanticGoalCondition],
    ) -> tuple[bool, str]:
        if not USE_MULTIMODAL_VISUAL_VERIFICATION:
            return True, "visual finish verification disabled"
        prompt = self._finish_prompt(world, conditions)
        return self._call(prompt, image_bytes)

    def _predicate_prompt(self, world: dict[str, Any], verify: dict[str, Any]) -> str:
        return json.dumps(
            {
                "role": "multimodal_verifier",
                "instruction": (
                    "Verify whether the last robot action visually achieved the geometric predicate. "
                    "Use the top-view image only for visible 2D geometry: object footprint, goal/table regions, and relative placement. "
                    "Important limitation: the camera is top-down, so it cannot reliably determine gripper open/closed state or whether an object is held. "
                    "Therefore, never reject a predicate because an object appears to be held; ignore grasp/held state entirely in this visual check. "
                    "For goal placement, verify that the full visible object footprint is inside the goal area, not merely touching a boundary. "
                    "For outside-goal placement, verify that the object is visibly outside all goal regions and on the table. Return JSON only."
                ),
                "predicate": verify,
                "world_model": self._compact_world(world),
                "required_response": {
                    "success": "boolean",
                    "confidence": "number from 0 to 1",
                    "reason": "short explanation",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def _finish_prompt(
        self, world: dict[str, Any], conditions: list[SemanticGoalCondition]
    ) -> str:
        return json.dumps(
            {
                "role": "multimodal_finish_verifier",
                "instruction": (
                    "Decide whether the entire user task is visually and geometrically complete. "
                    "Use the top-view image only for visible 2D geometry: object footprint, goal/table regions, and relative placement. "
                    "Important limitation: the camera is top-down, so it cannot reliably determine gripper open/closed state or whether an object is held. "
                    "Do not use visual inspection to decide held/released state; that is handled separately by the symbolic execution state. "
                    "Objects requested inside a goal are checked metrically by MarkerArray geometry first. Use the image only as an advisory sanity check for obvious contradictions, not as the exact boundary judge. Do not override passed marker geometry based on approximate visual boundary interpretation. Return JSON only."
                ),
                "goal_conditions": [condition.as_dict() for condition in conditions],
                "world_model": self._compact_world(world),
                "required_response": {
                    "success": "boolean",
                    "confidence": "number from 0 to 1",
                    "reason": "short explanation",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def _compact_world(self, world: dict[str, Any]) -> dict[str, Any]:
        return {
            "regions": {
                name: {
                    "type": region.get("type"),
                    "color": region.get("color"),
                    "center": region.get("center"),
                    "scale": region.get("scale"),
                }
                for name, region in world.get("regions", {}).items()
            },
            "objects": [
                {
                    "id": obj.get("id"),
                    "color": obj.get("color"),
                    "shape": obj.get("shape"),
                    "current_region": obj.get("current_region"),
                    "position": obj.get("pose", {}).get("position"),
                    "scale": obj.get("scale"),
                }
                for obj in world.get("objects", [])
            ],
            "robots": {
                robot_id: {
                    "end_effector_pose": robot.get("end_effector_pose"),
                }
                for robot_id, robot in world.get("robots", {}).items()
            },
            "geometry_policy": {
                "goal_success": "full object footprint inside goal, with inward margin",
                "goal_interior_margin_m": GOAL_INTERIOR_MARGIN_M,
                "top_view_limitation": "grasp/held state is not visually observable and is intentionally omitted from this verifier context",
            },
        }

    def _call(self, prompt: str, image_bytes: bytes) -> tuple[bool, str]:
        errors: list[str] = []
        modes = [LLM_BACKEND] if LLM_BACKEND != "auto" else ["openai", "ollama"]
        for mode in modes:
            try:
                if mode == "openai":
                    parsed = self._call_openai(prompt, image_bytes)
                elif mode == "ollama":
                    parsed = self._call_ollama(prompt, image_bytes)
                else:
                    errors.append(f"unsupported backend={mode!r}")
                    continue
                success = bool(parsed.get("success", False))
                confidence = float(parsed.get("confidence", 0.0) or 0.0)
                reason = str(parsed.get("reason") or "")
                if success and confidence >= VISUAL_VERIFY_CONFIDENCE_THRESHOLD:
                    return (
                        True,
                        f"visual verification passed confidence={confidence:.2f}: {reason}",
                    )
                return (
                    False,
                    f"visual verification rejected confidence={confidence:.2f}: {reason}",
                )
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
                if LLM_BACKEND != "auto":
                    break
        if REQUIRE_MULTIMODAL_VISUAL_VERIFICATION:
            return False, "visual verification unavailable: " + " | ".join(errors)
        return True, "visual verification unavailable but not required: " + " | ".join(
            errors
        )

    def _call_openai(self, prompt: str, image_bytes: bytes) -> dict[str, Any]:
        api_key = (
            os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
        ).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or CHATGPT_API_KEY is not set")
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict multimodal robot-task verifier. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image_bytes)},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        response = post_json(
            OPENAI_API_URL, payload, {"Authorization": f"Bearer {api_key}"}
        )
        choices = response.get("choices") or []
        message = (
            choices[0].get("message")
            if choices and isinstance(choices[0], dict)
            else {}
        )
        return extract_json_object(str((message or {}).get("content") or ""))

    def _call_ollama(self, prompt: str, image_bytes: bytes) -> dict[str, Any]:
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict multimodal robot-task verifier. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("utf-8")],
                },
            ],
            "format": "json",
            "options": {"temperature": 0.0},
        }
        response = post_json(OLLAMA_URL, payload, {})
        msg = response.get("message") or {}
        return extract_json_object(
            str(msg.get("content") or response.get("response") or "")
        )


class Executor:
    def __init__(
        self,
        node: RosWorldNode,
        memory: ExecutionMemory,
        builder: WorldModelBuilder,
        reporter: Reporter,
        visual_verifier: VisualVerifier,
    ) -> None:
        self.node = node
        self.memory = memory
        self.builder = builder
        self.reporter = reporter
        self.visual_verifier = visual_verifier

    def execute(self, decision: PrimitiveDecision) -> ExecutionResult:
        if decision.done:
            result = ExecutionResult(True, "Task marked complete.", decision)
            self.memory.append_result(result)
            return result

        if decision.robot_id not in CONTROL_SERVICE_TOPICS:
            result = ExecutionResult(
                False, f"Invalid robot_id: {decision.robot_id}", decision
            )
            self.memory.append_result(result)
            return result
        if decision.action not in ALLOWED_PRIMITIVE_ACTIONS:
            result = ExecutionResult(
                False, f"Invalid primitive action: {decision.action}", decision
            )
            self.memory.append_result(result)
            return result

        if DRY_RUN:
            success, message = True, "dry-run: ROS service call skipped"
        else:
            success, message = self.node.call_control_service(
                decision.robot_id, decision.action, decision.target_pose
            )
        result = ExecutionResult(success, message, decision)

        if (
            result.success
            and decision.action == ACTION_MOVING
            and decision.target_pose is not None
            and not DRY_RUN
        ):
            arrived = self.node.wait_for_eef_target(
                decision.robot_id, decision.target_pose, MOTION_TIMEOUT_SEC
            )
            if not arrived:
                result = ExecutionResult(
                    False,
                    f"{result.message}; EEF did not reach target before timeout",
                    decision,
                )
        elif result.success and not DRY_RUN:
            self._settle(SETTLE_SEC)

        if result.success:
            verified, verification_message = self._verify_after_action(decision)
            if not verified:
                result = ExecutionResult(
                    False,
                    f"{result.message}; verification failed: {verification_message}",
                    decision,
                )
            else:
                if verification_message:
                    result = ExecutionResult(
                        True, f"{result.message}; {verification_message}", decision
                    )

        self._update_holding_state(decision, result.success)
        self.memory.append_result(result)
        return result

    def _settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _verify_after_action(self, decision: PrimitiveDecision) -> tuple[bool, str]:
        verify = (
            decision.metadata.get("verify_after_action") if decision.metadata else None
        )
        if not verify:
            return True, ""
        time.sleep(VERIFY_DELAY_SEC)
        raw, image_bytes = self.node.snapshot_raw()
        world = self.builder.build(raw)
        # First run the strict marker/geometry predicate. This is the authoritative metric check.
        ok, message = verify_predicate(world, verify)
        if ok and self._needs_visual_verification(verify):
            visual_ok, visual_message = self.visual_verifier.verify_predicate(
                world, image_bytes, verify
            )
            if not visual_ok and is_topdown_grasp_state_rejection(visual_message):
                visual_message = (
                    "visual advisory rejection was ignored because it was based on grasp/held state, "
                    "which is not observable from the top-down camera: "
                    + visual_message
                )
            elif not visual_ok and not VISUAL_VERIFICATION_CAN_VETO_MARKER_SUCCESS:
                visual_message = (
                    "visual advisory rejection was ignored because strict MarkerArray geometry already passed: "
                    + visual_message
                )
            else:
                # This branch is kept for optional future stricter modes.
                ok = visual_ok
            message = f"{message}; {visual_message}"
        self.memory.last_verification = {
            "ok": ok,
            "message": message,
            "verify": verify,
            "time": time.time(),
        }
        return ok, message

    @staticmethod
    def _needs_visual_verification(verify: dict[str, Any]) -> bool:
        return verify.get("type") in {
            "object_in_region",
            "object_outside_goals",
            "object_near_pose",
        }

    def _update_holding_state(self, decision: PrimitiveDecision, success: bool) -> None:
        if decision.robot_id not in self.memory.held_objects:
            return
        robot_id = str(decision.robot_id)
        if not success:
            return

        # Any non-homing primitive means the robot should not be considered safely homed
        # for final verification. Homing itself is the only action that sets this true.
        if decision.action == ACTION_HOMING:
            self.memory.robots_homed[robot_id] = True
            return
        self.memory.robots_homed[robot_id] = False

        if decision.action == ACTION_MOVING and decision.intent == "pick":
            self.memory.pending_grip_targets[robot_id] = decision.target_object_id
        elif decision.action == ACTION_GRIP:
            self.memory.held_objects[robot_id] = (
                decision.target_object_id
                or self.memory.pending_grip_targets.get(robot_id)
            )
            self.memory.pending_grip_targets[robot_id] = None
        elif decision.action == ACTION_RELEASE:
            # Only clear after release succeeded and post-action verification passed.
            self.memory.held_objects[robot_id] = None
            self.memory.pending_grip_targets[robot_id] = None

    def emergency_release_and_home_all(self) -> list[str]:
        """Fail-safe recovery: sequentially open every gripper and home every robot.

        This is intentionally not planned by the LLM. It is a deterministic safety
        fallback for repeated execution/verification failures or stale holding state.
        """
        messages: list[str] = []
        for robot_id in EMERGENCY_ROBOT_ORDER:
            if DRY_RUN:
                ok, msg = True, "dry-run: emergency Realease skipped"
            else:
                ok, msg = self.node.call_control_service(robot_id, ACTION_RELEASE, None)
                self._settle(SETTLE_SEC)
            messages.append(f"{robot_id} {ACTION_RELEASE}: success={ok}, message={msg}")

            if DRY_RUN:
                ok, msg = True, "dry-run: emergency Homing skipped"
            else:
                ok, msg = self.node.call_control_service(robot_id, ACTION_HOMING, None)
                self._settle(SETTLE_SEC)
            messages.append(f"{robot_id} {ACTION_HOMING}: success={ok}, message={msg}")

        for robot_id in self.memory.held_objects:
            self.memory.held_objects[robot_id] = None
            self.memory.pending_grip_targets[robot_id] = None
            self.memory.robots_homed[robot_id] = True
        self.memory.last_verification = {
            "ok": True,
            "message": "emergency release/home executed; internal held state reset",
            "time": time.time(),
        }
        return messages


class FinishVerifier:
    def __init__(
        self,
        node: RosWorldNode,
        memory: ExecutionMemory,
        builder: WorldModelBuilder,
        runtime: SemanticRuntime,
        visual_verifier: VisualVerifier,
    ) -> None:
        self.node = node
        self.memory = memory
        self.builder = builder
        self.runtime = runtime
        self.visual_verifier = visual_verifier

    def is_complete(self) -> tuple[bool, str]:
        if any(self.memory.held_objects.values()):
            return (
                False,
                f"robot is still holding object(s): {self.memory.held_objects}",
            )
        if any(self.memory.pending_grip_targets.values()):
            return (
                False,
                f"pending grip target remains: {self.memory.pending_grip_targets}",
            )
        if (
            FINAL_VERIFY_REQUIRES_ALL_ROBOTS_HOMED
            and not self.memory.all_robots_homed()
        ):
            return (
                False,
                f"final verification requires both robots homed first; not_homed={self.memory.robots_not_homed()}, homed_state={self.memory.robots_homed}",
            )
        if not self.runtime.concrete_goal_conditions:
            return False, "no concrete goal conditions are available"

        last_message = ""
        last_world: dict[str, Any] | None = None
        last_image_bytes: bytes | None = None
        for check_idx in range(FINISH_VERIFY_CHECKS):
            raw, image_bytes = self.node.snapshot_raw()
            world = self.builder.build(raw)
            ok, message = verify_goal_conditions(
                world, self.runtime.concrete_goal_conditions
            )
            if not ok:
                return False, message
            last_message = message
            last_world = world
            last_image_bytes = image_bytes
            if check_idx + 1 < FINISH_VERIFY_CHECKS:
                time.sleep(FINISH_VERIFY_INTERVAL_SEC)
        if last_world is not None and last_image_bytes is not None:
            visual_ok, visual_message = self.visual_verifier.verify_finish(
                last_world, last_image_bytes, self.runtime.concrete_goal_conditions
            )
            if not visual_ok and is_topdown_grasp_state_rejection(visual_message):
                visual_message = (
                    "visual finish advisory rejection was ignored because it was based on grasp/held state, "
                    "which is not observable from the top-down camera: "
                    + visual_message
                )
            elif (
                not visual_ok and not FINAL_VISUAL_VERIFICATION_CAN_VETO_MARKER_SUCCESS
            ):
                visual_message = (
                    "visual finish advisory rejection was ignored because strict fresh MarkerArray geometry checks already passed: "
                    + visual_message
                )
            elif not visual_ok:
                return False, visual_message
            last_message = f"{last_message}; {visual_message}"
        return (
            True,
            f"finish verified after both robots homed by {FINISH_VERIFY_CHECKS} fresh marker checks; multimodal visual check was advisory: {last_message}",
        )


def verify_goal_conditions(
    world: dict[str, Any], conditions: list[SemanticGoalCondition]
) -> tuple[bool, str]:
    for condition in conditions:
        for object_id in condition.object_ids:
            obj = get_object(world, object_id)
            if obj is None:
                return False, f"object missing during finish verification: {object_id}"
            if obj.get("held_by"):
                return (
                    False,
                    f"object {object_id} is still held by {obj.get('held_by')}",
                )
            if condition.condition_type == "objects_in_goal":
                goal = world.get("regions", {}).get(str(condition.required_region))
                if not goal:
                    return (
                        False,
                        f"goal region missing during finish verification: {condition.required_region}",
                    )
                if not object_fully_inside_box_region(
                    obj, goal, margin=GOAL_INTERIOR_MARGIN_M
                ):
                    return False, (
                        f"object {object_id} is not fully inside {condition.required_region} "
                        f"with margin={GOAL_INTERIOR_MARGIN_M:.3f}; observed_region={obj.get('current_region')}, "
                        f"pose={obj.get('pose', {}).get('position')}, scale={obj.get('scale')}"
                    )
            elif condition.condition_type == "objects_outside_goals":
                if obj.get("current_region") in {"red_goal", "blue_goal"}:
                    return (
                        False,
                        f"object {object_id} is still inside {obj.get('current_region')}, expected outside_goals",
                    )
            elif condition.condition_type == "objects_near_region_pose":
                region = world.get("regions", {}).get(
                    str(condition.required_region), {}
                )
                pose = region.get("pose")
                if not pose:
                    return False, f"region {condition.required_region} has no pose"
                if (
                    xy_distance(obj["pose"]["position"], pose["position"])
                    > VERIFY_XY_TOLERANCE_M
                ):
                    return (
                        False,
                        f"object {object_id} is not near {condition.required_region}",
                    )
            else:
                return False, f"unknown goal condition type: {condition.condition_type}"
    return (
        True,
        f"all {len(conditions)} semantic goal condition(s) are satisfied and no object is held",
    )


def verify_predicate(world: dict[str, Any], verify: dict[str, Any]) -> tuple[bool, str]:
    vtype = verify.get("type")
    object_id = verify.get("object_id")
    obj = get_object(world, str(object_id)) if object_id else None
    if obj is None:
        return False, f"object not visible: {object_id}"
    if vtype == "object_in_region":
        region = str(verify.get("region"))
        goal = world.get("regions", {}).get(region)
        if goal and goal.get("type") == "goal":
            if object_fully_inside_box_region(obj, goal, margin=GOAL_INTERIOR_MARGIN_M):
                return (
                    True,
                    f"verified: object {object_id} is fully inside {region} with margin={GOAL_INTERIOR_MARGIN_M:.3f}",
                )
            return False, (
                f"object {object_id} is not fully inside {region}; "
                f"observed_region={obj.get('current_region')}, pose={obj.get('pose', {}).get('position')}, scale={obj.get('scale')}"
            )
        if obj.get("current_region") == region:
            return True, f"verified: object {object_id} is in {region}"
        return (
            False,
            f"object {object_id} is in {obj.get('current_region')}, expected {region}",
        )
    if vtype == "object_outside_goals":
        if obj.get("current_region") not in {"red_goal", "blue_goal"}:
            return True, f"verified: object {object_id} is outside all goals"
        return False, f"object {object_id} is still in {obj.get('current_region')}"
    if vtype == "object_near_pose":
        pose = verify.get("pose")
        if not isinstance(pose, dict):
            return False, "object_near_pose verification lacks pose"
        pos = pose["position"]
        obj_pos = obj["pose"]["position"]
        if (
            xy_distance(obj_pos, pos) <= VERIFY_XY_TOLERANCE_M
            and abs(float(obj_pos["z"]) - float(pos["z"])) <= VERIFY_Z_TOLERANCE_M
        ):
            return True, f"verified: object {object_id} is near expected pose"
        return (
            False,
            f"object {object_id} is not near expected pose; observed={obj_pos}, expected={pos}",
        )
    return False, f"unknown verification type: {vtype}"


# =============================================================================
# Runtime utilities and main loop
# =============================================================================


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


def save_frame(step: int, image_bytes: bytes) -> None:
    if SAVE_LLM_FRAMES_DIR is None:
        return
    SAVE_LLM_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_LLM_FRAMES_DIR / f"step_{step:03d}.png").write_bytes(image_bytes)


def request_llm_plan_with_validation(
    llm: LLMSemanticPlanner,
    validator: PlanValidator,
    command: str,
    world: dict[str, Any],
    image_bytes: bytes,
    feedback: str,
    reporter: Reporter,
) -> LLMPlan:
    current_feedback = feedback
    for attempt in range(1, MAX_LLM_REPLAN_ATTEMPTS + 1):
        plan = llm.plan(command, world, image_bytes, current_feedback)
        try:
            validator.validate(plan, world)
            return plan
        except ValueError as exc:
            current_feedback = (
                f"Your previous semantic plan was rejected before execution: {exc}. "
                "Return a corrected MCP-like semantic plan using only the supported tools. "
                "Important: do not reject merely because one robot cannot reach both the object and destination; "
                "the runtime supports handover through handover_buffer and will compute route feasibility. "
                "Use reject_task only for unsafe off-table commands, unknown objects/regions/colors/shapes, or true ambiguity."
            )
            reporter.error(f"Rejected LLM semantic plan attempt {attempt}: {exc}")
    raise RuntimeError(current_feedback)


def run() -> int:
    reporter = Reporter()
    command = read_interactive_command(reporter)

    if ROS_IMPORT_ERROR is not None:
        print(
            "ROS2 imports failed. Source the ROS2 workspace before running this CLI, "
            "then run it with the ROS2 Python interpreter, for example:\n"
            "  source install/setup.bash && /usr/bin/python3 web-app/cli_world_model_llm.py",
            file=sys.stderr,
        )
        print(f"Import error: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 2

    reporter.status(
        "Startup",
        {
            "planner": "world_model_llm_mcp_semantic_planner",
            "llm_backend": LLM_BACKEND,
            "openai_model": OPENAI_MODEL,
            "ollama_model": OLLAMA_MODEL,
            "dry_run": DRY_RUN,
            "command": command,
            "reach_rule": f"XY distance <= {REACH_RADIUS_TABLE_SCALE:.2f} * table major length",
            "max_steps": MAX_STEPS,
        },
    )

    rclpy.init()
    node = RosWorldNode()
    memory = ExecutionMemory()
    builder = WorldModelBuilder(memory)
    llm = LLMSemanticPlanner(reporter)
    validator = PlanValidator()
    runtime = SemanticRuntime(memory, reporter)
    visual_verifier = VisualVerifier(reporter)
    executor = Executor(node, memory, builder, reporter, visual_verifier)
    finish_verifier = FinishVerifier(node, memory, builder, runtime, visual_verifier)

    feedback = ""
    consecutive_failures = 0
    try:
        wait_for_ready(node, reporter)
        raw, image_bytes = node.snapshot_raw()
        world = builder.build(raw)
        save_frame(0, image_bytes)

        reporter.status(
            "Initial symbolic world",
            {
                "objects": [
                    f"{obj['id']}({obj['color']},{obj['shape']},{obj['current_region']})"
                    for obj in world["objects"]
                ],
                "reachable_regions": world["summary"].get("reachable_regions"),
                "held": memory.held_objects,
                "robots_homed": memory.robots_homed,
            },
        )

        try:
            plan = request_llm_plan_with_validation(
                llm, validator, command, world, image_bytes, feedback, reporter
            )
        except Exception as exc:
            reporter.error(f"Unable to obtain an executable semantic plan: {exc}")
            return 1
        runtime.load_plan(plan, world)
        reporter.status("Accepted semantic plan", plan.as_dict())

        for step in range(1, MAX_STEPS + 1):
            raw, image_bytes = node.snapshot_raw()
            world = builder.build(raw)
            save_frame(step, image_bytes)
            reporter.status(
                "Observation",
                {
                    "step": step,
                    "objects_by_region": world["summary"].get("objects_by_region"),
                    "held": memory.held_objects,
                    "robots_homed": memory.robots_homed,
                    "active_object": runtime.active_object_id,
                    "queued_primitives": len(runtime.primitive_queue),
                },
            )

            complete, finish_msg = finish_verifier.is_complete()
            if complete:
                reporter.info(finish_msg)
                return 0

            try:
                decision = runtime.next_decision(world)
            except Exception as exc:
                feedback = (
                    f"The previous semantic plan cannot be decomposed from the current world: {exc}. "
                    f"Current world summary: {world.get('summary')}. Recent execution: {memory.as_llm_context()}."
                )
                reporter.error(feedback)
                try:
                    plan = request_llm_plan_with_validation(
                        llm, validator, command, world, image_bytes, feedback, reporter
                    )
                    runtime.load_plan(plan, world)
                    reporter.status("Replanned semantic plan", plan.as_dict())
                    decision = runtime.next_decision(world)
                except Exception as replanning_exc:
                    reporter.error(f"Replanning failed: {replanning_exc}")
                    return 1

            if decision.done:
                complete, finish_msg = finish_verifier.is_complete()
                if complete:
                    result = executor.execute(decision)
                    reporter.status(
                        "Execution result",
                        {"success": result.success, "message": result.message},
                    )
                    reporter.info(finish_msg)
                    return 0
                reporter.error(
                    f"Finish was proposed but verifier rejected it: {finish_msg}"
                )
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_EMERGENCY_THRESHOLD:
                    reporter.error(
                        f"Consecutive failure threshold reached ({consecutive_failures}). "
                        "Running emergency Realease + Homing for all robots."
                    )
                    for line in executor.emergency_release_and_home_all():
                        reporter.info(line)
                    runtime.clear_plan()
                    consecutive_failures = 0
                    feedback = (
                        "Emergency recovery was executed after repeated finish verification failures. "
                        "All grippers were opened and all robots were homed; internal held state was reset. "
                        "Use the current fresh world model to continue only if task goal conditions remain unsatisfied."
                    )
                    continue
                feedback = f"Finish verifier rejected completion: {finish_msg}. Continue or replan."
                runtime.clear_plan()
                continue

            reporter.primitive(step, decision, len(runtime.primitive_queue))
            result = executor.execute(decision)
            reporter.status(
                "Execution result",
                {"success": result.success, "message": result.message},
            )
            if result.success:
                consecutive_failures = 0
            if not result.success:
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_EMERGENCY_THRESHOLD:
                    reporter.error(
                        f"Consecutive failure threshold reached ({consecutive_failures}). "
                        "Running emergency Realease + Homing for all robots."
                    )
                    for line in executor.emergency_release_and_home_all():
                        reporter.info(line)
                    runtime.clear_plan()
                    consecutive_failures = 0
                    feedback = (
                        "Emergency recovery was executed after repeated primitive failures. "
                        "All grippers were opened and all robots were homed; internal held state was reset. "
                        "Use the current fresh world model to continue only if task goal conditions remain unsatisfied."
                    )
                    continue
                feedback = (
                    "The previous primitive execution failed. "
                    f"decision={decision}, message={result.message}, runtime_state={memory.as_llm_context()}. "
                    "Use the updated symbolic world model and choose a corrected semantic plan."
                )
                runtime.clear_plan()
                try:
                    raw, image_bytes = node.snapshot_raw()
                    world = builder.build(raw)
                    plan = request_llm_plan_with_validation(
                        llm, validator, command, world, image_bytes, feedback, reporter
                    )
                    runtime.load_plan(plan, world)
                    reporter.status(
                        "Replanned semantic plan after failure", plan.as_dict()
                    )
                except Exception as exc:
                    reporter.error(f"Unable to recover after primitive failure: {exc}")
                    return 1

        reporter.error(f"Stopped after MAX_STEPS={MAX_STEPS}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
