#!/usr/bin/python3
"""Multimodal ROS2 CLI planner/executor for the dual-Franka scene."""

from __future__ import annotations

import argparse
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
from typing import Any

from dotenv import load_dotenv
from PIL import Image

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional terminal nicety
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


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")

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
ACTION_CENTERING = "Centering"
ACTION_GRIP = "Grip"
ACTION_REALEASE = "Realease"
ACTION_HOMING = "Homing"
ACTION_PLACING = "Placing"
ALLOWED_ACTIONS = {
    ACTION_MOVING,
    ACTION_CENTERING,
    ACTION_GRIP,
    ACTION_REALEASE,
    ACTION_HOMING,
}

DEFAULT_COMMAND = "빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라."
VERTICAL_EEF_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
DEFAULT_TABLE_CENTER = {"x": 0.6, "y": 0.0, "z": 0.46}
DEFAULT_TABLE_SIZE = {"x": 0.9, "y": 0.7, "z": 0.08}
MIN_DROP_CLEARANCE_M = 0.045
GOAL_MARGIN_M = 0.012
EEF_POSITION_TOLERANCE_M = 0.035
SERVICE_WAIT_TIMEOUT_SEC = 1.0
ROBOT_SCENE_LABELS = {
    "left": "bottom_robot",
    "right": "top_robot",
}


@dataclass
class Decision:
    done: bool = False
    robot_id: str | None = None
    action: str | None = None
    target_pose: dict[str, Any] | None = None
    target_object_id: str | None = None
    intent: str = ""
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    message: str
    decision: Decision


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120
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
    if color["r"] >= color["b"]:
        return "red"
    return "blue"


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


def normalize_pose_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if "position" in value:
        position = value.get("position") or {}
    else:
        position = value
    if not isinstance(position, dict):
        return None
    try:
        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": dict(VERTICAL_EEF_ORIENTATION),
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
    # Match control_command_gui.py exactly: only the position is planner-controlled.
    # world.py uses this quaternion as a hard IK target, so accepting arbitrary
    # LLM quaternions can make the Franka roll over or approach sideways.
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


def image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_planner_system_prompt() -> str:
    return (
        "You are a multimodal robot task planner controlling two Franka arms in Isaac Sim. "
        "Use both the top-view image and the serialized ROS2 world state. "
        "Return only one valid JSON object and no markdown. "
        "The global task is to place every red object inside the red goal region and every blue object inside the blue goal region. "
        "Never use the action string 'Placing'. It is forbidden at this stage. "
        "Allowed action strings are exactly: Moving, Centering, Grip, Realease, Homing. "
        "Robot IDs are names, not screen directions: robot_id='left' is the bottom robot in the top-view image, "
        "and robot_id='right' is the top robot in the top-view image. "
        "Do not choose a robot from the words left/right in the image. Choose the robot from metric reachability, "
        "EEF distance, object position, and the provided reachable_objects/candidate lists. "
        "For pick actions, prefer objects from recommended_pick_candidates and use that item's best_robot_id unless there is a clear reason not to. "
        "Object color determines the destination goal only: red objects go to the red goal, blue objects go to the blue goal. "
        "The CLI is recursive: you may output a short action_sequence, but only action_sequence[0] is executed. "
        "After that first action completes, you will receive updated observations and execution_history, then you must plan again. "
        "Therefore the first action must be immediately executable from the current state. "
        "Use this state machine for each object: "
        "1) if the matching robot is not holding the object, first Moving with intent='pick' to that object's marker pose; "
        "2) after that Moving succeeds, Grip with intent='grip'; "
        "3) only after Grip succeeds and held_objects shows the robot holding that object, Moving with intent='drop' to a chosen drop pose inside the matching goal; "
        "4) after the drop Moving succeeds, Realease with intent='release'; "
        "5) optionally Homing. "
        "Never use intent='drop' when held_objects for that robot is null. "
        "Never use Realease when held_objects for that robot is null. "
        "Do not blindly use the goal center as a drop pose. Choose a pose inside the goal region that avoids collisions "
        "and avoids stacking objects when possible. "
        "Use the ROS2 marker poses as metric ground truth. Use the image to verify color, layout, and spatial relations. "
        "If the previous execution failed, use execution_history, failed_motion_targets, recommended_pick_candidates, and feedback_from_executor to correct the next first action. "
        "Do not blindly repeat a failed Moving command with the same robot_id, intent, and target_object_id; "
        "retry it only if the failure looks like a temporary timeout, otherwise choose a different reachable candidate. "
        "The target_pose orientation is fixed by the executor; only choose target_pose.position. "
        "If you include orientation, it must be this exact vertical end-effector quaternion: "
        '{"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}. '
        "Prefer this response schema: "
        '{"done": boolean, "action_sequence": ['
        '{"robot_id": "left"|"right", "action": "Moving"|"Centering"|"Grip"|"Realease"|"Homing", '
        '"target_pose": {"position": {"x": number, "y": number, "z": number}, '
        '"orientation": {"x": number, "y": number, "z": number, "w": number}}, '
        '"target_object_id": string|null, "intent": "pick"|"drop"|"center"|"grip"|"release"|"home", "reason": string}'
        '], "reason": string}. '
        "The executor will run only the first item in action_sequence. "
        "A legacy single-action object with robot_id/action/target_pose is also accepted, but action_sequence is preferred. "
        "If the task is complete, return done=true with intent='finish'."
    )


def make_planner_user_prompt(
    command: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
    feedback: str = "",
) -> str:
    payload = {
        "high_level_command": command,
        "observation": observation,
        "execution_history": history[-20:],
        "feedback_from_executor": feedback,
    }
    return (
        "Plan a short sequence, but remember only the FIRST action will be executed before replanning.\n"
        "The first action must obey the current held_objects state and any failure feedback.\n"
        "Remember robot_id='left' is the bottom robot and robot_id='right' is the top robot in the image.\n"
        "Choose robots by reachability and distance, not by object color. Object color only selects the destination goal.\n"
        "For a new pick, use recommended_pick_candidates[0] unless it appears blocked or recently failed.\n"
        "Do not drop or release unless the selected robot is currently holding the target object.\n"
        "If no robot is holding a target object, the first useful action is normally Moving with intent='pick' to an unsorted object's marker pose.\n"
        "Here is the current serialized ROS2 state:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


class RosWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("multimodal_franka_cli")
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

        table_size = table.get("scale", DEFAULT_TABLE_SIZE)
        table_length = max(
            float(table_size.get("x", 0.9)), float(table_size.get("y", 0.7))
        )
        workspace_radius = table_length * 0.65
        object_list = sorted(objects.values(), key=lambda obj: obj["id"])

        robots: dict[str, Any] = {}
        for robot_id in ("left", "right"):
            base_pose = robot_poses.get(robot_id)
            reachable: list[str] = []
            object_distances: dict[str, float] = {}
            if base_pose is not None:
                base_position = base_pose["position"]
                for obj in object_list:
                    distance = xy_distance(base_position, obj["pose"]["position"])
                    object_distances[obj["id"]] = round(distance, 4)
                    if distance <= workspace_radius:
                        reachable.append(obj["id"])
            robots[robot_id] = {
                "scene_label": ROBOT_SCENE_LABELS[robot_id],
                "top_view_note": (
                    "This is the bottom robot in the camera image."
                    if robot_id == "left"
                    else "This is the top robot in the camera image."
                ),
                "base_pose": base_pose,
                "end_effector_pose": eef_poses.get(robot_id),
                "workspace_radius": workspace_radius,
                "reachable_objects": reachable,
                "object_xy_distances": object_distances,
                "control_service": CONTROL_SERVICE_TOPICS[robot_id],
            }

        pick_candidates = self._pick_candidates(object_list, robots)

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
                "pose": table.get(
                    "pose",
                    {
                        "position": DEFAULT_TABLE_CENTER,
                        "orientation": VERTICAL_EEF_ORIENTATION,
                    },
                ),
                "size": table_size,
            },
            "goals": goals,
            "robots": robots,
            "objects": object_list,
            "recommended_pick_candidates": pick_candidates,
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

    @staticmethod
    def _pick_candidates(
        objects: list[dict[str, Any]], robots: dict[str, Any]
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for obj in objects:
            if obj.get("sorted"):
                continue
            robot_options: list[dict[str, Any]] = []
            for robot_id, robot in robots.items():
                distance = robot.get("object_xy_distances", {}).get(obj["id"])
                reachable = obj["id"] in robot.get("reachable_objects", [])
                robot_options.append(
                    {
                        "robot_id": robot_id,
                        "scene_label": robot.get("scene_label"),
                        "xy_distance": distance,
                        "reachable": reachable,
                    }
                )
            robot_options.sort(
                key=lambda option: (
                    not bool(option["reachable"]),
                    (
                        float(option["xy_distance"])
                        if option["xy_distance"] is not None
                        else float("inf")
                    ),
                )
            )
            candidates.append(
                {
                    "object_id": obj["id"],
                    "namespace": obj["namespace"],
                    "color": obj["color"],
                    "shape": obj["shape"],
                    "position": obj["pose"]["position"],
                    "destination_goal": f"{obj['color']}_goal",
                    "best_robot_id": (
                        robot_options[0]["robot_id"] if robot_options else None
                    ),
                    "robot_options": robot_options,
                }
            )
        candidates.sort(
            key=lambda candidate: (
                (
                    not bool(candidate["robot_options"][0]["reachable"])
                    if candidate["robot_options"]
                    else True
                ),
                (
                    float(candidate["robot_options"][0]["xy_distance"])
                    if candidate["robot_options"]
                    and candidate["robot_options"][0]["xy_distance"] is not None
                    else float("inf")
                ),
                candidate["object_id"],
            )
        )
        return candidates

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
            if (
                xy_distance(eef_pose["position"], target_position)
                <= EEF_POSITION_TOLERANCE_M
            ):
                z_error = abs(
                    float(eef_pose["position"]["z"]) - float(target_position["z"])
                )
                if z_error <= EEF_POSITION_TOLERANCE_M:
                    return True
        return False


class MultimodalPlanner:
    def __init__(self, mode: str, model: str, temperature: float) -> None:
        self.mode = mode
        self.model = model
        self.temperature = temperature
        self.system_prompt = make_planner_system_prompt()

    def plan(
        self,
        command: str,
        observation: dict[str, Any],
        image_bytes: bytes,
        history: list[dict[str, Any]],
        feedback: str = "",
    ) -> Decision:
        user_prompt = make_planner_user_prompt(command, observation, history, feedback)
        if self.mode == "ollama":
            content = self._call_ollama(user_prompt, image_bytes)
        else:
            content = self._call_openai(user_prompt, image_bytes)
        parsed = _extract_json_object(content)
        return self._decision_from_dict(parsed)

    def _call_openai(self, user_prompt: str, image_bytes: bytes) -> str:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY or CHATGPT_API_KEY is required for --mode openai."
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image_bytes)},
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            response_payload = _post_json(
                OPENAI_API_URL,
                payload,
                {"Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
        except RuntimeError as exc:
            if "response_format" not in str(exc):
                raise
            payload.pop("response_format", None)
            response_payload = _post_json(
                OPENAI_API_URL,
                payload,
                {"Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
        choices = response_payload.get("choices") or []
        message = (
            choices[0].get("message")
            if choices and isinstance(choices[0], dict)
            else {}
        )
        content = str((message or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty message.")
        return content

    def _call_ollama(self, user_prompt: str, image_bytes: bytes) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [base64.b64encode(image_bytes).decode("utf-8")],
                },
            ],
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        response_payload = _post_json(OLLAMA_URL, payload, {})
        message = response_payload.get("message") or {}
        content = str(
            message.get("content") or response_payload.get("response") or ""
        ).strip()
        if not content:
            raise RuntimeError("Ollama returned an empty message.")
        return content

    @staticmethod
    def _decision_from_dict(parsed: dict[str, Any]) -> Decision:
        raw_response = parsed
        sequence = parsed.get("action_sequence")
        if not isinstance(sequence, list):
            sequence = parsed.get("actions")
        if isinstance(sequence, list) and sequence:
            first_action = sequence[0]
            if isinstance(first_action, dict):
                parsed = {**first_action, "done": bool(raw_response.get("done", False))}
                parsed.setdefault("reason", raw_response.get("reason", ""))

        target_pose = normalize_pose_dict(parsed.get("target_pose"))
        robot_id = parsed.get("robot_id")
        if robot_id is not None:
            robot_id = str(robot_id).strip().lower()
        action = parsed.get("action")
        if action is not None:
            action = str(action).strip()
        target_object_id = parsed.get("target_object_id")
        if target_object_id is not None:
            target_object_id = str(target_object_id).strip() or None
        return Decision(
            done=bool(parsed.get("done", False)),
            robot_id=robot_id,
            action=action,
            target_pose=target_pose,
            target_object_id=target_object_id,
            intent=str(parsed.get("intent") or "").strip().lower(),
            reason=str(parsed.get("reason") or "").strip(),
            raw=raw_response,
        )


class DecisionValidator:
    def __init__(self) -> None:
        self.last_feedback = ""

    def validate_or_repair(
        self,
        decision: Decision,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
    ) -> Decision:
        if decision.done:
            return decision
        if decision.robot_id not in CONTROL_SERVICE_TOPICS:
            raise ValueError(f"Invalid robot_id: {decision.robot_id!r}")
        if decision.action == ACTION_PLACING:
            raise ValueError(
                "Action 'Placing' is forbidden. Use Moving to a computed drop pose, then Realease."
            )
        if decision.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {decision.action!r}")

        if decision.action == ACTION_MOVING:
            if decision.target_pose is None:
                raise ValueError("Moving requires target_pose.")
            target_object = self._object_for_decision(
                decision, observation, held_objects
            )
            if decision.intent == "pick":
                if target_object is None:
                    raise ValueError(
                        "Pick Moving requires target_object_id for a visible object."
                    )
                self._validate_robot_can_reach_object(
                    decision, target_object, observation
                )
                decision.target_pose = {
                    "position": dict(target_object["pose"]["position"]),
                    "orientation": dict(VERTICAL_EEF_ORIENTATION),
                }
            elif decision.intent == "drop":
                held_object_id = held_objects.get(str(decision.robot_id))
                if not held_object_id:
                    raise ValueError(
                        "Cannot use intent='drop' because this robot is not holding any object. "
                        "First use Moving intent='pick', then Grip."
                    )
                decision.target_pose = self._validate_or_fallback_drop_pose(
                    decision, observation, held_objects
                )
            else:
                decision.target_pose["orientation"] = dict(VERTICAL_EEF_ORIENTATION)
        elif decision.action == ACTION_CENTERING:
            decision.target_pose = {
                "position": DEFAULT_TABLE_CENTER,
                "orientation": VERTICAL_EEF_ORIENTATION,
            }
        elif decision.action == ACTION_GRIP:
            if decision.intent and decision.intent != "grip":
                raise ValueError("Grip action must use intent='grip'.")
            target_object = self._object_for_decision(
                decision, observation, held_objects
            )
            if target_object is not None:
                self._validate_robot_can_reach_object(
                    decision, target_object, observation
                )
        elif decision.action == ACTION_REALEASE:
            if not held_objects.get(str(decision.robot_id)):
                raise ValueError(
                    "Cannot Realease because the selected robot is not holding an object."
                )
            if decision.intent and decision.intent != "release":
                raise ValueError("Realease action must use intent='release'.")
        return decision

    @staticmethod
    def _validate_robot_can_reach_object(
        decision: Decision, obj: dict[str, Any], observation: dict[str, Any]
    ) -> None:
        robot = observation.get("robots", {}).get(str(decision.robot_id), {})
        reachable_objects = set(robot.get("reachable_objects", []))
        if "reachable_objects" in robot and obj["id"] not in reachable_objects:
            options = [
                {
                    "robot_id": candidate.get("robot_id"),
                    "reachable": candidate.get("reachable"),
                    "xy_distance": candidate.get("xy_distance"),
                }
                for pick_candidate in observation.get("recommended_pick_candidates", [])
                if pick_candidate.get("object_id") == obj["id"]
                for candidate in pick_candidate.get("robot_options", [])
            ]
            raise ValueError(
                f"Object {obj['id']} is not in robot_id='{decision.robot_id}' reachable_objects. "
                f"Choose a reachable robot from these options: {options}"
            )

    def _validate_or_fallback_drop_pose(
        self,
        decision: Decision,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
    ) -> dict[str, Any]:
        obj = self._object_for_decision(decision, observation, held_objects)
        if obj is None:
            raise ValueError("Drop intent requires a known held or target object.")
        goal = observation["goals"].get(obj["color"])
        if goal is None:
            raise ValueError(f"No {obj['color']} goal is available.")
        pose = decision.target_pose
        assert pose is not None
        occupied = [
            candidate
            for candidate in observation["objects"]
            if candidate["id"] != obj["id"]
            and point_in_goal(candidate["pose"]["position"], goal, margin=0.0)
        ]
        if self._drop_pose_is_valid(pose, goal, occupied):
            return pose

        fallback = self._fallback_drop_pose(goal, occupied)
        self.last_feedback = (
            f"LLM drop pose was outside the {obj['color']} goal or too close to another object; "
            f"using fallback pose {fallback['position']}."
        )
        return fallback

    @staticmethod
    def _object_for_decision(
        decision: Decision,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
    ) -> dict[str, Any] | None:
        if decision.intent == "drop":
            object_id = (
                held_objects.get(str(decision.robot_id)) or decision.target_object_id
            )
        else:
            object_id = decision.target_object_id or held_objects.get(
                str(decision.robot_id)
            )
        for obj in observation["objects"]:
            if obj["id"] == object_id or obj["namespace"] == object_id:
                return obj
        return None

    @staticmethod
    def _drop_pose_is_valid(
        pose: dict[str, Any], goal: dict[str, Any], occupied: list[dict[str, Any]]
    ) -> bool:
        position = pose["position"]
        if not point_in_goal(position, goal, margin=GOAL_MARGIN_M):
            return False
        for obj in occupied:
            if xy_distance(position, obj["pose"]["position"]) < MIN_DROP_CLEARANCE_M:
                return False
        return True

    @staticmethod
    def _fallback_drop_pose(
        goal: dict[str, Any], occupied: list[dict[str, Any]]
    ) -> dict[str, Any]:
        center = goal["pose"]["position"]
        scale = goal["scale"]
        half_x = max(0.0, float(scale["x"]) * 0.5 - GOAL_MARGIN_M)
        half_y = max(0.0, float(scale["y"]) * 0.5 - GOAL_MARGIN_M)
        step = MIN_DROP_CLEARANCE_M
        candidates: list[dict[str, float]] = []
        x_count = max(1, int((half_x * 2.0) // step) + 1)
        y_count = max(1, int((half_y * 2.0) // step) + 1)
        for xi in range(x_count):
            for yi in range(y_count):
                x_offset = -half_x + (2.0 * half_x * xi / max(1, x_count - 1))
                y_offset = -half_y + (2.0 * half_y * yi / max(1, y_count - 1))
                candidates.append(
                    {
                        "x": float(center["x"]) + x_offset,
                        "y": float(center["y"]) + y_offset,
                        "z": float(center["z"]),
                    }
                )
        candidates.append(
            {"x": float(center["x"]), "y": float(center["y"]), "z": float(center["z"])}
        )
        candidates.sort(key=lambda point: xy_distance(point, center))
        for candidate in candidates:
            if all(
                xy_distance(candidate, obj["pose"]["position"]) >= MIN_DROP_CLEARANCE_M
                for obj in occupied
            ):
                return {"position": candidate, "orientation": VERTICAL_EEF_ORIENTATION}
        return {
            "position": {
                "x": float(center["x"]),
                "y": float(center["y"]),
                "z": float(center["z"]),
            },
            "orientation": VERTICAL_EEF_ORIENTATION,
        }


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
        self.failed_motion_targets: dict[str, int] = {}
        self.validator = DecisionValidator()

    def task_complete(self, observation: dict[str, Any]) -> bool:
        objects = observation.get("objects", [])
        return bool(objects) and all(bool(obj.get("sorted")) for obj in objects)

    def execute(
        self, decision: Decision, observation: dict[str, Any]
    ) -> ExecutionResult:
        if decision.done:
            result = ExecutionResult(True, "LLM marked task complete.", decision)
            self._append_history(result)
            return result

        if self.args.dry_run:
            result = ExecutionResult(True, "dry-run: service call skipped", decision)
            self._update_held_state(decision, observation, result.success)
            self._append_history(result)
            return result

        assert decision.robot_id is not None
        assert decision.action is not None
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
        self._update_failed_motion_targets(decision, result)
        self._append_history(result)
        return result

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
        if decision.action == ACTION_GRIP:
            self.held_objects[robot_id] = (
                decision.target_object_id
                or self.pending_grip_targets.get(robot_id)
                or self._nearest_object_to_eef(robot_id, observation)
            )
            self.pending_grip_targets[robot_id] = None
        elif decision.action == ACTION_REALEASE:
            self.held_objects[robot_id] = None
            self.pending_grip_targets[robot_id] = None
        elif decision.action == ACTION_MOVING and decision.intent == "pick":
            self.pending_grip_targets[robot_id] = decision.target_object_id

    def _update_failed_motion_targets(
        self, decision: Decision, result: ExecutionResult
    ) -> None:
        if decision.action != ACTION_MOVING or result.success:
            return
        key = "|".join(
            [
                str(decision.robot_id),
                str(decision.intent),
                str(decision.target_object_id),
            ]
        )
        self.failed_motion_targets[key] = self.failed_motion_targets.get(key, 0) + 1

    @staticmethod
    def _nearest_object_to_eef(
        robot_id: str, observation: dict[str, Any]
    ) -> str | None:
        eef_pose = observation["robots"].get(robot_id, {}).get("end_effector_pose")
        if not eef_pose:
            return None
        eef_position = eef_pose["position"]
        nearest: tuple[float, str] | None = None
        for obj in observation.get("objects", []):
            distance = xy_distance(eef_position, obj["pose"]["position"])
            if nearest is None or distance < nearest[0]:
                nearest = (distance, obj["id"])
        if nearest is not None and nearest[0] <= 0.09:
            return nearest[1]
        return None

    def _append_history(self, result: ExecutionResult) -> None:
        decision = result.decision
        self.history.append(
            {
                "time": time.time(),
                "success": result.success,
                "message": result.message,
                "robot_id": decision.robot_id,
                "action": decision.action,
                "target_pose": decision.target_pose,
                "target_object_id": decision.target_object_id,
                "intent": decision.intent,
                "reason": decision.reason,
                "held_objects": dict(self.held_objects),
                "pending_grip_targets": dict(self.pending_grip_targets),
                "failed_motion_targets": dict(self.failed_motion_targets),
            }
        )


def plan_valid_decision(
    planner: MultimodalPlanner,
    executor: Executor,
    args: argparse.Namespace,
    observation: dict[str, Any],
    image_bytes: bytes,
    feedback: str,
    reporter: "Reporter",
) -> tuple[Decision, str]:
    current_feedback = feedback
    for attempt in range(1, 4):
        decision = planner.plan(
            args.command, observation, image_bytes, executor.history, current_feedback
        )
        try:
            return (
                executor.validator.validate_or_repair(
                    decision,
                    observation,
                    executor.held_objects,
                ),
                "",
            )
        except ValueError as exc:
            current_feedback = (
                f"Your previous first action was rejected before execution: {exc}. "
                "Return a corrected action_sequence whose first action is immediately executable. "
                "Remember: first pick with Moving, then Grip, then drop with Moving only after held_objects is set, then Realease."
            )
            reporter.error(f"Rejected LLM decision attempt {attempt}: {exc}")
    raise RuntimeError(current_feedback)


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

    def decision(self, step: int, decision: Decision) -> None:
        rows = {
            "step": step,
            "done": decision.done,
            "robot": decision.robot_id,
            "action": decision.action,
            "intent": decision.intent,
            "target": decision.target_object_id,
            "reason": decision.reason,
        }
        if self.verbose and decision.target_pose is not None:
            rows["target_pose"] = json.dumps(decision.target_pose, ensure_ascii=False)
        self.status("LLM decision", rows)


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


def parse_args() -> argparse.Namespace:
    default_mode = os.getenv("MODE", "openai").strip().lower()
    if default_mode not in {"openai", "ollama"}:
        default_mode = "openai"
    parser = argparse.ArgumentParser(
        description="Multimodal ROS2 CLI that lets an LLM control the dual-Franka sorting task."
    )
    parser.add_argument(
        "--command", default=DEFAULT_COMMAND, help="High-level natural language task."
    )
    parser.add_argument(
        "--mode",
        choices=("openai", "ollama"),
        default=default_mode,
        help="Vision LLM provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to OPENAI_MODEL or OLLAMA_MODEL.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=80, help="Maximum LLM action steps."
    )
    parser.add_argument(
        "--once", action="store_true", help="Plan and execute only one step."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call the LLM but skip ROS2 service calls.",
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
        help="Seconds to wait after non-moving actions.",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="Directory to save PNG frames sent to the LLM.",
    )
    parser.add_argument(
        "--plain", action="store_true", help="Disable rich terminal UI."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional debugging details."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="LLM sampling temperature."
    )
    return parser.parse_args()


def save_frame(directory: Path, step: int, image_bytes: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"step_{step:03d}.png").write_bytes(image_bytes)


def run(args: argparse.Namespace) -> int:
    if ROS_IMPORT_ERROR is not None:
        print(
            "ROS2 imports failed. Source the ROS2 workspace before running this CLI, "
            "then run it with the ROS2 Python interpreter, for example: "
            "source install/setup.bash && /usr/bin/python3 web-app/cli.py",
            file=sys.stderr,
        )
        print(f"Import error: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 2

    model = args.model or (OLLAMA_MODEL if args.mode == "ollama" else OPENAI_MODEL)
    reporter = Reporter(plain=args.plain, verbose=args.verbose)
    reporter.status(
        "Startup",
        {
            "mode": args.mode,
            "model": model,
            "dry_run": args.dry_run,
            "command": args.command,
        },
    )

    rclpy.init()
    node = RosWorldNode()
    planner = MultimodalPlanner(
        mode=args.mode, model=model, temperature=args.temperature
    )
    executor = Executor(node, args, reporter)
    feedback = ""
    try:
        wait_for_ready(node, reporter)
        for step in range(1, int(args.max_steps) + 1):
            observation, image_bytes = node.snapshot()
            observation["runtime_state"] = {
                "held_objects": dict(executor.held_objects),
                "pending_grip_targets": dict(executor.pending_grip_targets),
                "failed_motion_targets": dict(executor.failed_motion_targets),
                "recursive_execution_note": (
                    "The LLM may return action_sequence, but only the first action is executed before replanning."
                ),
            }
            if args.save_frames is not None:
                save_frame(args.save_frames, step, image_bytes)

            reporter.status(
                "Observation",
                {
                    "step": step,
                    "unsorted": observation["summary"]["unsorted_count"],
                    "objects": observation["summary"]["object_count"],
                    "held": executor.held_objects,
                },
            )

            if executor.task_complete(observation):
                reporter.info(
                    "Task complete: all objects are in their matching goal regions."
                )
                return 0

            try:
                decision, feedback = plan_valid_decision(
                    planner,
                    executor,
                    args,
                    observation,
                    image_bytes,
                    feedback,
                    reporter,
                )
            except RuntimeError as exc:
                reporter.error(f"Unable to obtain a valid LLM decision: {exc}")
                return 1

            reporter.decision(step, decision)
            if decision.done:
                reporter.info("LLM reported completion.")
                return 0

            result = executor.execute(decision, observation)
            reporter.status(
                "Execution result",
                {"success": result.success, "message": result.message},
            )
            if executor.validator.last_feedback:
                reporter.info(executor.validator.last_feedback)
                executor.validator.last_feedback = ""
            if not result.success:
                feedback = (
                    "The previous executed first action failed. "
                    f"robot_id={decision.robot_id}, action={decision.action}, intent={decision.intent}, "
                    f"target_object_id={decision.target_object_id}, message={result.message}. "
                    f"failed_motion_targets={executor.failed_motion_targets}. "
                    "Use the updated observation and recommended_pick_candidates. "
                    "If this exact Moving target failed repeatedly, choose a different reachable object or robot."
                )
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
