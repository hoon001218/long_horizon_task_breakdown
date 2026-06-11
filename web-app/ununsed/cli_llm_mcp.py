#!/usr/bin/python3
"""LLM-semantic MCP-style CLI planner/executor for the dual-Franka scene.

This version uses the LLM only at the semantic-command boundary.  The LLM receives
the natural-language command plus the current ROS2 scene state and must choose one
of a small set of MCP-like semantic tools:

- place_objects
- evacuate_objects_from_goals
- reject_task

The deterministic planner then decomposes the accepted semantic tool call into
safe ROS service primitives: Moving, Grip, Realease, and Homing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import math
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


@dataclass(frozen=True)
class TaskSpec:
    """LLM-interpreted task definition.

    The command is interpreted into one of a small set of semantic task tools.
    The deterministic planner then converts that semantic tool call into safe
    ROS service primitives.  This is intentionally similar to an MCP-style
    boundary: the LLM can choose a well-defined semantic action, but it cannot
    directly issue arbitrary robot service calls.
    """

    raw_command: str
    task_type: str  # "place_to_goal" or "evacuate_from_goal"
    active_object_colors: set[str]
    destination_by_object_color: dict[str, str] = field(default_factory=dict)
    source_goal_colors: set[str] = field(default_factory=set)
    destination_region: str = ""
    reject_reason: str = ""

    def is_relevant_object(self, obj: dict[str, Any]) -> bool:
        return str(obj.get("color")) in self.active_object_colors

    def is_pending_object(self, obj: dict[str, Any], goals: dict[str, Any]) -> bool:
        if not self.is_relevant_object(obj):
            return False
        return not self.object_satisfies(obj, goals)

    def destination_for(self, obj: dict[str, Any] | str) -> str | None:
        if self.task_type != "place_to_goal":
            return None
        color = obj if isinstance(obj, str) else str(obj.get("color"))
        return self.destination_by_object_color.get(str(color))

    def source_goals_for_check(self, goals: dict[str, Any]) -> set[str]:
        if self.source_goal_colors:
            return set(self.source_goal_colors)
        return set(goals.keys())

    def object_satisfies(self, obj: dict[str, Any], goals: dict[str, Any]) -> bool:
        if not self.is_relevant_object(obj):
            return True
        if self.task_type == "place_to_goal":
            destination = self.destination_for(obj)
            goal = goals.get(destination) if destination else None
            return bool(goal and point_in_goal(obj["pose"]["position"], goal, margin=0.0))
        if self.task_type == "evacuate_from_goal":
            source_goals = self.source_goals_for_check(goals)
            for goal_color in source_goals:
                goal = goals.get(goal_color)
                if goal and point_in_goal(obj["pose"]["position"], goal, margin=0.0):
                    return False
            return True
        return False

    def final_target_description(self, obj: dict[str, Any]) -> str:
        if self.task_type == "place_to_goal":
            return f"{self.destination_for(obj)} goal"
        if self.task_type == "evacuate_from_goal":
            goals = sorted(self.source_goal_colors) if self.source_goal_colors else ["any goal"]
            return f"outside {goals}"
        return "unknown target"

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_command": self.raw_command,
            "task_type": self.task_type,
            "active_object_colors": sorted(self.active_object_colors),
            "destination_by_object_color": dict(sorted(self.destination_by_object_color.items())),
            "source_goal_colors": sorted(self.source_goal_colors),
            "destination_region": self.destination_region,
            "reject_reason": self.reject_reason,
        }


class SemanticTaskInterpreter:
    """LLM semantic interpreter with MCP-like tool contracts.

    The LLM does not emit low-level robot commands.  It chooses exactly one
    semantic task tool:
    - place_objects
    - evacuate_objects_from_goals
    - reject_task

    The result is converted into TaskSpec and then executed by the deterministic
    handover planner.
    """

    SUPPORTED_COLORS = {"red", "blue"}
    SUPPORTED_TASK_TYPES = {"place_to_goal", "evacuate_from_goal"}

    def __init__(self, args: argparse.Namespace, reporter: "Reporter") -> None:
        self.args = args
        self.reporter = reporter
        self.mode = str(getattr(args, "semantic_mode", "auto") or "auto").lower()

    def interpret(self, command: str, observation: dict[str, Any], image_bytes: bytes | None = None) -> TaskSpec:
        if self.mode == "rule":
            return RuleTaskInterpreter().parse(command)

        errors: list[str] = []
        modes: list[str]
        if self.mode == "auto":
            modes = ["openai", "ollama", "rule"]
        else:
            modes = [self.mode]

        for mode in modes:
            try:
                if mode == "openai":
                    return self._interpret_openai(command, observation, image_bytes)
                if mode == "ollama":
                    return self._interpret_ollama(command, observation, image_bytes)
                if mode == "rule":
                    return RuleTaskInterpreter().parse(command)
                errors.append(f"unknown semantic mode: {mode}")
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
                if self.mode != "auto":
                    break

        raise ValueError("Failed to interpret task command. " + " | ".join(errors))

    def _prompt_payload(self, command: str, observation: dict[str, Any]) -> tuple[str, str]:
        system = (
            "You are the semantic command interpreter for a dual-Franka robot task system. "
            "You must convert a user's natural-language command into exactly one MCP-like semantic tool call. "
            "Do not output robot motion primitives. Do not invent colors, goals, objects, or actions. "
            "Use the provided scene state as ground truth. Return only one JSON object.\n\n"
            "Available semantic tools:\n"
            "1) place_objects: place selected object colors into specified goal colors.\n"
            "   JSON form: {\"tool\":\"place_objects\", \"arguments\":{\"object_colors\":[\"red\"|\"blue\"], "
            "\"destination_by_object_color\":{\"red\":\"red\"|\"blue\", \"blue\":\"red\"|\"blue\"}}, "
            "\"reason\":\"...\"}\n"
            "2) evacuate_objects_from_goals: move selected object colors out of goal regions and leave them on the table outside goals.\n"
            "   JSON form: {\"tool\":\"evacuate_objects_from_goals\", \"arguments\":{\"object_colors\":[\"red\"|\"blue\"], "
            "\"source_goal_colors\":[\"red\"|\"blue\"] or [], \"destination_region\":\"outside_goals\"}, \"reason\":\"...\"}\n"
            "   Use an empty source_goal_colors list when the user says just 'goal(s)' without specifying red/blue goal.\n"
            "3) reject_task: reject commands that cannot be represented by the two tools above, are ambiguous, or request unsafe behavior.\n"
            "   JSON form: {\"tool\":\"reject_task\", \"arguments\":{\"reason\":\"...\"}}\n\n"
            "Interpret examples:\n"
            "- '파란 물체를 목표점에서 전부 밖으로 빼내라' => evacuate_objects_from_goals, object_colors=[blue], source_goal_colors=[], destination_region=outside_goals.\n"
            "- '파란 물체를 빨간 목표점에 넣어라' => place_objects, object_colors=[blue], destination_by_object_color={blue:red}.\n"
            "- '빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라' => place_objects with red:red and blue:blue.\n"
            "- '모든 물체를 파란 목표점에 넣어라' => place_objects with red:blue and blue:blue.\n"
        )
        scene_summary = {
            "available_object_colors": sorted({str(obj.get("color")) for obj in observation.get("objects", [])}),
            "available_goal_colors": sorted(observation.get("goals", {}).keys()),
            "objects": [
                {
                    "id": obj.get("id"),
                    "color": obj.get("color"),
                    "shape": obj.get("shape"),
                    "inside_goal": obj.get("inside_goal"),
                    "position": obj.get("pose", {}).get("position"),
                }
                for obj in observation.get("objects", [])
            ],
            "goals": {
                color: {
                    "id": goal.get("id"),
                    "position": goal.get("pose", {}).get("position"),
                    "scale": goal.get("scale"),
                }
                for color, goal in observation.get("goals", {}).items()
            },
        }
        user = json.dumps(
            {
                "user_command": command,
                "scene_state": scene_summary,
                "required_output": "one JSON object selecting exactly one semantic tool",
            },
            ensure_ascii=False,
            indent=2,
        )
        return system, user

    def _interpret_openai(self, command: str, observation: dict[str, Any], image_bytes: bytes | None) -> TaskSpec:
        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or CHATGPT_API_KEY is not set")
        model = self.args.semantic_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        system, user = self._prompt_payload(command, observation)
        content: Any = [{"type": "text", "text": user}]
        if image_bytes and not getattr(self.args, "no_semantic_image", False):
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_bytes)}})
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": float(getattr(self.args, "semantic_temperature", 0.0)),
            "response_format": {"type": "json_object"},
        }
        response = post_json(url, payload, {"Authorization": f"Bearer {api_key}"})
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        parsed = extract_json_object(str((message or {}).get("content") or ""))
        return self._task_spec_from_tool_call(command, parsed, observation)

    def _interpret_ollama(self, command: str, observation: dict[str, Any], image_bytes: bytes | None) -> TaskSpec:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        model = self.args.semantic_model or os.getenv("OLLAMA_MODEL", "llava")
        system, user = self._prompt_payload(command, observation)
        message: dict[str, Any] = {"role": "user", "content": user}
        if image_bytes and not getattr(self.args, "no_semantic_image", False):
            message["images"] = [base64.b64encode(image_bytes).decode("utf-8")]
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                message,
            ],
            "format": "json",
            "options": {"temperature": float(getattr(self.args, "semantic_temperature", 0.0))},
        }
        response = post_json(url, payload, {})
        msg = response.get("message") or {}
        parsed = extract_json_object(str(msg.get("content") or response.get("response") or ""))
        return self._task_spec_from_tool_call(command, parsed, observation)

    def _task_spec_from_tool_call(self, command: str, parsed: dict[str, Any], observation: dict[str, Any]) -> TaskSpec:
        tool = str(parsed.get("tool") or parsed.get("name") or "").strip()
        args = parsed.get("arguments") or parsed.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"semantic tool arguments must be an object: {parsed}")

        available_object_colors = {str(obj.get("color")) for obj in observation.get("objects", [])}
        available_goal_colors = set(observation.get("goals", {}).keys())

        if tool == "reject_task":
            reason = str(args.get("reason") or parsed.get("reason") or "command rejected")
            raise ValueError(reason)

        if tool == "place_objects":
            object_colors = self._clean_color_set(args.get("object_colors"), available_object_colors, "object_colors")
            mapping_raw = args.get("destination_by_object_color") or {}
            if not isinstance(mapping_raw, dict):
                raise ValueError("destination_by_object_color must be an object")
            mapping: dict[str, str] = {}
            for color in object_colors:
                dst = str(mapping_raw.get(color) or "").strip().lower()
                if dst not in self.SUPPORTED_COLORS:
                    raise ValueError(f"missing or invalid destination goal for {color}: {dst!r}")
                mapping[color] = dst
            missing_goal = sorted(set(mapping.values()) - available_goal_colors)
            if missing_goal:
                raise ValueError(f"command requires unavailable goal color(s): {missing_goal}")
            return TaskSpec(command, "place_to_goal", object_colors, mapping, destination_region="goal")

        if tool == "evacuate_objects_from_goals":
            object_colors = self._clean_color_set(args.get("object_colors"), available_object_colors, "object_colors")
            source_raw = args.get("source_goal_colors") or []
            source_goals = self._clean_color_set(source_raw, available_goal_colors, "source_goal_colors") if source_raw else set()
            destination_region = str(args.get("destination_region") or "outside_goals").strip().lower()
            if destination_region not in {"outside_goals", "outside_goal", "table_outside_goals"}:
                raise ValueError(f"unsupported evacuation destination_region: {destination_region}")
            return TaskSpec(
                command,
                "evacuate_from_goal",
                object_colors,
                {},
                source_goal_colors=source_goals,
                destination_region="outside_goals",
            )

        raise ValueError(f"unsupported semantic tool: {tool!r}; parsed={parsed}")

    def _clean_color_set(self, value: Any, available: set[str], field_name: str) -> set[str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field_name} must be a non-empty list")
        colors = {str(item).strip().lower() for item in value}
        unsupported = sorted(colors - self.SUPPORTED_COLORS)
        if unsupported:
            raise ValueError(f"unsupported color(s) in {field_name}: {unsupported}")
        missing = sorted(colors - available)
        if missing:
            raise ValueError(f"{field_name} references color(s) not present in scene: {missing}")
        return colors


class RuleTaskInterpreter:
    """Limited fallback interpreter.  It is deliberately conservative."""

    COLOR_PATTERNS: dict[str, tuple[str, ...]] = {
        "red": ("red", "빨강", "빨간", "빨갛", "붉은", "적색"),
        "blue": ("blue", "파랑", "파란", "파랗", "청색"),
    }
    ALL_TOKENS = ("all", "every", "전체", "모든", "전부", "모두", "다 ")
    EVACUATE_TOKENS = ("밖", "빼", "꺼내", "remove", "out of", "outside", "evacuate")

    def parse(self, command: str) -> TaskSpec:
        command = (command or DEFAULT_COMMAND).strip() or DEFAULT_COMMAND
        lowered = command.lower()
        mentions = self._color_mentions(lowered)
        has_all = any(token in lowered for token in self.ALL_TOKENS)
        if not mentions:
            raise ValueError("Task command does not specify supported colors red/blue.")

        if any(token in lowered for token in self.EVACUATE_TOKENS):
            # For evacuation, the first explicitly mentioned color is the object color.
            object_color = mentions[0][1]
            source_goals = {mentions[1][1]} if len(mentions) >= 2 and "목표" in lowered else set()
            return TaskSpec(
                raw_command=command,
                task_type="evacuate_from_goal",
                active_object_colors={object_color},
                source_goal_colors=source_goals,
                destination_region="outside_goals",
            )

        mapping: dict[str, str] = {}
        if len(mentions) >= 4 and len(mentions) % 2 == 0:
            for i in range(0, len(mentions), 2):
                mapping[mentions[i][1]] = mentions[i + 1][1]
        elif len(mentions) == 2:
            mapping[mentions[0][1]] = mentions[1][1]
        elif has_all and len(mentions) == 1:
            dst = mentions[0][1]
            mapping = {"red": dst, "blue": dst}
        else:
            raise ValueError("Ambiguous task command. Use explicit source and destination colors.")
        return TaskSpec(command, "place_to_goal", set(mapping.keys()), mapping, destination_region="goal")

    def _color_mentions(self, text: str) -> list[tuple[int, str]]:
        mentions: list[tuple[int, str]] = []
        for color, patterns in self.COLOR_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(re.escape(pattern.lower()), text):
                    mentions.append((match.start(), color))
        mentions.sort(key=lambda item: item[0])
        deduped: list[tuple[int, str]] = []
        for pos, color in mentions:
            if deduped and color == deduped[-1][1] and abs(pos - deduped[-1][0]) <= 2:
                continue
            deduped.append((pos, color))
        return deduped


def object_satisfies_task(obj: dict[str, Any], goals: dict[str, Any], task_spec: TaskSpec) -> bool:
    return task_spec.object_satisfies(obj, goals)


def task_complete_by_vision(
    observation: dict[str, Any],
    task_spec: TaskSpec,
    held_objects: dict[str, str | None] | None = None,
    pending_grip_targets: dict[str, str | None] | None = None,
) -> tuple[bool, str]:
    if held_objects and any(value is not None for value in held_objects.values()):
        return False, f"robot still holds object(s): {held_objects}"
    if pending_grip_targets and any(value is not None for value in pending_grip_targets.values()):
        return False, f"pending grip target(s) remain: {pending_grip_targets}"

    goals = observation.get("goals", {})
    if task_spec.task_type == "place_to_goal":
        missing_goals = sorted(set(task_spec.destination_by_object_color.values()) - set(goals.keys()))
        if missing_goals:
            return False, f"required goal marker(s) missing: {missing_goals}"
    elif task_spec.task_type == "evacuate_from_goal":
        missing_sources = sorted(task_spec.source_goal_colors - set(goals.keys()))
        if missing_sources:
            return False, f"source goal marker(s) missing: {missing_sources}"

    relevant_objects = [obj for obj in observation.get("objects", []) if task_spec.is_relevant_object(obj)]
    if not relevant_objects:
        return False, f"no visible relevant objects for colors {sorted(task_spec.active_object_colors)}"

    failures: list[str] = []
    for obj in relevant_objects:
        if not task_spec.object_satisfies(obj, goals):
            failures.append(
                f"{obj['id']} color={obj.get('color')} target={task_spec.final_target_description(obj)} "
                f"inside_goal={obj.get('inside_goal')} position={obj['pose']['position']}"
            )
    if failures:
        return False, "; ".join(failures)
    return True, f"all relevant objects satisfy {task_spec.as_dict()} and no robot is holding an object"


def validate_task_spec_against_scene(task_spec: TaskSpec, observation: dict[str, Any]) -> None:
    visible_colors = {str(obj.get("color")) for obj in observation.get("objects", [])}
    missing_object_colors = sorted(task_spec.active_object_colors - visible_colors)
    if missing_object_colors:
        raise ValueError(f"Command targets object color(s) not visible in the scene: {missing_object_colors}")
    if task_spec.task_type == "place_to_goal":
        missing_goals = sorted(set(task_spec.destination_by_object_color.values()) - set(observation.get("goals", {}).keys()))
        if missing_goals:
            raise ValueError(f"Command requires goal color(s) not visible in the scene: {missing_goals}")
    if task_spec.task_type == "evacuate_from_goal":
        missing_sources = sorted(task_spec.source_goal_colors - set(observation.get("goals", {}).keys()))
        if missing_sources:
            raise ValueError(f"Command references source goal color(s) not visible in the scene: {missing_sources}")


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
        raise ValueError("LLM semantic response must be a JSON object")
    return parsed


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
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


def image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"



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


def point_in_goal(point: dict[str, float], goal: dict[str, Any], margin: float = 0.0) -> bool:
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


def raised_pose(pose_or_position: dict[str, Any], lift_delta: float, safe_z: float) -> dict[str, Any]:
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
        self.create_subscription(PoseStamped, CAMERA_POSE_TOPIC, self._on_camera_pose, 10)

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
                "markers": self._table is not None and bool(self._goals) and bool(self._objects),
                "image": self._latest_image is not None,
                "camera_pose": self._camera_pose is not None,
                "robot_poses": all(robot_id in self._robot_poses for robot_id in ROBOT_POSE_TOPICS),
                "eef_poses": all(robot_id in self._eef_poses for robot_id in EEF_POSE_TOPICS),
            }

    def snapshot(self) -> tuple[dict[str, Any], bytes]:
        with self._lock:
            table = dict(self._table) if self._table is not None else self._default_table()
            goals = {key: dict(value) for key, value in self._goals.items()}
            objects = {key: dict(value) for key, value in self._objects.items()}
            latest_image = self._latest_image
            image_age = time.monotonic() - self._latest_image_time if self._latest_image_time else None
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
                "image_size": {"width": int(latest_image.width), "height": int(latest_image.height)},
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
            "pose": {"position": DEFAULT_TABLE_CENTER, "orientation": VERTICAL_EEF_ORIENTATION},
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

    def call_control_service(self, robot_id: str, action: str, target_pose: dict[str, Any] | None) -> tuple[bool, str]:
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

    def wait_for_eef_target(self, robot_id: str, target_pose: dict[str, Any], timeout_sec: float) -> bool:
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
                if abs(float(current["z"]) - float(target_position["z"])) <= EEF_POSITION_TOLERANCE_M:
                    return True
        return False


# ---------------------------------------------------------------------------
# Deterministic collaborative planner
# ---------------------------------------------------------------------------


class HandoverTaskPlanner:
    """Command-aware hierarchical task director + primitive decomposer.

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

    def __init__(self, args: argparse.Namespace, task_spec: TaskSpec) -> None:
        self.args = args
        self.task_spec = task_spec
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
        self,
        observation: dict[str, Any],
        held_objects: dict[str, str | None],
        pending_grip_targets: dict[str, str | None] | None = None,
    ) -> Decision:
        # Continue an already decomposed object-level plan before considering finish.
        # Otherwise the planner can stop while an object is merely being carried over
        # the goal or while a required post-release homing/verification sequence remains.
        if self.queue:
            decision = self.queue.pop(0)
            decision.metadata.setdefault("high_level_order", list(self.task_order))
            decision.metadata.setdefault("active_object", self.active_object_id)
            decision.metadata.setdefault("active_route", self.active_route_type)
            decision.metadata.setdefault("task_rationale", self.active_task_rationale)
            decision.metadata.setdefault("remaining_primitives", len(self.queue))
            return decision

        complete, complete_detail = task_complete_by_vision(
            observation,
            self.task_spec,
            held_objects=held_objects,
            pending_grip_targets=pending_grip_targets,
        )
        if complete:
            return Decision(
                None,
                ACTION_HOMING,
                done=True,
                intent="finish",
                reason=f"Vision/marker task completion check passed: {complete_detail}",
                metadata={"high_level_state": "complete", "task_spec": self.task_spec.as_dict()},
            )

        # Otherwise run the high-level director first, then decompose only the
        # top-ranked object-level task.
        task = self._select_high_level_task(observation, held_objects)
        self.active_object_id = task.object_id
        self.active_route_type = task.route.route_type
        self.active_task_rationale = task.rationale
        self.queue = self._build_steps(task.route, observation, task)
        if not self.queue:
            raise RuntimeError(f"Generated empty primitive plan for object {task.object_id}")
        decision = self.queue.pop(0)
        decision.metadata.setdefault("high_level_order", list(self.task_order))
        decision.metadata.setdefault("active_object", self.active_object_id)
        decision.metadata.setdefault("active_route", self.active_route_type)
        decision.metadata.setdefault("task_rationale", self.active_task_rationale)
        decision.metadata.setdefault("remaining_primitives", len(self.queue))
        return decision

    def _select_high_level_task(self, observation: dict[str, Any], held_objects: dict[str, str | None]) -> ObjectTask:
        tasks = self._make_high_level_plan(observation, held_objects)
        if not tasks:
            raise RuntimeError(
                "No feasible object-level task found. Check robot base poses, workspace radius, goals, and object markers."
            )
        self.task_plan = tasks
        self.task_order = [task.object_id for task in tasks]
        self.plan_revision += 1
        return tasks[0]

    def _make_high_level_plan(self, observation: dict[str, Any], held_objects: dict[str, str | None]) -> list[ObjectTask]:
        """Compute the object order before primitive decomposition.

        Ranking policy:
        - First, continue any object already held by a robot.
        - Then prefer feasible direct routes over handovers if scores are similar.
        - Otherwise use a route-distance score so the closest currently solvable
          object is processed first.
        - Ties are deterministic by object id.
        """
        unsorted = [
            obj for obj in observation.get("objects", [])
            if self.task_spec.is_relevant_object(obj)
            and not object_satisfies_task(obj, observation.get("goals", {}), self.task_spec)
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
                "evacuate_after_grip": 0,
                "direct": 1,
                "evacuate_direct": 1,
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
        destination_color = self.task_spec.final_target_description(obj) if obj else None
        if route.object_id in held_objects.values():
            holder = next((rid for rid, oid in held_objects.items() if oid == route.object_id), "unknown")
            return (
                f"Continue object {route.object_id}: it is already held by {holder}, "
                f"so finish the placement before selecting another object."
            )
        if route.route_type == "direct":
            return (
                f"Process {color} object {route.object_id} first: robot {route.source_robot_id} "
                f"can reach both the object and the corrected {destination_color} drop pose."
            )
        if route.route_type in {"evacuate_direct", "evacuate_after_grip"}:
            return (
                f"Evacuate {color} object {route.object_id}: robot {route.source_robot_id} "
                "will move it from the goal region to a safe table pose outside all relevant goals."
            )
        if route.route_type == "handover":
            return (
                f"Process {color} object {route.object_id} toward the {destination_color} goal via handover: robot {route.source_robot_id} "
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
        routes: list[RouteCandidate] = []

        if self.task_spec.task_type == "place_to_goal":
            destination_color = self.task_spec.destination_for(obj)
            if destination_color is None:
                return []
            goal = observation["goals"].get(destination_color)
            if goal is None:
                return []

            if object_id in held_objects.values():
                holder = next((rid for rid, held in held_objects.items() if held == object_id), None)
                if holder is not None:
                    drop_pose = self._choose_drop_pose(observation, goal, holder, obj)
                    if drop_pose is not None and self._robot_can_reach_pose(observation, holder, drop_pose):
                        routes.append(
                            RouteCandidate(
                                object_id=object_id,
                                route_type="direct_after_grip",
                                source_robot_id=holder,
                                destination_robot_id=holder,
                                score=0.0,
                                drop_pose=drop_pose,
                                reason="The object is already held; finish the command-specified goal drop sequence.",
                            )
                        )
                return routes

            source_robots = [
                rid for rid in CONTROL_SERVICE_TOPICS
                if self._robot_can_reach_position(observation, rid, obj["pose"]["position"])
            ]
            if not source_robots:
                return []

            buffer_pose = self._choose_shared_buffer_pose(observation, obj)

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
                        score=self._route_distance_score(observation, source, obj["pose"], drop_pose),
                        drop_pose=drop_pose,
                        reason="Single robot can reach both the object and corrected goal pose.",
                    )
                )

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
                                    self._route_distance_score(observation, source, obj["pose"], buffer_pose)
                                    + self._route_distance_score(observation, dest, buffer_pose, drop_pose)
                                    + 0.25
                                ),
                                drop_pose=drop_pose,
                                buffer_pose=buffer_pose,
                                reason="Object side and goal side are assigned to different robots through a shared buffer.",
                            )
                        )
            return routes

        if self.task_spec.task_type == "evacuate_from_goal":
            # Evacuation means: take the selected object out of the relevant goal region(s)
            # and deposit it on the table at a safe pose that is not inside any goal.
            if object_id in held_objects.values():
                holder = next((rid for rid, held in held_objects.items() if held == object_id), None)
                if holder is not None:
                    outside_pose = self._choose_outside_drop_pose(observation, holder, obj)
                    if outside_pose is not None:
                        routes.append(
                            RouteCandidate(
                                object_id=object_id,
                                route_type="evacuate_after_grip",
                                source_robot_id=holder,
                                destination_robot_id=holder,
                                score=0.0,
                                drop_pose=outside_pose,
                                reason="The object is already held; finish evacuating it outside the goal regions.",
                            )
                        )
                return routes

            source_robots = [
                rid for rid in CONTROL_SERVICE_TOPICS
                if self._robot_can_reach_position(observation, rid, obj["pose"]["position"])
            ]
            for source in source_robots:
                outside_pose = self._choose_outside_drop_pose(observation, source, obj)
                if outside_pose is None:
                    continue
                routes.append(
                    RouteCandidate(
                        object_id=object_id,
                        route_type="evacuate_direct",
                        source_robot_id=source,
                        destination_robot_id=source,
                        score=self._route_distance_score(observation, source, obj["pose"], outside_pose),
                        drop_pose=outside_pose,
                        reason="Single robot can move the object from the goal region to a safe outside-goal table pose.",
                    )
                )
            return routes

        return []

    def _build_steps(self, route: RouteCandidate, observation: dict[str, Any], task: ObjectTask) -> list[Decision]:
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
            "task_spec": self.task_spec.as_dict(),
            "target_goal_color": self.task_spec.destination_for(obj),
        }

        def attach(decisions: list[Decision]) -> list[Decision]:
            total = len(decisions)
            for idx, decision in enumerate(decisions, start=1):
                decision.metadata.update(common_meta)
                verify_spec = decision.metadata.get("verify_after_action")
                if isinstance(verify_spec, dict):
                    verify_spec.setdefault("expected_goal_color", common_meta.get("target_goal_color"))
                    verify_spec.setdefault("task_spec", self.task_spec.as_dict())
                decision.metadata["primitive_index"] = idx
                decision.metadata["primitive_count"] = total
            return decisions

        if route.route_type == "evacuate_after_grip":
            rid = route.source_robot_id
            return attach([
                self._moving(rid, object_above, obj["id"], "lift", "Current object is already held; lift it before moving outside the goal region."),
                self._moving(rid, drop_above, obj["id"], "outside_approach", "Move above the selected safe outside-goal pose."),
                self._moving(rid, drop_pose, obj["id"], "outside_drop", "Lower to the selected outside-goal table pose."),
                self._release(rid, obj["id"], "Release the object outside the relevant goal regions.", verify_type="outside_drop", expected_pose=drop_pose),
                self._homing(rid, obj["id"], "Return the robot home after evacuation."),
            ])

        if route.route_type == "evacuate_direct":
            rid = route.source_robot_id
            return attach([
                self._moving(rid, object_pose, obj["id"], "pick", "Move to the object that must be evacuated from the goal region."),
                self._grip(rid, obj["id"], "Grip the selected object."),
                self._moving(rid, object_above, obj["id"], "lift", "Lift before moving outside the goal region."),
                self._moving(rid, drop_above, obj["id"], "outside_approach", "Move above the selected safe outside-goal pose."),
                self._moving(rid, drop_pose, obj["id"], "outside_drop", "Lower to the selected outside-goal table pose."),
                self._release(rid, obj["id"], "Release the object outside the relevant goal regions.", verify_type="outside_drop", expected_pose=drop_pose),
                self._homing(rid, obj["id"], "Return the robot home after evacuation."),
            ])

        if route.route_type == "direct_after_grip":
            rid = route.source_robot_id
            return attach([
                self._moving(rid, object_above, obj["id"], "lift", "Current object is already held; lift it before lateral transfer."),
                self._moving(rid, drop_above, obj["id"], "drop_approach", "Move above the selected corrected goal pose."),
                self._moving(rid, drop_pose, obj["id"], "drop", "Lower to the selected goal pose with z offset for goal thickness."),
                self._release(rid, obj["id"], "Release the object inside its command-specified goal.", verify_type="final_drop", expected_pose=drop_pose),
                self._homing(rid, obj["id"], "Return the robot home after placing."),
            ])

        if route.route_type == "direct":
            rid = route.source_robot_id
            return attach([
                self._moving(rid, object_pose, obj["id"], "pick", "Move to the selected object pose."),
                self._grip(rid, obj["id"], "Grip the selected object."),
                self._moving(rid, object_above, obj["id"], "lift", "Lift before horizontal transfer."),
                self._moving(rid, drop_above, obj["id"], "drop_approach", "Move above the selected corrected goal pose."),
                self._moving(rid, drop_pose, obj["id"], "drop", "Lower to the corrected goal pose; z includes goal thickness offset."),
                self._release(rid, obj["id"], "Release the object inside the command-specified goal.", verify_type="final_drop", expected_pose=drop_pose),
                self._homing(rid, obj["id"], "Return the robot home after placing."),
            ])

        if route.route_type == "handover":
            if route.buffer_pose is None:
                raise RuntimeError("Handover route lacks a buffer pose.")
            source = route.source_robot_id
            dest = route.destination_robot_id
            buffer_pose = route.buffer_pose
            buffer_above = raised_pose(buffer_pose, self.args.lift_delta, safe_z)
            return attach([
                self._moving(source, object_pose, obj["id"], "pick", "Source robot moves to the selected object pose."),
                self._grip(source, obj["id"], "Source robot grips the object."),
                self._moving(source, object_above, obj["id"], "lift", "Source robot lifts the object for collision avoidance."),
                self._moving(source, buffer_above, obj["id"], "handover_approach", "Source robot moves above the shared table-center buffer."),
                self._moving(source, buffer_pose, obj["id"], "handover_place", "Source robot lowers the object onto the shared buffer."),
                self._release(source, obj["id"], "Source robot releases the object at the shared buffer.", verify_type="buffer_deposit", expected_pose=buffer_pose),
                self._homing(source, obj["id"], "Source robot homes after the handover deposit."),
                self._moving(dest, buffer_above, obj["id"], "handover_pick_approach", "Destination robot moves above the shared buffer."),
                self._moving(dest, buffer_pose, obj["id"], "handover_pick", "Destination robot lowers to the handed-over object."),
                self._grip(dest, obj["id"], "Destination robot grips the object from the buffer."),
                self._moving(dest, buffer_above, obj["id"], "lift", "Destination robot lifts the object before final transfer."),
                self._moving(dest, drop_above, obj["id"], "drop_approach", "Destination robot moves above the corrected matching goal pose."),
                self._moving(dest, drop_pose, obj["id"], "drop", "Destination robot lowers to the corrected goal pose; z includes goal thickness offset."),
                self._release(dest, obj["id"], "Destination robot releases the object inside the command-specified goal.", verify_type="final_drop", expected_pose=drop_pose),
                self._homing(dest, obj["id"], "Destination robot homes after placing."),
            ])

        raise RuntimeError(f"Unsupported route type: {route.route_type}")

    def _breakdown_text(self, route: RouteCandidate, obj: dict[str, Any]) -> str:
        if route.route_type == "direct":
            return (
                f"High-level task: move {obj['id']} to the {self.task_spec.final_target_description(obj)}. "
                f"Breakdown: {route.source_robot_id} pick -> grip -> lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        if route.route_type == "direct_after_grip":
            return (
                f"High-level task: finish placing already-held {obj['id']} to {self.task_spec.final_target_description(obj)}. "
                "Breakdown: lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        if route.route_type in {"evacuate_direct", "evacuate_after_grip"}:
            return (
                f"High-level task: evacuate {obj['id']} to a safe pose outside the relevant goal regions. "
                f"Breakdown: {route.source_robot_id} pick/continue -> grip if needed -> lift -> outside approach -> outside drop -> release -> home."
            )
        if route.route_type in {"evacuate_direct", "evacuate_after_grip"}:
            return (
                f"Evacuate {color} object {route.object_id}: robot {route.source_robot_id} "
                "will move it from the goal region to a safe table pose outside all relevant goals."
            )
        if route.route_type == "handover":
            return (
                f"High-level task: move {obj['id']} to {self.task_spec.final_target_description(obj)} through handover. "
                f"Breakdown: {route.source_robot_id} pick -> grip -> lift -> buffer deposit -> release -> home; "
                f"then {route.destination_robot_id} buffer pick -> grip -> lift -> corrected goal approach -> corrected drop -> release -> home."
            )
        return f"High-level task: move {obj['id']}."

    @staticmethod
    def _object_by_id(observation: dict[str, Any], object_id: str) -> dict[str, Any] | None:
        for obj in observation.get("objects", []):
            if obj.get("id") == object_id:
                return obj
        return None

    def _moving(self, robot_id: str, pose: dict[str, Any], object_id: str | None, intent: str, reason: str) -> Decision:
        return Decision(robot_id, ACTION_MOVING, pose, object_id, intent=intent, reason=reason)

    def _grip(self, robot_id: str, object_id: str, reason: str) -> Decision:
        return Decision(robot_id, ACTION_GRIP, None, object_id, intent="grip", reason=reason)

    def _release(
        self,
        robot_id: str,
        object_id: str,
        reason: str,
        verify_type: str | None = None,
        expected_pose: dict[str, Any] | None = None,
    ) -> Decision:
        decision = Decision(robot_id, ACTION_RELEASE, None, object_id, intent="release", reason=reason)
        if verify_type is not None:
            decision.metadata["verify_after_action"] = {
                "type": verify_type,
                "object_id": object_id,
                "expected_pose": expected_pose,
            }
        return decision

    def _homing(self, robot_id: str, object_id: str | None, reason: str) -> Decision:
        return Decision(robot_id, ACTION_HOMING, None, object_id, intent="home", reason=reason)

    def _safe_z(self, observation: dict[str, Any]) -> float:
        table_center = observation["table"]["center"]
        return float(table_center.get("z", DEFAULT_TABLE_CENTER["z"])) + float(self.args.safe_z_offset)

    def _robot_workspace_radius(self, observation: dict[str, Any], robot_id: str) -> float:
        if self.args.workspace_radius is not None:
            return float(self.args.workspace_radius)
        robot = observation["robots"].get(robot_id, {})
        base_pose = robot.get("base_pose")
        if base_pose is None:
            table_size = observation["table"].get("size", DEFAULT_TABLE_SIZE)
            return max(float(table_size.get("x", 0.9)), float(table_size.get("y", 0.7))) * 0.65
        center = observation["table"]["center"]
        return xy_distance(base_pose["position"], center) + float(self.args.center_reach_margin)

    def _robot_can_reach_position(self, observation: dict[str, Any], robot_id: str, position: dict[str, float]) -> bool:
        robot = observation["robots"].get(robot_id, {})
        base_pose = robot.get("base_pose")
        if base_pose is None:
            return False
        return xy_distance(base_pose["position"], position) <= self._robot_workspace_radius(observation, robot_id)

    def _robot_can_reach_pose(self, observation: dict[str, Any], robot_id: str, pose: dict[str, Any]) -> bool:
        return self._robot_can_reach_position(observation, robot_id, pose["position"])

    def _route_distance_score(self, observation: dict[str, Any], robot_id: str, start: dict[str, Any], end: dict[str, Any]) -> float:
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
        if all(self._robot_can_reach_pose(observation, rid, center_pose) for rid in CONTROL_SERVICE_TOPICS):
            return center_pose

        size = table.get("size", DEFAULT_TABLE_SIZE)
        half_x = float(size.get("x", DEFAULT_TABLE_SIZE["x"])) * 0.5 - DEFAULT_BUFFER_CLEARANCE_M
        half_y = float(size.get("y", DEFAULT_TABLE_SIZE["y"])) * 0.5 - DEFAULT_BUFFER_CLEARANCE_M
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
            if all(self._robot_can_reach_pose(observation, rid, pose) for rid in CONTROL_SERVICE_TOPICS):
                return pose
        return None

    def _choose_outside_drop_pose(
        self,
        observation: dict[str, Any],
        robot_id: str,
        moving_obj: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Choose a safe table pose outside all relevant goal regions.

        Used for commands such as "move blue objects out of the goal".  The
        candidate must be on the table, outside every goal marker, away from
        other objects, and reachable by the selected robot.  Z is derived from
        the moving object's marker z, not from the table center.
        """
        table = observation["table"]
        center = table.get("center", DEFAULT_TABLE_CENTER)
        size = table.get("size", DEFAULT_TABLE_SIZE)
        half_x = max(0.0, float(size.get("x", DEFAULT_TABLE_SIZE["x"])) * 0.5 - float(self.args.buffer_clearance))
        half_y = max(0.0, float(size.get("y", DEFAULT_TABLE_SIZE["y"])) * 0.5 - float(self.args.buffer_clearance))
        z = float(moving_obj["pose"]["position"].get("z", DEFAULT_TABLE_CENTER["z"])) + float(self.args.buffer_z_offset)
        goals = observation.get("goals", {})
        occupied = [obj for obj in observation.get("objects", []) if obj.get("id") != moving_obj.get("id")]

        # Prefer points near the object but outside the goal, then generally near the table center.
        samples: list[dict[str, float]] = []
        grid = 9
        for ix in range(grid):
            for iy in range(grid):
                samples.append(
                    {
                        "x": float(center["x"]) - half_x + (2.0 * half_x * ix / max(1, grid - 1)),
                        "y": float(center["y"]) - half_y + (2.0 * half_y * iy / max(1, grid - 1)),
                        "z": z,
                    }
                )
        obj_pos = moving_obj["pose"]["position"]
        samples.sort(key=lambda p: (xy_distance(p, obj_pos), xy_distance(p, center)))
        for point in samples:
            pose = make_pose(point)
            if not self._robot_can_reach_pose(observation, robot_id, pose):
                continue
            if any(point_in_goal(point, goal, margin=-0.005) for goal in goals.values()):
                continue
            if any(xy_distance(point, other["pose"]["position"]) < float(self.args.drop_clearance) for other in occupied):
                continue
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

    def _drop_pose_candidates(self, goal: dict[str, Any], occupied: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
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
                        "x": float(center["x"]) - half_x + (2.0 * half_x * xi / max(1, x_count - 1)),
                        "y": float(center["y"]) - half_y + (2.0 * half_y * yi / max(1, y_count - 1)),
                        "z": corrected_z,
                    }
                )
        points.append({"x": float(center["x"]), "y": float(center["y"]), "z": corrected_z})
        points.sort(key=lambda p: xy_distance(p, center))
        for point in points:
            if all(xy_distance(point, obj["pose"]["position"]) >= step for obj in occupied):
                yield make_pose(point)

# ---------------------------------------------------------------------------
# Executor and validation
# ---------------------------------------------------------------------------


class Executor:
    def __init__(self, node: RosWorldNode, args: argparse.Namespace, reporter: "Reporter") -> None:
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
        if decision.action == ACTION_RELEASE and not self.held_objects.get(str(decision.robot_id)):
            raise ValueError("Release requested while the selected robot is not holding an object.")
        if decision.action == ACTION_MOVING and decision.intent in {
            "lift",
            "handover_approach",
            "handover_place",
            "drop_approach",
            "drop",
            "outside_approach",
            "outside_drop",
        }:
            if not self.held_objects.get(str(decision.robot_id)):
                raise ValueError(f"Moving intent={decision.intent!r} requires the robot to hold an object.")

    def execute(self, decision: Decision, observation: dict[str, Any]) -> ExecutionResult:
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
        success, message = self.node.call_control_service(decision.robot_id, decision.action, decision.target_pose)
        result = ExecutionResult(success, message, decision)

        if success and decision.action == ACTION_MOVING and decision.target_pose is not None:
            arrived = self.node.wait_for_eef_target(
                decision.robot_id,
                decision.target_pose,
                timeout_sec=float(self.args.motion_timeout),
            )
            if not arrived:
                result = ExecutionResult(False, f"{message}; EEF did not reach target before timeout", decision)
        elif success:
            self._settle()

        # For release actions, do not clear the internal held state until the
        # fresh marker/image verification confirms that the object was actually
        # deposited at the expected buffer or goal.  This prevents a false finish
        # while an object is still held over the goal.
        if decision.action == ACTION_RELEASE:
            result = self._verify_after_action(result)
            self._update_held_state(decision, observation, result.success)
        else:
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
        verify_spec = decision.metadata.get("verify_after_action") if decision.metadata else None
        if not result.success or not verify_spec or self.args.dry_run:
            return result

        self._wait_for_fresh_observation(float(self.args.verify_delay))
        try:
            updated_observation, _ = self.node.snapshot()
        except Exception as exc:
            return ExecutionResult(False, f"{result.message}; post-action verification snapshot failed: {exc}", decision)

        ok, detail = self._check_verification_spec(updated_observation, verify_spec)
        if ok:
            return ExecutionResult(True, f"{result.message}; verified: {detail}", decision)
        return ExecutionResult(False, f"{result.message}; verification failed: {detail}", decision)

    def _wait_for_fresh_observation(self, delay_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, delay_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _check_verification_spec(self, observation: dict[str, Any], verify_spec: dict[str, Any]) -> tuple[bool, str]:
        object_id = str(verify_spec.get("object_id") or "")
        obj = self._object_by_id(observation, object_id)
        if obj is None:
            return False, f"object {object_id!r} is not visible in the updated marker state"

        verification_type = str(verify_spec.get("type") or "")
        expected_pose = verify_spec.get("expected_pose") or {}
        expected_position = expected_pose.get("position") if isinstance(expected_pose, dict) else None
        actual_position = obj["pose"]["position"]

        if verification_type == "buffer_deposit":
            if not isinstance(expected_position, dict):
                return False, "buffer verification lacks expected_pose.position"
            xy_err = xy_distance(actual_position, expected_position)
            z_err = abs(float(actual_position["z"]) - float(expected_position["z"]))
            if xy_err <= float(self.args.verify_xy_tolerance) and z_err <= float(self.args.verify_z_tolerance):
                return True, (
                    f"object {object_id} is at the shared buffer "
                    f"within xy_err={xy_err:.4f} m, z_err={z_err:.4f} m"
                )
            return False, (
                f"object {object_id} was expected at buffer {expected_position}, "
                f"but marker is at {actual_position}; xy_err={xy_err:.4f} m, z_err={z_err:.4f} m"
            )

        if verification_type == "final_drop":
            expected_goal_color = str(verify_spec.get("expected_goal_color") or obj.get("color"))
            goal = observation.get("goals", {}).get(expected_goal_color)
            inside_command_goal = goal is not None and point_in_goal(actual_position, goal, margin=0.0)
            if inside_command_goal:
                return True, f"object {object_id} is inside the command-specified {expected_goal_color} goal"
            return False, (
                f"object {object_id} is not in the command-specified {expected_goal_color} goal after release; "
                f"inside_goal={obj.get('inside_goal')}, default_sorted={obj.get('sorted')}, position={actual_position}"
            )

        if verification_type == "outside_drop":
            expected_pose = verify_spec.get("expected_pose") or {}
            expected_position = expected_pose.get("position") if isinstance(expected_pose, dict) else None
            if any(point_in_goal(actual_position, goal, margin=0.0) for goal in observation.get("goals", {}).values()):
                return False, (
                    f"object {object_id} is still inside a goal after evacuation release; "
                    f"inside_goal={obj.get('inside_goal')}, position={actual_position}"
                )
            if isinstance(expected_position, dict):
                xy_err = xy_distance(actual_position, expected_position)
                z_err = abs(float(actual_position["z"]) - float(expected_position["z"]))
                if xy_err > float(self.args.verify_xy_tolerance) or z_err > float(self.args.verify_z_tolerance):
                    return False, (
                        f"object {object_id} is outside goals but not near expected outside pose; "
                        f"expected={expected_position}, actual={actual_position}, xy_err={xy_err:.4f}, z_err={z_err:.4f}"
                    )
            return True, f"object {object_id} is outside all goal regions after evacuation"

        return False, f"unknown verification type: {verification_type!r}"

    @staticmethod
    def _object_by_id(observation: dict[str, Any], object_id: str) -> dict[str, Any] | None:
        for obj in observation.get("objects", []):
            if obj.get("id") == object_id:
                return obj
        return None

    def _settle(self) -> None:
        deadline = time.monotonic() + float(self.args.settle_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _update_held_state(self, decision: Decision, observation: dict[str, Any], success: bool) -> None:
        if not success or decision.robot_id not in self.held_objects:
            return
        robot_id = str(decision.robot_id)
        if decision.action == ACTION_MOVING and decision.intent in {"pick", "handover_pick"}:
            self.pending_grip_targets[robot_id] = decision.target_object_id
        elif decision.action == ACTION_GRIP:
            self.held_objects[robot_id] = decision.target_object_id or self.pending_grip_targets.get(robot_id)
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
            rows["phase"] = f"{decision.metadata.get('primitive_index')}/{decision.metadata.get('primitive_count')}"
        if self.verbose:
            if decision.target_pose is not None:
                rows["target_pose"] = json.dumps(decision.target_pose, ensure_ascii=False)
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
            reporter.status("Waiting for ROS2 observations", {"missing": ", ".join(waiting)})
            last_report = now


def save_frame(directory: Path, step: int, image_bytes: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"step_{step:03d}.png").write_bytes(image_bytes)


def spin_sleep(node: RosWorldNode, delay_sec: float) -> None:
    deadline = time.monotonic() + max(0.0, delay_sec)
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def verify_final_completion(
    node: RosWorldNode,
    args: argparse.Namespace,
    task_spec: TaskSpec,
    executor: Executor,
) -> tuple[bool, str]:
    """Require fresh vision/marker checks before accepting finish.

    This guards against the failure mode where the object marker is geometrically
    inside the goal while the gripper is still holding it over the goal.  The
    check requires no internally held objects and repeated fresh observations in
    which every active object is inside its command-specified destination goal.
    """
    checks = max(1, int(args.finish_verify_checks))
    detail = ""
    for index in range(checks):
        spin_sleep(node, float(args.verify_delay if index == 0 else args.finish_verify_interval))
        observation, _ = node.snapshot()
        ok, detail = task_complete_by_vision(
            observation,
            task_spec,
            held_objects=executor.held_objects,
            pending_grip_targets=executor.pending_grip_targets,
        )
        if not ok:
            return False, f"finish check {index + 1}/{checks} failed: {detail}"
    return True, f"finish verified by {checks} fresh marker/image check(s): {detail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive dual-Franka CLI with command-aware handover planning."
    )
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Optional non-interactive high-level task command. "
            "If omitted, the CLI opens an interactive command input prompt."
        ),
    )
    parser.add_argument(
        "--input",
        dest="input_command",
        default=None,
        help="Backward-compatible alias for --command. Prefer the interactive prompt or --command.",
    )
    parser.add_argument(
        "--use-default-command",
        action="store_true",
        help="Skip the interactive prompt and use the built-in default red-to-red / blue-to-blue command.",
    )
    parser.add_argument("--max-steps", type=int, default=160, help="Maximum primitive-action steps.")
    parser.add_argument(
        "--semantic-mode",
        choices=("auto", "openai", "ollama", "rule"),
        default="auto",
        help="Semantic interpreter backend. auto tries OpenAI, then Ollama, then conservative rule fallback.",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help="Model name for semantic interpretation. Defaults to OPENAI_MODEL or OLLAMA_MODEL.",
    )
    parser.add_argument(
        "--semantic-temperature",
        type=float,
        default=0.0,
        help="LLM temperature for semantic command interpretation.",
    )
    parser.add_argument(
        "--no-semantic-image",
        action="store_true",
        help="Do not send the top-view image to the semantic LLM; use serialized scene state only.",
    )
    parser.add_argument("--once", action="store_true", help="Execute only one primitive action.")
    parser.add_argument("--dry-run", action="store_true", help="Skip ROS2 service calls and only update internal state.")
    parser.add_argument("--motion-timeout", type=float, default=20.0, help="Seconds to wait for Moving convergence.")
    parser.add_argument("--settle-sec", type=float, default=0.7, help="Seconds to wait after grip/release/home.")
    parser.add_argument("--save-frames", type=Path, default=None, help="Directory to save top-view frames.")
    parser.add_argument("--plain", action="store_true", help="Disable rich terminal UI.")
    parser.add_argument("--verbose", action="store_true", help="Print target poses and extra debugging details.")

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
    parser.add_argument("--lift-delta", type=float, default=DEFAULT_LIFT_DELTA_M, help="Vertical lift added after grasp.")
    parser.add_argument("--safe-z-offset", type=float, default=DEFAULT_SAFE_Z_OFFSET_M, help="Safe Z above table center.")
    parser.add_argument("--goal-margin", type=float, default=DEFAULT_GOAL_MARGIN_M, help="Inset margin inside goal region.")
    parser.add_argument("--drop-clearance", type=float, default=DEFAULT_DROP_CLEARANCE_M, help="Minimum XY spacing between dropped objects.")
    parser.add_argument("--buffer-clearance", type=float, default=DEFAULT_BUFFER_CLEARANCE_M, help="Inset margin used when sampling table-center/outside-goal buffer poses.")
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
    parser.add_argument(
        "--finish-verify-checks",
        type=int,
        default=2,
        help="Number of fresh marker/image checks required before final task completion is accepted.",
    )
    parser.add_argument(
        "--finish-verify-interval",
        type=float,
        default=0.4,
        help="Interval between final completion verification checks.",
    )
    return parser.parse_args()



def resolve_task_command_text(args: argparse.Namespace, reporter: Reporter) -> str | None:
    """Get the raw natural-language task command from CLI or interactive input.

    Semantic interpretation is intentionally delayed until after ROS observations
    are available, so the LLM can see the actual scene and choose an MCP-like
    semantic tool with grounded colors/goals/objects.
    """
    provided = (args.input_command or args.command or "").strip()
    if args.use_default_command and not provided:
        provided = DEFAULT_COMMAND
    if provided:
        return provided

    if not sys.stdin.isatty():
        reporter.error(
            "No task command was provided and stdin is not interactive. "
            "Run with --command \"...\" or --use-default-command."
        )
        return None

    reporter.info("\nEnter the task command for this run.")
    reporter.info("Examples:")
    reporter.info("  1) 빨간 물체는 빨간 목표점에, 파란 물체는 파란 목표점에 놓아라")
    reporter.info("  2) 파란 물체를 전부 빨간 목표점에 넣어라")
    reporter.info("  3) 파란 물체를 목표점에서 전부 밖으로 빼내라")
    reporter.info("  4) move all blue objects out of the goals")
    reporter.info("Type 'q' or 'quit' to exit. Press Enter to use the default command.\n")

    while True:
        try:
            typed = input("Task command> ").strip()
        except (EOFError, KeyboardInterrupt):
            reporter.error("Task command input cancelled.")
            return None
        if typed.lower() in {"q", "quit", "exit"}:
            reporter.error("Task command input cancelled.")
            return None
        if not typed:
            typed = DEFAULT_COMMAND
        return typed


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
    raw_command = resolve_task_command_text(args, reporter)
    if raw_command is None:
        return 1
    args.command = raw_command

    reporter.status(
        "Startup",
        {
            "planner": "llm_semantic_mcp_interpreter_plus_hierarchical_handover_decomposer",
            "semantic_mode": args.semantic_mode,
            "semantic_model": args.semantic_model or "default",
            "dry_run": args.dry_run,
            "command": args.command,
            "workspace_radius": args.workspace_radius,
            "center_reach_margin": args.center_reach_margin,
            "goal_z_offset": args.goal_z_offset,
            "buffer_z_offset": args.buffer_z_offset,
            "verify_delay": args.verify_delay,
            "finish_verify_checks": args.finish_verify_checks,
        },
    )

    rclpy.init()
    node = RosWorldNode()
    executor = Executor(node, args, reporter)
    planner: HandoverTaskPlanner | None = None
    task_spec: TaskSpec | None = None

    try:
        wait_for_ready(node, reporter)
        initial_observation, initial_image_bytes = node.snapshot()
        try:
            semantic_interpreter = SemanticTaskInterpreter(args, reporter)
            task_spec = semantic_interpreter.interpret(args.command, initial_observation, initial_image_bytes)
            validate_task_spec_against_scene(task_spec, initial_observation)
        except ValueError as exc:
            reporter.error(f"Rejected task command for this scene: {exc}")
            return 1
        except Exception as exc:
            reporter.error(f"Semantic interpretation failed: {exc}")
            return 1

        reporter.status(
            "Accepted semantic task",
            {
                "command": args.command,
                "task_type": task_spec.task_type,
                "task_spec": task_spec.as_dict(),
            },
        )
        planner = HandoverTaskPlanner(args, task_spec)

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
                    "task_pending": [
                        obj["id"] for obj in observation.get("objects", [])
                        if task_spec.is_relevant_object(obj)
                        and not object_satisfies_task(obj, observation.get("goals", {}), task_spec)
                    ],
                    "held": executor.held_objects,
                    "active_object": planner.active_object_id,
                    "active_route": planner.active_route_type,
                    "high_level_order": planner.task_order,
                },
            )

            try:
                decision = planner.next_decision(
                    observation,
                    executor.held_objects,
                    pending_grip_targets=executor.pending_grip_targets,
                )
                reporter.decision(step, decision, queue_len=len(planner.queue))
                result = executor.execute(decision, observation)
            except Exception as exc:
                planner.clear_active_plan()
                reporter.error(f"Planning/execution error: {exc}")
                return 1

            reporter.status("Execution result", {"success": result.success, "message": result.message})
            if decision.done:
                ok, detail = verify_final_completion(node, args, task_spec, executor)
                if ok:
                    reporter.info(detail)
                    return 0
                planner.clear_active_plan()
                reporter.error(f"Finish was rejected by fresh vision/marker verification: {detail}")
                if args.once:
                    return 1
                continue
            if not result.success:
                planner.clear_active_plan()
                reporter.error("Cleared the active plan because the previous primitive failed.")
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
