#!/usr/bin/env python3
"""Deterministic single-file CLI for the dual-Franka Isaac Sim world.

Design intent:
- LLM is used only to interpret non-standard human commands.
- Planning, action compilation, execution, verification, and recovery are deterministic.
- No external MD files are loaded. All agent/prompt text lives in this file.

Run after sourcing the ROS2 workspace:
    source install/setup.bash
    python3 cli_simple_robot_planner.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import rclpy
    from custom_msgs.srv import ControlCommand
    from geometry_msgs.msg import Pose
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import MarkerArray
except Exception as exc:  # allow py_compile without a sourced ROS env
    ROS_IMPORT_ERROR = exc
    rclpy = None  # type: ignore[assignment]
    ControlCommand = None  # type: ignore[assignment]
    JointState = None  # type: ignore[assignment]
    MarkerArray = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass

else:
    ROS_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Embedded instruction text. This replaces separate CLAUDE.md / MEMORY.md files.
# ---------------------------------------------------------------------------

PLANNER_AGENT_PROMPT = """
You are a command interpreter for a dual-Franka table-top manipulation scene.
Return JSON only.

Input contains:
- user_command
- objects: id, color, position
- goals: available goal colors

Your job is only to map objects to destination goal colors.
Do not create primitive actions. Do not choose robot ids. Do not create handover steps.
The deterministic Python planner will handle reachability, handover, action compilation,
and verification.

Output schema:
{
  "assignments": [
    {"object_id": "cube_1:3", "goal_color": "red"}
  ],
  "reason": "short explanation"
}

Rules:
- If the user asks all/every object to one goal, map every visible object to that goal.
- If the user asks color matching, map each object to the goal matching its intrinsic color.
- If the user selects only red/blue objects, include only those objects.
- Use exact object_id values from input.
- goal_color must be one of the provided goals.
"""

RUNTIME_POLICY = """
Runtime policy:
1. The high-level object route is fixed before execution.
2. Pose coordinates are not cached in the plan. Moving/Placing/Centering poses are resolved
   from the newest snapshot immediately before each primitive is sent.
3. Primitive service success is only command acceptance. Stage success is decided by strict
   snapshot verification after the stage.
4. A stage failure never advances to the next object or next stage. Retry the same stage;
   if recovery is exhausted, stop and report failure.
5. Handover is explicit: source robot picks and drops at table_center_handover, destination
   robot then picks from the current object pose and drops at the requested goal.
"""


# ---------------------------------------------------------------------------
# Constants matching world.py/control_command_gui.py
# ---------------------------------------------------------------------------

ACTION_MOVING = "Moving"
ACTION_CENTERING = "Centering"
ACTION_PLACING = "Placing"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Release"
ACTION_HOMING = "Homing"

MARKER_TOPIC = "/world/object_markers"
JOINT_STATE_TOPICS = {
    "left": "/franka_left/joint_states",
    "right": "/franka_right/joint_states",
}
EEF_POSE_TOPICS = {
    "left": "/franka_left/end_effector_pose",
    "right": "/franka_right/end_effector_pose",
}
ROBOT_POSE_TOPICS = {
    "left": "/franka_left/pose",
    "right": "/franka_right/pose",
}
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}

VERTICAL_EEF_XYZW = (1.0, 0.0, 0.0, 0.0)
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

WORKSPACE_RADIUS = 0.585
WORKSPACE_BORDERLINE_MARGIN = 0.06
GRIPPER_OPEN_MIN = 0.033
GRIPPER_OPEN_VEL_TOL = 0.01
GRIPPER_EMPTY_CLOSED_MAX = 0.012
ARM_MOTION_VEL_TOL = 0.05

PICK_VERIFY_TIMEOUT = 5.0
RELEASE_VERIFY_TIMEOUT = 4.0
DROP_VERIFY_TIMEOUT = 5.0
PRE_GRIP_SETTLE_TIMEOUT = 8.0
SERVICE_WAIT_TIMEOUT = 1.0
SERVICE_CALL_TIMEOUT = 25.0
MAX_STAGE_RETRIES = 3

GOAL_XY_TOL_MARGIN = 0.015
HANDOVER_XY_TOL = 0.075
SUPPORT_Z_MIN = 0.39
SUPPORT_Z_MAX = 0.50
EEF_DETACH_XY_MIN = 0.06
EEF_DETACH_Z_MIN = 0.055
HELD_XY_TOL = 0.055
HELD_Z_TOL = 0.13

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Vec3 = dict[str, float]
RobotId = Literal["left", "right"]
StageKind = Literal["PICK", "DROP_HANDOVER", "DROP_GOAL"]


@dataclass
class ObjectInfo:
    object_id: str
    ns: str
    marker_id: int
    color: str
    position: Vec3
    scale: Vec3


@dataclass
class GoalInfo:
    color: str
    object_id: str
    position: Vec3
    scale: Vec3


@dataclass
class RobotState:
    robot_id: RobotId
    base_position: Vec3
    eef_position: Vec3 | None = None
    joint_names: list[str] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)


@dataclass
class Snapshot:
    objects: dict[str, ObjectInfo]
    goals: dict[str, GoalInfo]
    robots: dict[RobotId, RobotState]
    timestamp: float


@dataclass
class Assignment:
    object_id: str
    goal_color: str


@dataclass
class Stage:
    kind: StageKind
    robot_id: RobotId
    object_id: str
    goal_color: str | None = None


@dataclass
class ObjectPlan:
    object_id: str
    goal_color: str
    stages: list[Stage]


@dataclass
class CommandPlan:
    assignments: list[Assignment]
    object_plans: list[ObjectPlan]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def distance_xy(a: Vec3, b: Vec3) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def distance_xyz(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
    )


def color_name_from_rgba(r: float, g: float, b: float) -> str:
    if r >= b and r >= g:
        return "red"
    if b >= r and b >= g:
        return "blue"
    return "unknown"


def pose_dict(position: Vec3) -> dict[str, Any]:
    return {
        "position": {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        },
        "orientation": {
            "x": VERTICAL_EEF_XYZW[0],
            "y": VERTICAL_EEF_XYZW[1],
            "z": VERTICAL_EEF_XYZW[2],
            "w": VERTICAL_EEF_XYZW[3],
        },
    }


def ros_pose_from_dict(data: dict[str, Any] | None) -> Any:
    pose = Pose()
    if data is not None:
        pos = data.get("position", {})
        ori = data.get("orientation", {})
        pose.position.x = float(pos.get("x", 0.0))
        pose.position.y = float(pos.get("y", 0.0))
        pose.position.z = float(pos.get("z", 0.0))
        pose.orientation.x = float(ori.get("x", VERTICAL_EEF_XYZW[0]))
        pose.orientation.y = float(ori.get("y", VERTICAL_EEF_XYZW[1]))
        pose.orientation.z = float(ori.get("z", VERTICAL_EEF_XYZW[2]))
        pose.orientation.w = float(ori.get("w", VERTICAL_EEF_XYZW[3]))
    else:
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = VERTICAL_EEF_XYZW
    return pose


def offset_slot(base: Vec3, index: int) -> Vec3:
    dx, dy = DROP_SLOT_OFFSETS[index % len(DROP_SLOT_OFFSETS)]
    return {
        "x": base["x"] + dx * DROP_SLOT_STEP_M,
        "y": base["y"] + dy * DROP_SLOT_STEP_M,
        "z": base["z"],
    }


def goal_drop_position(goal_color: str, object_index: int) -> Vec3:
    base = CONTROL_GOAL_TARGETS[goal_color]
    return offset_slot(base, object_index)


def handover_drop_position(object_index: int) -> Vec3:
    return offset_slot(CONTROL_TABLE_CENTER_TARGET, object_index)


def robot_base_default(robot_id: RobotId) -> Vec3:
    return {
        "left": {"x": 0.0, "y": 0.0, "z": 0.0},
        "right": {"x": 1.2, "y": 0.0, "z": 0.0},
    }[robot_id]


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


class SceneNode(Node):
    def __init__(self) -> None:
        super().__init__("simple_robot_planner_cli")
        self.lock = threading.Lock()
        self.latest_markers: MarkerArray | None = None
        self.robot_states: dict[RobotId, RobotState] = {
            "left": RobotState("left", robot_base_default("left")),
            "right": RobotState("right", robot_base_default("right")),
        }
        self.create_subscription(MarkerArray, MARKER_TOPIC, self._markers_cb, 10)
        for rid, topic in JOINT_STATE_TOPICS.items():
            self.create_subscription(
                JointState, topic, lambda msg, rid=rid: self._joint_cb(rid, msg), 10
            )
        for rid, topic in EEF_POSE_TOPICS.items():
            self.create_subscription(
                (
                    type("PoseStampedAlias", (), {})
                    if False
                    else __import__("geometry_msgs.msg").msg.PoseStamped
                ),
                topic,
                lambda msg, rid=rid: self._eef_cb(rid, msg),
                10,
            )
        for rid, topic in ROBOT_POSE_TOPICS.items():
            self.create_subscription(
                __import__("geometry_msgs.msg").msg.PoseStamped,
                topic,
                lambda msg, rid=rid: self._robot_pose_cb(rid, msg),
                10,
            )
        self.cclients = {
            rid: self.create_client(ControlCommand, topic)
            for rid, topic in CONTROL_SERVICE_TOPICS.items()
        }

    def _markers_cb(self, msg: Any) -> None:
        with self.lock:
            self.latest_markers = msg

    def _joint_cb(self, rid: RobotId, msg: Any) -> None:
        with self.lock:
            st = self.robot_states[rid]
            st.joint_names = list(msg.name)
            st.joint_positions = [float(v) for v in msg.position]
            st.joint_velocities = [float(v) for v in msg.velocity]

    def _eef_cb(self, rid: RobotId, msg: Any) -> None:
        p = msg.pose.position
        with self.lock:
            self.robot_states[rid].eef_position = {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            }

    def _robot_pose_cb(self, rid: RobotId, msg: Any) -> None:
        p = msg.pose.position
        with self.lock:
            self.robot_states[rid].base_position = {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            }

    def snapshot(self) -> Snapshot:
        with self.lock:
            markers = self.latest_markers
            robots = {
                rid: RobotState(
                    rid,
                    dict(st.base_position),
                    None if st.eef_position is None else dict(st.eef_position),
                    list(st.joint_names),
                    list(st.joint_positions),
                    list(st.joint_velocities),
                )
                for rid, st in self.robot_states.items()
            }
        if markers is None:
            raise RuntimeError("No MarkerArray has been received yet.")

        objects: dict[str, ObjectInfo] = {}
        goals: dict[str, GoalInfo] = {}
        for marker in markers.markers:
            ns = str(marker.ns)
            mid = int(marker.id)
            pos = {
                "x": float(marker.pose.position.x),
                "y": float(marker.pose.position.y),
                "z": float(marker.pose.position.z),
            }
            scale = {
                "x": float(marker.scale.x),
                "y": float(marker.scale.y),
                "z": float(marker.scale.z),
            }
            if ns == "table":
                continue
            if ns in {"red_goal", "blue_goal"}:
                color = "red" if ns.startswith("red") else "blue"
                goals[color] = GoalInfo(color, f"{ns}:{mid}", pos, scale)
                continue
            color = color_name_from_rgba(
                float(marker.color.r), float(marker.color.g), float(marker.color.b)
            )
            object_id = f"{ns}:{mid}"
            objects[object_id] = ObjectInfo(object_id, ns, mid, color, pos, scale)
        return Snapshot(objects, goals, robots, time.time())

    def wait_for_observations(self, timeout: float = 15.0) -> Snapshot:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                snap = self.snapshot()
                if (
                    snap.objects
                    and snap.goals
                    and all(snap.robots[r].joint_names for r in ("left", "right"))
                ):
                    return snap
            except Exception:
                pass
            time.sleep(0.1)
        return self.snapshot()

    def call_control(
        self, robot_id: RobotId, action: str, target_pose: dict[str, Any] | None
    ) -> tuple[bool, str]:
        client = self.cclients[robot_id]
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT):
            return False, f"service unavailable: {CONTROL_SERVICE_TOPICS[robot_id]}"
        req = ControlCommand.Request()
        req.action = action
        req.target_pose = ros_pose_from_dict(target_pose)
        future = client.call_async(req)
        start = time.time()
        while time.time() - start < SERVICE_CALL_TIMEOUT:
            if future.done():
                try:
                    resp = future.result()
                except Exception as exc:
                    return False, f"service exception: {exc}"
                return bool(resp.success), str(resp.message)
            time.sleep(0.03)
        return False, f"service timeout after {SERVICE_CALL_TIMEOUT}s"


# ---------------------------------------------------------------------------
# Interpreter and planner
# ---------------------------------------------------------------------------


def standard_interpret(command: str, snap: Snapshot) -> list[Assignment] | None:
    text = command.lower().strip()
    has_all = any(k in text for k in ["모든", "전체", "전부", "all", "every"])
    mentions_red_goal = any(
        k in text for k in ["빨간 목표", "빨간 목표점", "red goal", "red_goal"]
    )
    mentions_blue_goal = any(
        k in text for k in ["파란 목표", "파란 목표점", "blue goal", "blue_goal"]
    )
    mentions_red_obj = any(
        k in text for k in ["빨간 물체", "빨간색 물체", "red object", "red objects"]
    )
    mentions_blue_obj = any(
        k in text for k in ["파란 물체", "파란색 물체", "blue object", "blue objects"]
    )
    color_sort = any(
        k in text
        for k in ["색상별", "색깔별", "자기 색", "같은 색", "matching", "sort"]
    )
    explicit_both = (
        mentions_red_obj
        and mentions_blue_obj
        and mentions_red_goal
        and mentions_blue_goal
    )

    objects = list(snap.objects.values())
    if not objects:
        return []

    if has_all and mentions_red_goal and not mentions_blue_goal:
        return [Assignment(obj.object_id, "red") for obj in objects]
    if has_all and mentions_blue_goal and not mentions_red_goal:
        return [Assignment(obj.object_id, "blue") for obj in objects]
    if color_sort or explicit_both:
        return [
            Assignment(obj.object_id, obj.color)
            for obj in objects
            if obj.color in snap.goals
        ]
    if mentions_red_obj and mentions_red_goal and not mentions_blue_obj:
        return [
            Assignment(obj.object_id, "red") for obj in objects if obj.color == "red"
        ]
    if mentions_blue_obj and mentions_blue_goal and not mentions_red_obj:
        return [
            Assignment(obj.object_id, "blue") for obj in objects if obj.color == "blue"
        ]
    if mentions_red_obj and mentions_blue_goal and not mentions_blue_obj:
        return [
            Assignment(obj.object_id, "blue") for obj in objects if obj.color == "red"
        ]
    if mentions_blue_obj and mentions_red_goal and not mentions_red_obj:
        return [
            Assignment(obj.object_id, "red") for obj in objects if obj.color == "blue"
        ]
    return None


def llm_interpret(command: str, snap: Snapshot) -> list[Assignment]:
    if not OPENAI_API_KEY:
        raise RuntimeError("Unknown command and OPENAI_API_KEY is not set.")
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": PLANNER_AGENT_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_command": command,
                        "objects": [
                            {
                                "id": o.object_id,
                                "color": o.color,
                                "position": o.position,
                            }
                            for o in snap.objects.values()
                        ],
                        "goals": sorted(snap.goals.keys()),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    req = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    out: list[Assignment] = []
    for item in parsed.get("assignments", []):
        oid = str(item.get("object_id"))
        goal = str(item.get("goal_color"))
        if oid not in snap.objects:
            raise ValueError(f"LLM returned unknown object_id: {oid}")
        if goal not in snap.goals:
            raise ValueError(f"LLM returned unknown goal_color: {goal}")
        out.append(Assignment(oid, goal))
    return out


def can_reach(robot: RobotState, point: Vec3, include_borderline: bool = True) -> bool:
    radius = WORKSPACE_RADIUS + (
        WORKSPACE_BORDERLINE_MARGIN if include_borderline else 0.0
    )
    return distance_xy(robot.base_position, point) <= radius


def nearest_robot_to(
    snap: Snapshot, point: Vec3, candidates: list[RobotId] | None = None
) -> RobotId:
    ids: list[RobotId] = candidates if candidates else ["left", "right"]
    return min(ids, key=lambda rid: distance_xy(snap.robots[rid].base_position, point))


def object_index(object_id: str) -> int:
    m = re.search(r":(\d+)$", object_id)
    return int(m.group(1)) if m else 0


def goal_region_contains(snap: Snapshot, obj: ObjectInfo, goal_color: str) -> bool:
    goal = snap.goals.get(goal_color)
    if goal is None:
        return False
    half_x = max(goal.scale["x"] / 2.0, 0.10) + GOAL_XY_TOL_MARGIN
    half_y = max(goal.scale["y"] / 2.0, 0.10) + GOAL_XY_TOL_MARGIN
    dx = abs(obj.position["x"] - goal.position["x"])
    dy = abs(obj.position["y"] - goal.position["y"])
    return (
        dx <= half_x
        and dy <= half_y
        and SUPPORT_Z_MIN <= obj.position["z"] <= SUPPORT_Z_MAX
    )


def build_plan(command: str, snap: Snapshot) -> CommandPlan:
    assignments = standard_interpret(command, snap)
    source = "standard parser"
    if assignments is None:
        assignments = llm_interpret(command, snap)
        source = f"LLM {OPENAI_MODEL}"
    assignments = [
        a
        for a in assignments
        if a.object_id in snap.objects and a.goal_color in snap.goals
    ]
    object_plans: list[ObjectPlan] = []

    log(f"\nINTERPRETATION ({source})")
    for a in assignments:
        log(f"  - {a.object_id} -> {a.goal_color}_goal")

    for a in assignments:
        obj = snap.objects[a.object_id]
        if goal_region_contains(snap, obj, a.goal_color):
            log(f"SKIP already satisfied: {a.object_id} in {a.goal_color}_goal")
            continue
        goal_pos = goal_drop_position(a.goal_color, object_index(a.object_id))
        pickup_candidates = [
            rid
            for rid in ("left", "right")
            if can_reach(snap.robots[rid], obj.position)
        ]
        dest_candidates = [
            rid for rid in ("left", "right") if can_reach(snap.robots[rid], goal_pos)
        ]
        direct = [rid for rid in pickup_candidates if rid in dest_candidates]
        stages: list[Stage]
        if direct:
            rid = nearest_robot_to(snap, obj.position, direct)
            stages = [
                Stage("PICK", rid, a.object_id),
                Stage("DROP_GOAL", rid, a.object_id, a.goal_color),
            ]
        else:
            pickup_robot = nearest_robot_to(
                snap, obj.position, pickup_candidates or None
            )
            goal_robot = nearest_robot_to(snap, goal_pos, dest_candidates or None)
            stages = [
                Stage("PICK", pickup_robot, a.object_id),
                Stage("DROP_HANDOVER", pickup_robot, a.object_id),
                Stage("PICK", goal_robot, a.object_id),
                Stage("DROP_GOAL", goal_robot, a.object_id, a.goal_color),
            ]
        object_plans.append(ObjectPlan(a.object_id, a.goal_color, stages))
    return CommandPlan(assignments, object_plans)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def finger_values(robot: RobotState) -> tuple[list[float], list[float]]:
    pos: list[float] = []
    vel: list[float] = []
    for name in ["panda_finger_joint1", "panda_finger_joint2"]:
        if name in robot.joint_names:
            i = robot.joint_names.index(name)
            if i < len(robot.joint_positions):
                pos.append(robot.joint_positions[i])
            if i < len(robot.joint_velocities):
                vel.append(robot.joint_velocities[i])
    return pos, vel


def gripper_open(robot: RobotState) -> bool:
    pos, vel = finger_values(robot)
    if len(pos) < 2:
        return False
    mean_pos = sum(pos) / len(pos)
    max_vel = max((abs(v) for v in vel), default=0.0)
    return mean_pos >= GRIPPER_OPEN_MIN and max_vel <= GRIPPER_OPEN_VEL_TOL


def arm_moving(robot: RobotState) -> bool:
    vals = []
    for name, vel in zip(robot.joint_names, robot.joint_velocities):
        if "finger" not in name:
            vals.append(abs(float(vel)))
    return max(vals, default=0.0) > ARM_MOTION_VEL_TOL


def object_near_eef(
    snap: Snapshot,
    robot_id: RobotId,
    object_id: str,
    xy_tol: float = HELD_XY_TOL,
    z_tol: float = HELD_Z_TOL,
) -> bool:
    obj = snap.objects.get(object_id)
    eef = snap.robots[robot_id].eef_position
    if obj is None or eef is None:
        return False
    return (
        distance_xy(obj.position, eef) <= xy_tol
        and abs(obj.position["z"] - eef["z"]) <= z_tol
    )


def likely_held(snap: Snapshot, robot_id: RobotId, object_id: str) -> bool:
    robot = snap.robots[robot_id]
    if gripper_open(robot):
        return False
    return object_near_eef(snap, robot_id, object_id)


def detached_from_eef(snap: Snapshot, robot_id: RobotId, object_id: str) -> bool:
    obj = snap.objects.get(object_id)
    eef = snap.robots[robot_id].eef_position
    if obj is None or eef is None:
        return True
    return (
        distance_xy(obj.position, eef) >= EEF_DETACH_XY_MIN
        or abs(obj.position["z"] - eef["z"]) >= EEF_DETACH_Z_MIN
    )


def object_at_handover(snap: Snapshot, object_id: str) -> bool:
    obj = snap.objects.get(object_id)
    if obj is None:
        return False
    return (
        distance_xy(obj.position, CONTROL_TABLE_CENTER_TARGET) <= HANDOVER_XY_TOL
        and SUPPORT_Z_MIN <= obj.position["z"] <= SUPPORT_Z_MAX
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    def __init__(self, node: SceneNode) -> None:
        self.node = node

    def send(
        self, robot_id: RobotId, action: str, pose: dict[str, Any] | None = None
    ) -> bool:
        log(
            f"  ACTION {robot_id:5s} {action:9s} {json.dumps(pose['position'], ensure_ascii=False) if pose else ''}"
        )
        ok, msg = self.node.call_control(robot_id, action, pose)
        log(f"    -> {ok}: {msg}")
        return ok

    def wait_until(
        self, description: str, predicate, timeout: float, period: float = 0.15
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.node.snapshot()
            if predicate(snap):
                return True
            time.sleep(period)
        snap = self.node.snapshot()
        return bool(predicate(snap))

    def ensure_gripper_open(self, robot_id: RobotId) -> bool:
        snap = self.node.snapshot()
        if gripper_open(snap.robots[robot_id]):
            return True
        if not self.send(robot_id, ACTION_RELEASE):
            return False
        return self.wait_until(
            f"{robot_id} gripper open",
            lambda s: gripper_open(s.robots[robot_id]),
            RELEASE_VERIFY_TIMEOUT,
        )

    def wait_pre_grip_ready(self, robot_id: RobotId, object_id: str) -> bool:
        return self.wait_until(
            "pre-grip ready",
            lambda s: (not arm_moving(s.robots[robot_id]))
            and object_near_eef(s, robot_id, object_id, xy_tol=0.055, z_tol=0.16),
            PRE_GRIP_SETTLE_TIMEOUT,
        )

    def execute_pick(self, robot_id: RobotId, object_id: str) -> bool:
        for attempt in range(1, MAX_STAGE_RETRIES + 1):
            log(f"\nSTAGE PICK {object_id} with {robot_id} attempt {attempt}")
            if not self.ensure_gripper_open(robot_id):
                log("  FAIL gripper did not open before pick")
                continue
            snap = self.node.snapshot()
            obj = snap.objects.get(object_id)
            if obj is None:
                log(f"  FAIL object disappeared: {object_id}")
                return False
            if not can_reach(snap.robots[robot_id], obj.position):
                log(
                    f"  FAIL {robot_id} cannot reach current object pose {obj.position}"
                )
                return False
            if not self.send(robot_id, ACTION_MOVING, pose_dict(obj.position)):
                continue
            if not self.wait_pre_grip_ready(robot_id, object_id):
                log("  WARN pre-grip readiness not confirmed; trying Grip once")
            if not self.send(robot_id, ACTION_GRIP):
                continue
            time.sleep(1.0)
            if self.wait_until(
                "held",
                lambda s: likely_held(s, robot_id, object_id),
                PICK_VERIFY_TIMEOUT,
            ):
                log(f"  OK held {object_id} with {robot_id}")
                return True
            log("  FAIL pick verifier: object not held")
        return False

    def execute_drop_handover(self, robot_id: RobotId, object_id: str) -> bool:
        for attempt in range(1, MAX_STAGE_RETRIES + 1):
            log(f"\nSTAGE DROP_HANDOVER {object_id} with {robot_id} attempt {attempt}")
            snap = self.node.snapshot()
            if not likely_held(snap, robot_id, object_id):
                log("  FAIL object is not held; must pick before drop")
                return False
            target = handover_drop_position(object_index(object_id))
            if not self.send(robot_id, ACTION_CENTERING, pose_dict(target)):
                continue
            time.sleep(1.0)
            if not self.send(robot_id, ACTION_RELEASE):
                continue
            opened = self.wait_until(
                "release open",
                lambda s: gripper_open(s.robots[robot_id]),
                RELEASE_VERIFY_TIMEOUT,
            )
            placed = self.wait_until(
                "handover placed",
                lambda s: object_at_handover(s, object_id)
                and detached_from_eef(s, robot_id, object_id),
                DROP_VERIFY_TIMEOUT,
            )
            self.send(robot_id, ACTION_HOMING)
            if opened and placed:
                log(f"  OK {object_id} resting at handover")
                return True
            log(f"  FAIL handover verifier opened={opened} placed={placed}")
        return False

    def execute_drop_goal(
        self, robot_id: RobotId, object_id: str, goal_color: str
    ) -> bool:
        for attempt in range(1, MAX_STAGE_RETRIES + 1):
            log(
                f"\nSTAGE DROP_GOAL {object_id} -> {goal_color}_goal with {robot_id} attempt {attempt}"
            )
            snap = self.node.snapshot()
            if not likely_held(snap, robot_id, object_id):
                log("  FAIL object is not held; must pick before drop")
                return False
            target = goal_drop_position(goal_color, object_index(object_id))
            if not self.send(robot_id, ACTION_PLACING, pose_dict(target)):
                continue
            time.sleep(1.0)
            if not self.send(robot_id, ACTION_RELEASE):
                continue
            opened = self.wait_until(
                "release open",
                lambda s: gripper_open(s.robots[robot_id]),
                RELEASE_VERIFY_TIMEOUT,
            )
            placed = self.wait_until(
                "goal placed",
                lambda s: (
                    object_id in s.objects
                    and goal_region_contains(s, s.objects[object_id], goal_color)
                    and detached_from_eef(s, robot_id, object_id)
                ),
                DROP_VERIFY_TIMEOUT,
            )
            self.send(robot_id, ACTION_HOMING)
            if opened and placed:
                log(f"  OK {object_id} resting in {goal_color}_goal")
                return True
            log(f"  FAIL goal verifier opened={opened} placed={placed}")
        return False

    def execute_stage(self, stage: Stage) -> bool:
        if stage.kind == "PICK":
            return self.execute_pick(stage.robot_id, stage.object_id)
        if stage.kind == "DROP_HANDOVER":
            return self.execute_drop_handover(stage.robot_id, stage.object_id)
        if stage.kind == "DROP_GOAL":
            if stage.goal_color is None:
                raise ValueError("DROP_GOAL stage missing goal_color")
            return self.execute_drop_goal(
                stage.robot_id, stage.object_id, stage.goal_color
            )
        raise ValueError(f"unknown stage kind: {stage.kind}")

    def execute_plan(self, plan: CommandPlan) -> bool:
        for op in plan.object_plans:
            log(f"\nOBJECT PLAN {op.object_id} -> {op.goal_color}_goal")
            for st in op.stages:
                log(f"  - {st.kind} by {st.robot_id}")
            for st in op.stages:
                if st.kind == "DROP_GOAL":
                    snap = self.node.snapshot()
                    obj = snap.objects.get(st.object_id)
                    if obj and goal_region_contains(
                        snap, obj, st.goal_color or op.goal_color
                    ):
                        log(
                            f"  SKIP stage: {st.object_id} already in {st.goal_color}_goal"
                        )
                        continue
                if not self.execute_stage(st):
                    log(
                        f"\nSTOP: failed stage {st.kind} for {st.object_id} by {st.robot_id}"
                    )
                    return False
        return self.verify_command_done(plan)

    def verify_command_done(self, plan: CommandPlan) -> bool:
        snap = self.node.snapshot()
        failed: list[str] = []
        for a in plan.assignments:
            obj = snap.objects.get(a.object_id)
            if obj is None or not goal_region_contains(snap, obj, a.goal_color):
                failed.append(f"{a.object_id}->{a.goal_color}")
        if failed:
            log("\nCOMMAND INCOMPLETE: " + ", ".join(failed))
            return False
        log("\nCOMMAND COMPLETE: all assigned objects are at requested goals.")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_plan(plan: CommandPlan) -> None:
    log("\nFIXED PLAN")
    for op in plan.object_plans:
        route = " -> ".join(f"{s.kind}({s.robot_id})" for s in op.stages)
        log(f"  {op.object_id}: {route} -> {op.goal_color}_goal")


def main() -> int:
    if ROS_IMPORT_ERROR is not None:
        print(
            f"ROS imports failed. Source the ROS2 workspace first: {ROS_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 2
    rclpy.init()
    node = SceneNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        log("Waiting for ROS2 observations...")
        snap = node.wait_for_observations()
        log(f"Observed {len(snap.objects)} objects and goals={sorted(snap.goals)}")
        command = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
        if not command:
            command = input("Command: ").strip()
        if not command:
            log("No command.")
            return 1
        plan = build_plan(command, node.snapshot())
        if not plan.object_plans:
            log("No executable object plans were generated.")
            return 1
        print_plan(plan)
        ok = Executor(node).execute_plan(plan)
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
