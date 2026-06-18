"""ROS2 state manager for the Isaac Sim world in world.py.

The contracts here are taken from world.py and control_command_gui.py:
ControlCommand.action plus target_pose are sent to the left/right control
services, and MarkerArray is the source of truth for object and goal poses.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from custom_msgs.srv import ControlCommand
from geometry_msgs.msg import Point, Pose
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import Marker, MarkerArray


MARKER_TOPIC = "/world/object_markers"
RGB_CAMERA_TOPIC = "/world/top_camera/image_raw"
CAMERA_POSE_TOPIC = "/world/top_camera/pose"
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}
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

ACTION_MOVING = "Moving"
ACTION_CENTERING = "Centering"
ACTION_PLACING = "Placing"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Release"
ACTION_HOMING = "Homing"
ALLOWED_ACTIONS = {
    ACTION_MOVING,
    ACTION_CENTERING,
    ACTION_PLACING,
    ACTION_GRIP,
    ACTION_RELEASE,
    ACTION_HOMING,
}

TABLE_CENTER_TARGET = (0.6, 0.0, 0.46)
GOAL_TARGET_POSITIONS = {
    "red_goal": (0.272, 0.228, 0.466),
    "blue_goal": (0.928, -0.228, 0.466),
}
VERTICAL_EEF_ORIENTATION_XYZW = (1.0, 0.0, 0.0, 0.0)
GOAL_TO_ROBOT = {
    "red_goal": "left",
    "blue_goal": "right",
}
GOAL_COLORS = {
    "red_goal": "red",
    "blue_goal": "blue",
}


@dataclass
class ServiceResult:
    ok: bool
    message: str
    service_topic: str
    action: str
    elapsed_sec: float


def pose_to_dict(pose: Pose) -> dict[str, Any]:
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


def point_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(
        (left["x"] - right["x"]) ** 2
        + (left["y"] - right["y"]) ** 2
        + (left["z"] - right["z"]) ** 2
    )


def color_name(marker: Marker) -> str:
    r = float(marker.color.r)
    b = float(marker.color.b)
    if r >= 0.65 and r > b:
        return "red"
    if b >= 0.65 and b > r:
        return "blue"
    return "unknown"


def marker_id(marker: Marker) -> str:
    if marker.ns in GOAL_COLORS:
        return marker.ns
    if marker.ns == "table":
        return "table"
    return f"{marker.ns}[{marker.id}]"


def marker_pose_dict(marker: Marker) -> dict[str, Any]:
    return pose_to_dict(marker.pose)


def marker_scale_dict(marker: Marker) -> dict[str, float]:
    return {
        "x": float(marker.scale.x),
        "y": float(marker.scale.y),
        "z": float(marker.scale.z),
    }


def make_pose(position: dict[str, float] | tuple[float, float, float]) -> Pose:
    if isinstance(position, tuple):
        x, y, z = position
    else:
        x, y, z = position["x"], position["y"], position["z"]
    pose = Pose()
    pose.position = Point(x=float(x), y=float(y), z=float(z))
    pose.orientation.x = VERTICAL_EEF_ORIENTATION_XYZW[0]
    pose.orientation.y = VERTICAL_EEF_ORIENTATION_XYZW[1]
    pose.orientation.z = VERTICAL_EEF_ORIENTATION_XYZW[2]
    pose.orientation.w = VERTICAL_EEF_ORIENTATION_XYZW[3]
    return pose


class IsaacRosStateManager(Node):
    def __init__(self) -> None:
        super().__init__("llm_control_agent")
        self._lock = threading.Lock()
        self._markers: dict[str, Marker] = {}
        self._robot_poses: dict[str, Any] = {}
        self._eef_poses: dict[str, Any] = {}
        self._joint_states: dict[str, JointState] = {}
        self._image: dict[str, Any] | None = None
        self._last_marker_time = 0.0

        self.create_subscription(MarkerArray, MARKER_TOPIC, self._markers_cb, 10)
        self.create_subscription(Image, RGB_CAMERA_TOPIC, self._image_cb, 2)

        from geometry_msgs.msg import PoseStamped

        for robot_id, topic in ROBOT_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, rid=robot_id: self._pose_cb(self._robot_poses, rid, msg),
                10,
            )
        for robot_id, topic in EEF_POSE_TOPICS.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, rid=robot_id: self._pose_cb(self._eef_poses, rid, msg),
                10,
            )
        for robot_id, topic in JOINT_STATE_TOPICS.items():
            self.create_subscription(
                JointState,
                topic,
                lambda msg, rid=robot_id: self._joint_state_cb(rid, msg),
                10,
            )

        self._control_clients = {
            robot_id: self.create_client(ControlCommand, topic)
            for robot_id, topic in CONTROL_SERVICE_TOPICS.items()
        }

    def _markers_cb(self, message: MarkerArray) -> None:
        with self._lock:
            self._markers = {marker_id(marker): copy.deepcopy(marker) for marker in message.markers}
            self._last_marker_time = time.time()

    def _pose_cb(self, store: dict[str, Any], robot_id: str, message: Any) -> None:
        with self._lock:
            store[robot_id] = {
                "pose": pose_to_dict(message.pose),
                "stamp": time.time(),
            }

    def _joint_state_cb(self, robot_id: str, message: JointState) -> None:
        with self._lock:
            self._joint_states[robot_id] = copy.deepcopy(message)

    def _image_cb(self, message: Image) -> None:
        with self._lock:
            self._image = {
                "topic": RGB_CAMERA_TOPIC,
                "available": True,
                "width": int(message.width),
                "height": int(message.height),
                "encoding": str(message.encoding),
                "stamp": time.time(),
            }

    def wait_for_initial_snapshot(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot["objects"]:
                return snapshot
            time.sleep(0.05)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            markers = copy.deepcopy(self._markers)
            robot_poses = copy.deepcopy(self._robot_poses)
            eef_poses = copy.deepcopy(self._eef_poses)
            joint_states = copy.deepcopy(self._joint_states)
            image = copy.deepcopy(self._image)
            marker_age = None if not self._last_marker_time else now - self._last_marker_time

        goals: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        table = None
        for key, marker in markers.items():
            entry = {
                "id": key,
                "namespace": marker.ns,
                "marker_id": int(marker.id),
                "color": color_name(marker),
                "pose": marker_pose_dict(marker),
                "scale": marker_scale_dict(marker),
            }
            if marker.ns == "table":
                table = entry
            elif marker.ns in GOAL_COLORS:
                entry["marker_pose"] = entry["pose"]
                entry["service_target_pose"] = pose_to_dict(
                    make_pose(GOAL_TARGET_POSITIONS[marker.ns])
                )
                entry["robot_id"] = GOAL_TO_ROBOT.get(marker.ns)
                goals.append(entry)
            else:
                objects.append(entry)

        goal_ids = {goal["id"] for goal in goals}
        for fallback_goal, robot_id in GOAL_TO_ROBOT.items():
            if fallback_goal not in goal_ids:
                x, y, z = GOAL_TARGET_POSITIONS[fallback_goal]
                goals.append(
                    {
                        "id": fallback_goal,
                        "namespace": fallback_goal,
                        "marker_id": -2,
                        "color": GOAL_COLORS[fallback_goal],
                        "robot_id": robot_id,
                        "pose": pose_to_dict(make_pose((x, y, z))),
                        "service_target_pose": pose_to_dict(make_pose((x, y, z))),
                        "scale": {"x": 0.2, "y": 0.2, "z": 0.008},
                        "source": "fallback_from_world_contract",
                    }
                )

        robots = {}
        for robot_id, service_topic in CONTROL_SERVICE_TOPICS.items():
            joint_state = joint_states.get(robot_id)
            gripper = self._gripper_state(joint_state)
            robots[robot_id] = {
                "id": robot_id,
                "service_topic": service_topic,
                "pose": robot_poses.get(robot_id, {}).get("pose"),
                "pose_age_sec": self._age(robot_poses.get(robot_id), now),
                "end_effector_pose": eef_poses.get(robot_id, {}).get("pose"),
                "end_effector_pose_age_sec": self._age(eef_poses.get(robot_id), now),
                "gripper": gripper,
                "joint_state_available": joint_state is not None,
            }

        if image is not None:
            image["age_sec"] = now - float(image["stamp"])
        else:
            image = {"topic": RGB_CAMERA_TOPIC, "available": False}

        snapshot = {
            "objects": sorted(objects, key=lambda item: item["id"]),
            "goals": sorted(goals, key=lambda item: item["id"]),
            "robots": robots,
            "table": table,
            "table_center": {
                "id": "table_center",
                "pose": pose_to_dict(make_pose(TABLE_CENTER_TARGET)),
            },
            "image_available": bool(image.get("available")),
            "image": image,
            "topics": {
                "markers": MARKER_TOPIC,
                "image": RGB_CAMERA_TOPIC,
                "camera_pose": CAMERA_POSE_TOPIC,
                "robot_pose": ROBOT_POSE_TOPICS,
                "eef_pose": EEF_POSE_TOPICS,
                "joint_state": JOINT_STATE_TOPICS,
            },
            "service_topics": CONTROL_SERVICE_TOPICS,
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "marker_age_sec": marker_age,
            "timestamp": now,
        }
        return snapshot

    @staticmethod
    def _age(stamped: dict[str, Any] | None, now: float) -> float | None:
        if not stamped or "stamp" not in stamped:
            return None
        return now - float(stamped["stamp"])

    @staticmethod
    def _gripper_state(joint_state: JointState | None) -> dict[str, Any]:
        if joint_state is None:
            return {"state": "unknown", "positions": None}
        positions = {}
        for name, position in zip(joint_state.name, joint_state.position):
            if "finger_joint" in name:
                positions[name] = float(position)
        if not positions:
            return {"state": "unknown", "positions": None}
        avg = sum(positions.values()) / len(positions)
        if avg >= 0.03:
            state = "open"
        elif avg <= 0.01:
            state = "closed"
        else:
            state = "moving"
        return {"state": state, "positions": positions}

    def call_control(
        self,
        robot_id: str,
        action: str,
        target_pose: Pose | None,
        service_timeout_sec: float,
    ) -> ServiceResult:
        if robot_id not in self._control_clients:
            return ServiceResult(False, f"Unknown robot_id '{robot_id}'", "", action, 0.0)
        client = self._control_clients[robot_id]
        service_topic = CONTROL_SERVICE_TOPICS[robot_id]
        started = time.monotonic()
        if not client.wait_for_service(timeout_sec=service_timeout_sec):
            return ServiceResult(
                False,
                f"Service not available: {service_topic}",
                service_topic,
                action,
                time.monotonic() - started,
            )
        request = ControlCommand.Request()
        request.action = action
        pose = target_pose if target_pose is not None else make_pose((0.0, 0.0, 0.0))
        request.target_pose.position.x = float(pose.position.x)
        request.target_pose.position.y = float(pose.position.y)
        request.target_pose.position.z = float(pose.position.z)
        request.target_pose.orientation.x = float(pose.orientation.x)
        request.target_pose.orientation.y = float(pose.orientation.y)
        request.target_pose.orientation.z = float(pose.orientation.z)
        request.target_pose.orientation.w = float(pose.orientation.w)
        future = client.call_async(request)
        deadline = time.monotonic() + service_timeout_sec
        while time.monotonic() < deadline:
            if future.done():
                try:
                    response = future.result()
                except Exception as exc:
                    return ServiceResult(
                        False,
                        f"{action} failed: {exc}",
                        service_topic,
                        action,
                        time.monotonic() - started,
                    )
                return ServiceResult(
                    bool(response.success),
                    str(response.message),
                    service_topic,
                    action,
                    time.monotonic() - started,
                )
            time.sleep(0.02)
        return ServiceResult(
            False,
            f"{action} timed out waiting for service response.",
            service_topic,
            action,
            time.monotonic() - started,
        )
