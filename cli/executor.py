"""Validation and execution for primitive ControlCommand actions."""

from __future__ import annotations

import time
from typing import Any

from memory import JsonlLogger
from ros import (
    ACTION_CENTERING,
    ACTION_GRIP,
    ACTION_HOMING,
    ACTION_MOVING,
    ACTION_PLACING,
    ACTION_RELEASE,
    ALLOWED_ACTIONS,
    IsaacRosStateManager,
    make_pose,
    point_distance,
)


class ActionValidationError(ValueError):
    pass


class ActionExecutor:
    def __init__(
        self,
        ros: IsaacRosStateManager,
        logger: JsonlLogger,
        service_timeout_sec: float = 6.0,
        motion_timeout_sec: float = 15.0,
        pre_grip_delay_sec: float = 2.0,
    ) -> None:
        self.ros = ros
        self.logger = logger
        self.service_timeout_sec = service_timeout_sec
        self.motion_timeout_sec = motion_timeout_sec
        self.pre_grip_delay_sec = pre_grip_delay_sec

    def execute_plan(
        self, actions: list[dict[str, Any]], snapshot: dict[str, Any]
    ) -> list[dict[str, Any]]:
        results = []
        for index, action in enumerate(actions):
            try:
                self.validate_action(action, snapshot)
                self.apply_pre_action_delay(action, index)
                result = self.execute_action(action)
            except Exception as exc:
                result = {
                    "ok": False,
                    "action": action,
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                }
            self.logger.write("action_result", {"index": index, "result": result})
            results.append(result)
            if not result.get("ok"):
                break
            snapshot = self.ros.snapshot()
        return results

    def apply_pre_action_delay(self, action: dict[str, Any], index: int) -> None:
        if action.get("action") != ACTION_GRIP:
            return
        if self.pre_grip_delay_sec <= 0.0:
            return
        self.logger.write(
            "pre_grip_delay",
            {
                "index": index,
                "robot_id": action.get("robot_id"),
                "delay_sec": self.pre_grip_delay_sec,
                "reason": "Wait before Grip so queued approach motion can physically settle.",
            },
        )
        time.sleep(self.pre_grip_delay_sec)

    def validate_action(self, action: dict[str, Any], snapshot: dict[str, Any]) -> None:
        name = action.get("action")
        if name not in ALLOWED_ACTIONS:
            raise ActionValidationError(f"Unsupported action: {name}")
        robot_id = action.get("robot_id")
        if robot_id not in snapshot.get("robots", {}):
            raise ActionValidationError(f"Unknown robot_id: {robot_id}")
        object_ids = {item["id"] for item in snapshot.get("objects", [])}
        goal_ids = {item["id"] for item in snapshot.get("goals", [])}
        if name == ACTION_MOVING and action.get("object_id") not in object_ids:
            raise ActionValidationError(
                f"Moving requires an existing object_id, got {action.get('object_id')}"
            )
        if name == ACTION_PLACING and action.get("goal_id") not in goal_ids:
            raise ActionValidationError(
                f"Placing requires an existing goal_id, got {action.get('goal_id')}"
            )
        if name == ACTION_CENTERING and action.get("target_id", "table_center") != "table_center":
            raise ActionValidationError("Centering only supports target_id=table_center")

    def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.ros.snapshot()
        name = action["action"]
        target_pose = self.resolve_target_pose(action, snapshot)
        service_result = self.ros.call_control(
            robot_id=action["robot_id"],
            action=name,
            target_pose=target_pose,
            service_timeout_sec=self.service_timeout_sec,
        )
        result = {
            "ok": service_result.ok,
            "message": service_result.message,
            "service_topic": service_result.service_topic,
            "elapsed_sec": service_result.elapsed_sec,
            "action": action,
            "target_pose": pose_to_result(target_pose),
        }
        if service_result.ok:
            result["observation"] = self.wait_for_observation(action, target_pose)
            if result["observation"].get("settled") is False:
                result["ok"] = False
                result["message"] = result["observation"].get(
                    "message", "Action service succeeded but observation did not settle."
                )
        return result

    def resolve_target_pose(
        self, action: dict[str, Any], snapshot: dict[str, Any]
    ):
        name = action["action"]
        if name == ACTION_MOVING:
            target = self.find_by_id(snapshot["objects"], action["object_id"])
            return make_pose(target["pose"]["position"])
        if name == ACTION_PLACING:
            target = self.find_by_id(snapshot["goals"], action["goal_id"])
            target_pose = target.get("service_target_pose", target["pose"])
            return make_pose(target_pose["position"])
        if name == ACTION_CENTERING:
            return make_pose(snapshot["table_center"]["pose"]["position"])
        return None

    @staticmethod
    def find_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
        for item in items:
            if item["id"] == item_id:
                return item
        raise ActionValidationError(f"Unknown target id: {item_id}")

    def wait_for_observation(self, action: dict[str, Any], target_pose) -> dict[str, Any]:
        name = action["action"]
        if name in {ACTION_MOVING, ACTION_CENTERING, ACTION_PLACING, ACTION_HOMING}:
            return self.wait_for_motion(action, target_pose)
        if name in {ACTION_GRIP, ACTION_RELEASE}:
            return self.wait_for_gripper(action)
        return {"settled": True}

    def wait_for_motion(self, action: dict[str, Any], target_pose) -> dict[str, Any]:
        if action["action"] == ACTION_HOMING or target_pose is None:
            time.sleep(1.0)
            return {"settled": True, "note": "Homing was queued; exact home pose is internal to world.py."}
        target = {
            "x": float(target_pose.position.x),
            "y": float(target_pose.position.y),
            "z": float(target_pose.position.z),
        }
        deadline = time.monotonic() + self.motion_timeout_sec
        last_distance = None
        while time.monotonic() < deadline:
            snapshot = self.ros.snapshot()
            robot = snapshot["robots"].get(action["robot_id"], {})
            eef_pose = robot.get("end_effector_pose")
            if eef_pose is not None:
                last_distance = point_distance(eef_pose["position"], target)
                if last_distance <= 0.04:
                    return {"settled": True, "eef_distance_to_target": last_distance}
            time.sleep(0.1)
        return {
            "settled": False,
            "eef_distance_to_target": last_distance,
            "message": "Timed out waiting for end-effector to reach target neighborhood.",
        }

    def wait_for_gripper(self, action: dict[str, Any]) -> dict[str, Any]:
        desired = "closed" if action["action"] == ACTION_GRIP else "open"
        deadline = time.monotonic() + min(self.motion_timeout_sec, 4.0)
        state = "unknown"
        while time.monotonic() < deadline:
            snapshot = self.ros.snapshot()
            robot = snapshot["robots"].get(action["robot_id"], {})
            state = robot.get("gripper", {}).get("state", "unknown")
            if state == desired:
                return {"settled": True, "gripper_state": state}
            time.sleep(0.1)
        return {"settled": state != "unknown", "gripper_state": state}


def pose_to_result(pose) -> dict[str, Any] | None:
    if pose is None:
        return None
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
