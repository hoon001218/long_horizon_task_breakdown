"""Offline checks for repeated IK failure tracking.

The main module normally imports ROS2 runtime modules. These tests stub only
the imports needed to exercise the pure policy helpers.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

rclpy_stub = types.ModuleType("rclpy")
rclpy_stub.init = lambda: None
rclpy_stub.shutdown = lambda: None
executors_stub = types.ModuleType("rclpy.executors")
executors_stub.MultiThreadedExecutor = object
sys.modules["rclpy"] = rclpy_stub
sys.modules["rclpy.executors"] = executors_stub

ros_stub = types.ModuleType("ros")
ros_stub.ACTION_GRIP = "Grip"
ros_stub.ACTION_HOMING = "Homing"
ros_stub.ACTION_MOVING = "Moving"
ros_stub.ACTION_PLACING = "Placing"
ros_stub.ACTION_RELEASE = "Release"
ros_stub.IsaacRosStateManager = object
sys.modules["ros"] = ros_stub

executor_stub = types.ModuleType("executor")
executor_stub.ActionExecutor = object
sys.modules["executor"] = executor_stub

import main


def ik_result(x: float, y: float, z: float = 0.43) -> dict:
    return {
        "ok": False,
        "message": "Action 'Moving' rejected: IK could not solve phase 1.",
        "action": {"action": "Moving", "robot_id": "right", "object_id": "cube_3[5]"},
        "target_pose": {"position": {"x": x, "y": y, "z": z}},
    }


def pose(x: float, y: float, z: float = 0.43) -> dict:
    return {"position": {"x": x, "y": y, "z": z}}


def minimal_snapshot() -> dict:
    return {
        "objects": [
            {
                "id": "cube_1[3]",
                "color": "blue",
                "pose": pose(0.82, 0.11),
                "scale": {"x": 0.03, "y": 0.03, "z": 0.03},
            }
        ],
        "goals": [
            {
                "id": "blue_goal",
                "color": "blue",
                "pose": pose(0.905, -0.205, 0.424),
                "service_target_pose": pose(0.928, -0.228, 0.466),
                "scale": {"x": 0.2, "y": 0.2, "z": 0.008},
                "robot_id": "right",
            }
        ],
        "robots": {"right": {"end_effector_pose": pose(0.89, 0.0, 0.49)}},
    }


class IkPolicyTests(unittest.TestCase):
    def test_task_completion_requires_snapshot_goal_membership(self) -> None:
        task = {
            "task_id": "task_1",
            "object_id": "cube_1[3]",
            "goal_id": "blue_goal",
            "robot_id": "right",
            "status": "pending",
        }
        snapshot = minimal_snapshot()

        self.assertFalse(main.task_complete_in_snapshot(task, snapshot))

        snapshot["objects"][0]["pose"] = pose(0.905, -0.205, 0.44)

        self.assertTrue(main.task_complete_in_snapshot(task, snapshot))

    def test_completed_task_reopens_when_snapshot_no_longer_satisfies_goal(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        memory = main.TaskMemory(FakeLogger())
        memory.tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_1[3]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "completed",
                "completion_reason": "Previously verified.",
            }
        ]

        main.update_task_completion(memory, minimal_snapshot())

        self.assertEqual(memory.tasks[0]["status"], "pending")
        self.assertEqual(memory.logger.events[0][0], "task_reopened")

    def test_choose_task_ignores_infeasible_tasks(self) -> None:
        tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_3[5]",
                "goal_id": "blue_goal",
                "robot_id": None,
                "status": "infeasible",
            },
            {
                "task_id": "task_2",
                "object_id": "cube_4[6]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
            },
        ]

        self.assertEqual(main.choose_task(tasks, "task_1")["task_id"], "task_2")
        self.assertEqual(main.infeasible_tasks(tasks)[0]["task_id"], "task_1")

    def test_unchanged_replan_marks_pending_tasks_infeasible(self) -> None:
        class FakeTaskAgent:
            def create_tasks(self, user_command, snapshot, replan_context=None):
                return [
                    {
                        "task_id": "task_1",
                        "object_id": "cube_1[3]",
                        "goal_id": "blue_goal",
                        "robot_id": "right",
                        "status": "pending",
                    }
                ]

        class FakeMemory:
            def __init__(self):
                self.tasks = [
                    {
                        "task_id": "task_1",
                        "object_id": "cube_1[3]",
                        "goal_id": "blue_goal",
                        "robot_id": "right",
                        "status": "pending",
                    }
                ]

            def set_tasks(self, tasks):
                self.tasks = tasks

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        memory = FakeMemory()
        logger = FakeLogger()

        tasks = main.replan_tasks(
            FakeTaskAgent(),
            "Move all objects to blue goal.",
            minimal_snapshot(),
            memory,
            logger,
            {},
            "MainAgent requested replan but task list did not change.",
        )

        self.assertEqual(tasks[0]["status"], "infeasible")
        self.assertIn("replan_unchanged", [event for event, _ in logger.events])

    def test_initial_ik_replan_does_not_mark_task_infeasible(self) -> None:
        class FakeTaskAgent:
            def create_tasks(self, user_command, snapshot, replan_context=None):
                return [
                    {
                        "task_id": "task_1",
                        "object_id": "cube_1[3]",
                        "goal_id": "blue_goal",
                        "robot_id": "right",
                        "status": "pending",
                    }
                ]

        class FakeMemory:
            def __init__(self):
                self.tasks = [
                    {
                        "task_id": "task_1",
                        "object_id": "cube_1[3]",
                        "goal_id": "blue_goal",
                        "robot_id": "right",
                        "status": "pending",
                    }
                ]

            def set_tasks(self, tasks):
                self.tasks = tasks

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        logger = FakeLogger()

        tasks = main.replan_tasks(
            FakeTaskAgent(),
            "Move all objects to blue goal.",
            minimal_snapshot(),
            FakeMemory(),
            logger,
            {},
            "IK could not solve the target; Critic requested replan.",
        )

        self.assertEqual(tasks[0]["status"], "pending")
        replan_event = next(payload for event, payload in logger.events if event == "replan_unchanged")
        self.assertTrue(replan_event["deferred_for_initial_ik_failure"])

    def test_same_pose_reaches_repeated_failure_threshold(self) -> None:
        task = {"object_id": "cube_3[5]", "goal_id": "blue_goal", "robot_id": "right"}
        failures = {}

        first = main.collect_repeated_ik_failures(task, [ik_result(0.79, 0.01)], failures, 2)
        second = main.collect_repeated_ik_failures(task, [ik_result(0.79, 0.01)], failures, 2)

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["robot_id"], "right")
        self.assertEqual(second[0]["target_pose_signature"], "0.790,0.010,0.430")

    def test_different_pose_does_not_share_failure_count(self) -> None:
        task = {"object_id": "cube_3[5]", "goal_id": "blue_goal", "robot_id": "right"}
        failures = {}

        main.collect_repeated_ik_failures(task, [ik_result(0.79, 0.01)], failures, 2)
        repeated = main.collect_repeated_ik_failures(
            task, [ik_result(0.82, 0.04)], failures, 2
        )

        self.assertEqual(repeated, [])
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
