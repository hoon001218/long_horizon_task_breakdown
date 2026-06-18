"""Offline checks for command parsing and task validation.

These tests stub the ROS constants that agents.py imports so they can run
without a sourced ROS2 workspace.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ros_stub = types.ModuleType("ros")
ros_stub.ACTION_GRIP = "Grip"
ros_stub.ACTION_HOMING = "Homing"
ros_stub.ACTION_MOVING = "Moving"
ros_stub.ACTION_PLACING = "Placing"
ros_stub.ACTION_RELEASE = "Release"
sys.modules.setdefault("ros", ros_stub)

import agents


def pose(x: float, y: float, z: float = 0.43) -> dict:
    return {"position": {"x": x, "y": y, "z": z}}


def snapshot() -> dict:
    return {
        "objects": [
            {
                "id": "cube_1[3]",
                "color": "blue",
                "pose": pose(0.82, 0.11),
                "scale": {"x": 0.03, "y": 0.03, "z": 0.03},
            },
            {
                "id": "cube_2[4]",
                "color": "red",
                "pose": pose(0.29, -0.01),
                "scale": {"x": 0.03, "y": 0.03, "z": 0.03},
            },
            {
                "id": "cube_3[5]",
                "color": "blue",
                "pose": pose(0.79, 0.01),
                "scale": {"x": 0.03, "y": 0.03, "z": 0.03},
            },
            {
                "id": "cube_4[6]",
                "color": "red",
                "pose": pose(0.91, 0.16),
                "scale": {"x": 0.03, "y": 0.03, "z": 0.03},
            },
        ],
        "goals": [
            {
                "id": "blue_goal",
                "color": "blue",
                "pose": pose(0.905, -0.205, 0.424),
                "service_target_pose": pose(0.928, -0.228, 0.466),
                "scale": {"x": 0.2, "y": 0.2, "z": 0.008},
                "robot_id": "right",
            },
            {
                "id": "red_goal",
                "color": "red",
                "pose": pose(0.295, 0.205, 0.424),
                "service_target_pose": pose(0.272, 0.228, 0.466),
                "scale": {"x": 0.2, "y": 0.2, "z": 0.008},
                "robot_id": "left",
            },
        ],
        "robots": {
            "left": {"end_effector_pose": pose(0.31, 0.0, 0.49)},
            "right": {"end_effector_pose": pose(0.89, 0.0, 0.49)},
        },
    }


class CommandContextTests(unittest.TestCase):
    def test_all_objects_to_blue_goal_has_no_object_color_filter(self) -> None:
        ctx = agents.build_command_context("Move all objects to blue goal.", snapshot())

        self.assertEqual(ctx["parsed_command"]["object_scope"], "all")
        self.assertEqual(ctx["parsed_command"]["target_goal_id"], "blue_goal")
        self.assertIsNone(ctx["parsed_command"]["object_color_filter"])
        self.assertFalse(ctx["color_matching_requested"])
        self.assertFalse(ctx["goal_color_is_object_filter"])
        self.assertFalse(ctx["object_color_filter_active"])
        self.assertEqual(
            set(ctx["expected_pending_object_ids"]),
            {"cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"},
        )
        self.assertEqual(
            {
                (pair["object_id"], pair["goal_id"])
                for pair in ctx["expected_pending_task_pairs"]
            },
            {
                ("cube_1[3]", "blue_goal"),
                ("cube_2[4]", "blue_goal"),
                ("cube_3[5]", "blue_goal"),
                ("cube_4[6]", "blue_goal"),
            },
        )

    def test_color_matching_phrase_is_explicit_context_only(self) -> None:
        ctx = agents.build_command_context("Sort all objects by matching colors.", snapshot())

        self.assertTrue(ctx["color_matching_requested"])

    def test_goal_color_does_not_filter_red_objects(self) -> None:
        parsed = agents.symbolic_parse_command(
            "Move all objects to blue goal.", snapshot()
        )
        tasks = agents.symbolic_tasks_from_command(parsed, snapshot())

        self.assertEqual({task["goal_id"] for task in tasks}, {"blue_goal"})
        self.assertIn("cube_2[4]", {task["object_id"] for task in tasks})
        self.assertIn("cube_4[6]", {task["object_id"] for task in tasks})

    def test_object_color_filter_still_allows_different_goal_color(self) -> None:
        ctx = agents.build_command_context("Move all red objects to blue goal.", snapshot())
        parsed = agents.symbolic_parse_command(
            "Move all red objects to blue goal.", snapshot()
        )
        tasks = agents.symbolic_tasks_from_command(parsed, snapshot())

        self.assertEqual(ctx["parsed_command"]["object_scope"], "all")
        self.assertEqual(ctx["parsed_command"]["target_goal_id"], "blue_goal")
        self.assertEqual(ctx["parsed_command"]["object_color_filter"], "red")
        self.assertTrue(ctx["object_color_filter_active"])
        self.assertFalse(ctx["goal_color_is_object_filter"])
        self.assertEqual(
            {(task["object_id"], task["goal_id"]) for task in tasks},
            {("cube_2[4]", "blue_goal"), ("cube_4[6]", "blue_goal")},
        )

    def test_task_agent_repairs_incomplete_all_object_output(self) -> None:
        class FakeClient(agents.LLMClient):
            def complete_json(self, agent_name, system_prompt, payload, logger):
                return {
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "object_id": "cube_1[3]",
                            "goal_id": "blue_goal",
                            "robot_id": "right",
                            "status": "pending",
                            "reason": "Incorrectly selected only blue objects.",
                        },
                        {
                            "task_id": "task_2",
                            "object_id": "cube_3[5]",
                            "goal_id": "blue_goal",
                            "robot_id": "right",
                            "status": "pending",
                            "reason": "Incorrectly selected only blue objects.",
                        },
                    ],
                    "reason": "Incomplete output.",
                }

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        logger = FakeLogger()
        task_agent = agents.TaskAgent(FakeClient(), logger)

        tasks = task_agent.create_tasks("Move all objects to blue goal.", snapshot())

        self.assertEqual({task["goal_id"] for task in tasks}, {"blue_goal"})
        self.assertEqual(
            {task["object_id"] for task in tasks},
            {"cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"},
        )
        self.assertIn("task_validation_fallback", [event for event, _ in logger.events])

    def test_task_agent_repairs_goal_side_robot_assignment(self) -> None:
        class FakeClient(agents.LLMClient):
            def complete_json(self, agent_name, system_prompt, payload, logger):
                return {
                    "tasks": [
                        {
                            "task_id": f"task_{index}",
                            "object_id": object_id,
                            "goal_id": "blue_goal",
                            "robot_id": "right",
                            "status": "pending",
                            "reason": "Incorrectly chose the blue-goal side robot.",
                        }
                        for index, object_id in enumerate(
                            ["cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"],
                            start=1,
                        )
                    ],
                    "reason": "All tasks covered but robot assignment follows goal side.",
                }

        class FakeLogger:
            def write(self, event, payload):
                pass

        task_agent = agents.TaskAgent(FakeClient(), FakeLogger())

        tasks = task_agent.create_tasks("Move all objects to blue goal.", snapshot())

        cube_2_task = next(task for task in tasks if task["object_id"] == "cube_2[4]")
        self.assertEqual(cube_2_task["robot_id"], "left")

    def test_main_context_accepts_red_objects_pending_for_blue_goal(self) -> None:
        tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_2[4]",
                "goal_id": "blue_goal",
                "robot_id": "left",
                "status": "pending",
                "reason": "Move red cube to requested blue goal.",
            }
        ]

        ctx = agents.build_command_context("Move all objects to blue goal.", snapshot(), tasks)

        self.assertEqual(ctx["expected_goal_id"], "blue_goal")
        self.assertEqual(ctx["unexpected_pending_task_goal_ids"], [])
        self.assertIn("cube_2[4]", ctx["current_pending_task_object_ids"])

    def test_robot_candidates_do_not_follow_goal_side_only(self) -> None:
        ctx = agents.build_command_context("Move all objects to blue goal.", snapshot())

        cube_2_candidates = ctx["robot_candidates_by_pair"]["cube_2[4]->blue_goal"]

        self.assertEqual(cube_2_candidates["recommended_robot_id"], "left")
        self.assertEqual(cube_2_candidates["candidates"][0]["robot_id"], "left")
        self.assertLess(
            cube_2_candidates["candidates"][0]["pick_distance_xy"],
            cube_2_candidates["candidates"][1]["pick_distance_xy"],
        )

    def test_heuristic_tasks_do_not_follow_goal_side_only(self) -> None:
        response = agents.heuristic_tasks("Move red object to blue goal.", snapshot())

        self.assertEqual(response["tasks"][0]["object_id"], "cube_2[4]")
        self.assertEqual(response["tasks"][0]["goal_id"], "blue_goal")
        self.assertEqual(response["tasks"][0]["robot_id"], "left")

    def test_robot_candidates_include_known_infeasible_robots(self) -> None:
        ctx = agents.build_command_context(
            "Move all objects to blue goal.",
            snapshot(),
            infeasible={("cube_3[5]", "blue_goal"): {"right"}},
        )

        cube_3_candidates = ctx["robot_candidates_by_pair"]["cube_3[5]->blue_goal"]

        self.assertEqual(cube_3_candidates["recommended_robot_id"], "left")
        self.assertTrue(
            next(
                candidate
                for candidate in cube_3_candidates["candidates"]
                if candidate["robot_id"] == "right"
            )["known_infeasible"]
        )

    def test_main_context_flags_same_color_goal_when_command_wants_blue_goal(self) -> None:
        tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_2[4]",
                "goal_id": "red_goal",
                "robot_id": "left",
                "status": "pending",
                "reason": "Incorrect same-color destination.",
            }
        ]

        ctx = agents.build_command_context("Move all objects to blue goal.", snapshot(), tasks)

        self.assertEqual(ctx["unexpected_pending_task_goal_ids"], ["red_goal"])

    def test_main_agent_repairs_invalid_color_replan(self) -> None:
        class FakeClient(agents.LLMClient):
            def __init__(self):
                self.payload = None

            def complete_json(self, agent_name, system_prompt, payload, logger):
                self.payload = payload
                return {
                    "status": "replan",
                    "reason": "The red cube should go to the red goal, not blue goal.",
                }

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_1[3]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
                "reason": "Move requested object to requested goal.",
            },
            {
                "task_id": "task_2",
                "object_id": "cube_2[4]",
                "goal_id": "blue_goal",
                "robot_id": "left",
                "status": "pending",
                "reason": "Move requested object to requested goal.",
            },
            {
                "task_id": "task_3",
                "object_id": "cube_3[5]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
                "reason": "Move requested object to requested goal.",
            },
            {
                "task_id": "task_4",
                "object_id": "cube_4[6]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
                "reason": "Move requested object to requested goal.",
            },
        ]
        logger = FakeLogger()
        client = FakeClient()
        main_agent = agents.MainAgent(client, logger)

        decision = main_agent.decide("Move all objects to blue goal.", tasks, snapshot())

        self.assertEqual(decision["status"], "continue")
        self.assertIn(decision["task_id"], {"task_1", "task_2", "task_3", "task_4"})
        self.assertIn("main_decision_fallback", [event for event, _ in logger.events])
        self.assertTrue(client.payload is not None)
        self.assertFalse(any("reason" in task for task in client.payload["tasks"]))

    def test_main_agent_repairs_color_mismatch_replan_without_goal_name(self) -> None:
        class FakeClient(agents.LLMClient):
            def complete_json(self, agent_name, system_prompt, payload, logger):
                return {
                    "status": "replan",
                    "reason": "Task goals and object colors have a color mismatch.",
                }

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        tasks = agents.regenerate_tasks("Move all objects to blue goal.", snapshot())
        logger = FakeLogger()
        main_agent = agents.MainAgent(FakeClient(), logger)

        decision = main_agent.decide("Move all objects to blue goal.", tasks, snapshot())

        self.assertEqual(decision["status"], "continue")
        self.assertIn("main_decision_fallback", [event for event, _ in logger.events])

    def test_main_agent_does_not_repair_explicit_color_matching_replan(self) -> None:
        class FakeClient(agents.LLMClient):
            def complete_json(self, agent_name, system_prompt, payload, logger):
                return {
                    "status": "replan",
                    "reason": "The red cube should go to the red goal because matching colors were requested.",
                }

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_1[3]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
            },
            {
                "task_id": "task_2",
                "object_id": "cube_2[4]",
                "goal_id": "blue_goal",
                "robot_id": "left",
                "status": "pending",
            },
            {
                "task_id": "task_3",
                "object_id": "cube_3[5]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
            },
            {
                "task_id": "task_4",
                "object_id": "cube_4[6]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
            },
        ]
        logger = FakeLogger()
        main_agent = agents.MainAgent(FakeClient(), logger)

        decision = main_agent.decide(
            "Move all objects to blue goal with matching colors.",
            tasks,
            snapshot(),
        )

        self.assertEqual(decision["status"], "replan")
        self.assertNotIn("main_decision_fallback", [event for event, _ in logger.events])

    def test_infeasible_robot_context_reassigns_task(self) -> None:
        raw_tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_3[5]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
                "reason": "LLM chose right robot.",
            }
        ]
        replan_context = {"infeasible_robots": {"cube_3[5]->blue_goal": ["right"]}}

        validated = agents.validate_tasks(
            raw_tasks,
            snapshot(),
            agents.symbolic_parse_command("Move all objects to blue goal.", snapshot()),
            replan_context,
        )

        self.assertEqual(validated[0]["robot_id"], "left")

    def test_all_robots_infeasible_marks_task_infeasible(self) -> None:
        raw_tasks = [
            {
                "task_id": "task_1",
                "object_id": "cube_3[5]",
                "goal_id": "blue_goal",
                "robot_id": "right",
                "status": "pending",
                "reason": "Both robots have failed.",
            }
        ]
        replan_context = {
            "infeasible_robots": {"cube_3[5]->blue_goal": ["left", "right"]}
        }

        validated = agents.validate_tasks(
            raw_tasks,
            snapshot(),
            agents.symbolic_parse_command("Move all objects to blue goal.", snapshot()),
            replan_context,
        )

        self.assertIsNone(validated[0]["robot_id"])
        self.assertEqual(validated[0]["status"], "infeasible")

    def test_symbolic_replan_marks_task_infeasible_when_all_robots_failed(self) -> None:
        tasks = agents.regenerate_tasks(
            "Move all objects to blue goal.",
            snapshot(),
            infeasible={("cube_3[5]", "blue_goal"): {"left", "right"}},
            reason="Repeated IK failure for all robots.",
        )

        cube_3_task = next(task for task in tasks if task["object_id"] == "cube_3[5]")
        self.assertIsNone(cube_3_task["robot_id"])
        self.assertEqual(cube_3_task["status"], "infeasible")

    def test_symbolic_replan_preserves_all_objects_and_avoids_failed_robot(self) -> None:
        tasks = agents.regenerate_tasks(
            "Move all objects to blue goal.",
            snapshot(),
            infeasible={("cube_3[5]", "blue_goal"): {"right"}},
            reason="Repeated IK failure.",
        )

        self.assertEqual({task["goal_id"] for task in tasks}, {"blue_goal"})
        self.assertEqual(
            {task["object_id"] for task in tasks},
            {"cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"},
        )
        cube_3_task = next(task for task in tasks if task["object_id"] == "cube_3[5]")
        self.assertEqual(cube_3_task["robot_id"], "left")

    def test_heuristic_critic_replans_on_ik_failure(self) -> None:
        response = agents.heuristic_critic(
            {
                "task": {
                    "task_id": "task_1",
                    "object_id": "cube_3[5]",
                    "goal_id": "blue_goal",
                    "robot_id": "right",
                },
                "results": [
                    {
                        "ok": False,
                        "message": "Action 'Moving' rejected: IK could not solve phase 1.",
                    }
                ],
                "after_snapshot": snapshot(),
            }
        )

        self.assertEqual(response["status"], "replan")


if __name__ == "__main__":
    unittest.main()
