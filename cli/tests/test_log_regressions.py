"""Regression tests built from the real logs that exposed the color-goal bug."""

from __future__ import annotations

import json
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


BLUE_ONLY_LOG_PATH = ROOT / "logs" / "run_20260618T092227Z.jsonl"
MAIN_COLOR_REPLAN_LOG_PATH = ROOT / "logs" / "run_20260618T092911Z.jsonl"


def read_event(event_name: str, log_path: Path = BLUE_ONLY_LOG_PATH) -> dict:
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == event_name:
                return event["payload"]
    raise AssertionError(f"Missing event in regression log: {event_name}")


class LogRegressionTests(unittest.TestCase):
    def test_logged_typo_command_still_means_all_objects_to_blue_goal(self) -> None:
        command = read_event("run_started")["command"]
        snapshot = read_event("initial_snapshot")["snapshot"]

        ctx = agents.build_command_context(command, snapshot)

        self.assertEqual(ctx["parsed_command"]["object_scope"], "all")
        self.assertEqual(ctx["parsed_command"]["target_goal_id"], "blue_goal")
        self.assertIsNone(ctx["parsed_command"]["object_color_filter"])
        self.assertEqual(
            set(ctx["expected_pending_object_ids"]),
            {"cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"},
        )

    def test_logged_blue_only_taskagent_output_is_repaired(self) -> None:
        command = read_event("run_started")["command"]
        snapshot = read_event("initial_snapshot")["snapshot"]
        bad_logged_tasks = read_event("tasks_created")["tasks"]

        class FakeClient(agents.LLMClient):
            def complete_json(self, agent_name, system_prompt, payload, logger):
                return {
                    "tasks": bad_logged_tasks,
                    "reason": "Replay the incomplete historical TaskAgent output.",
                }

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        logger = FakeLogger()
        task_agent = agents.TaskAgent(FakeClient(), logger)

        repaired = task_agent.create_tasks(command, snapshot)

        self.assertEqual({task["goal_id"] for task in repaired}, {"blue_goal"})
        self.assertEqual(
            {task["object_id"] for task in repaired},
            {"cube_1[3]", "cube_2[4]", "cube_3[5]", "cube_4[6]"},
        )
        self.assertIn("task_validation_fallback", [event for event, _ in logger.events])

    def test_logged_mainagent_color_replan_is_repaired(self) -> None:
        command = read_event("run_started", MAIN_COLOR_REPLAN_LOG_PATH)["command"]
        snapshot = read_event("initial_snapshot", MAIN_COLOR_REPLAN_LOG_PATH)["snapshot"]
        tasks = read_event("tasks_created", MAIN_COLOR_REPLAN_LOG_PATH)["tasks"]
        bad_decision = read_event("main_decision", MAIN_COLOR_REPLAN_LOG_PATH)["decision"]

        class FakeClient(agents.LLMClient):
            def __init__(self):
                self.payload = None

            def complete_json(self, agent_name, system_prompt, payload, logger):
                self.payload = payload
                return bad_decision

        class FakeLogger:
            def __init__(self):
                self.events = []

            def write(self, event, payload):
                self.events.append((event, payload))

        logger = FakeLogger()
        client = FakeClient()
        main_agent = agents.MainAgent(client, logger)

        decision = main_agent.decide(command, tasks, snapshot)

        self.assertEqual(decision["status"], "continue")
        self.assertIn(decision["task_id"], {task["task_id"] for task in tasks})
        self.assertIn("main_decision_fallback", [event for event, _ in logger.events])
        self.assertFalse(any("reason" in task for task in client.payload["tasks"]))


if __name__ == "__main__":
    unittest.main()
