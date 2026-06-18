"""Closed-loop LLM controller for the Isaac Sim ROS2 world."""

from __future__ import annotations

import argparse
import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from agents import (
    ActionAgent,
    CriticAgent,
    MainAgent,
    TaskAgent,
    load_dotenv,
    make_llm_client,
    object_in_goal,
    regenerate_tasks,
)
from executor import ActionExecutor
from memory import JsonlLogger, TaskMemory
from ros import IsaacRosStateManager


def cli_line(text: str = "") -> None:
    print(text, flush=True)


def cli_section(title: str) -> None:
    cli_line()
    cli_line(f"== {title} ==")


def cli_agent(agent_name: str, payload: dict | list) -> None:
    cli_line(f"[{agent_name}]")
    cli_line(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM closed-loop agent for world.py ControlCommand services."
    )
    parser.add_argument("command", nargs="*", help="Natural language task command.")
    parser.add_argument("--planner", choices=["auto", "openai", "heuristic"], default=None)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-task-failures", type=int, default=3)
    parser.add_argument("--snapshot-timeout", type=float, default=8.0)
    parser.add_argument("--service-timeout", type=float, default=6.0)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--pre-grip-delay", type=float, default=2.0)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--quiet-agents",
        action="store_true",
        help="Do not print agent decisions to the CLI.",
    )
    return parser.parse_args()


def update_task_completion(memory: TaskMemory, snapshot: dict) -> None:
    for task in memory.tasks:
        if task.get("status") == "infeasible":
            continue
        complete = task_complete_in_snapshot(task, snapshot)
        if task.get("status", "pending") == "pending" and complete:
            memory.mark_completed(task["task_id"], "Snapshot places object inside goal zone.")
        elif task.get("status") == "completed" and not complete:
            memory.mark_pending(
                task["task_id"],
                "Snapshot no longer places object inside goal zone.",
            )


def task_complete_in_snapshot(task: dict, snapshot: dict) -> bool:
    if task.get("status") == "infeasible":
        return False
    obj = next(
        (item for item in snapshot.get("objects", []) if item["id"] == task["object_id"]),
        None,
    )
    goal = next(
        (item for item in snapshot.get("goals", []) if item["id"] == task["goal_id"]),
        None,
    )
    return bool(obj and goal and object_in_goal(obj, goal))


def choose_task(tasks: list[dict], task_id: str | None) -> dict | None:
    if task_id:
        for task in tasks:
            if task.get("task_id") == task_id and task.get("status", "pending") == "pending":
                return task
    return next((task for task in tasks if task.get("status", "pending") == "pending"), None)


def infeasible_tasks(tasks: list[dict]) -> list[dict]:
    return [task for task in tasks if task.get("status") == "infeasible"]


def task_signature(tasks: list[dict]) -> tuple:
    return tuple(
        sorted(
            (
                task.get("object_id"),
                task.get("goal_id"),
                task.get("robot_id"),
                task.get("status", "pending"),
            )
            for task in tasks
        )
    )


def mark_pending_tasks_infeasible(tasks: list[dict], reason: str) -> list[dict]:
    marked = []
    for task in tasks:
        updated = dict(task)
        if updated.get("status", "pending") == "pending":
            updated["status"] = "infeasible"
            updated["reason"] = reason
        marked.append(updated)
    return marked


def ik_replan_without_known_infeasible(
    reason: str, infeasible_robots: dict[tuple[str, str], set[str]]
) -> bool:
    if infeasible_robots:
        return False
    normalized = reason.lower()
    return "ik" in normalized or "inverse kinematics" in normalized


def replan_tasks(
    task_agent: TaskAgent,
    user_command: str,
    snapshot: dict,
    memory: TaskMemory,
    logger: JsonlLogger,
    infeasible_robots: dict[tuple[str, str], set[str]],
    reason: str,
) -> list[dict]:
    replan_context = {
        "reason": reason,
        "infeasible_robots": {
            f"{object_id}->{goal_id}": sorted(robots)
            for (object_id, goal_id), robots in infeasible_robots.items()
        },
        "previous_tasks": memory.tasks,
    }
    previous_signature = task_signature(memory.tasks)
    tasks = task_agent.create_tasks(
        user_command,
        snapshot,
        replan_context=replan_context,
    )
    used_fallback = False
    if not tasks or task_signature(tasks) == previous_signature:
        used_fallback = True
        tasks = regenerate_tasks(
            user_command,
            snapshot,
            previous_tasks=memory.tasks,
            infeasible=infeasible_robots,
            reason=reason,
        )
    if tasks and task_signature(tasks) == previous_signature:
        logger.write(
            "replan_unchanged",
            {
                "reason": reason,
                "deferred_for_initial_ik_failure": ik_replan_without_known_infeasible(
                    reason, infeasible_robots
                ),
                "tasks": tasks,
            },
        )
        if not ik_replan_without_known_infeasible(reason, infeasible_robots):
            tasks = mark_pending_tasks_infeasible(
                tasks,
                "Replan produced an unchanged task list; aborting to avoid a replan loop.",
            )
    logger.write(
        "tasks_replanned",
        {
            "reason": reason,
            "infeasible_robots": replan_context["infeasible_robots"],
            "used_fallback": used_fallback,
            "tasks": tasks,
        },
    )
    memory.set_tasks(tasks)
    return tasks


def target_pose_signature(result: dict) -> str:
    target_pose = result.get("target_pose") or {}
    position = target_pose.get("position") if isinstance(target_pose, dict) else None
    if not isinstance(position, dict):
        return "unknown"
    return ",".join(
        f"{float(position[axis]):.3f}"
        for axis in ("x", "y", "z")
        if axis in position
    )


def ik_failure_key(task: dict, result: dict) -> tuple[str, str, str, str, str] | None:
    action = result.get("action") or {}
    message = str(result.get("message", ""))
    if "IK could not solve" not in message:
        return None
    action_name = action.get("action")
    if action_name not in {"Moving", "Placing", "Centering", "Homing"}:
        return None
    robot_id = action.get("robot_id") or task.get("robot_id")
    object_id = task.get("object_id") or action.get("object_id")
    goal_id = task.get("goal_id")
    if not robot_id or not object_id or not goal_id:
        return None
    return (object_id, goal_id, robot_id, action_name, target_pose_signature(result))


def collect_repeated_ik_failures(
    task: dict,
    results: list[dict],
    ik_failures: dict[tuple[str, str, str, str, str], int],
    threshold: int,
) -> list[dict]:
    repeated = []
    for result in results:
        key = ik_failure_key(task, result)
        if key is None:
            continue
        ik_failures[key] = ik_failures.get(key, 0) + 1
        count = ik_failures[key]
        if count >= threshold:
            object_id, goal_id, robot_id, action_name, target_signature = key
            repeated.append(
                {
                    "object_id": object_id,
                    "goal_id": goal_id,
                    "robot_id": robot_id,
                    "action": action_name,
                    "target_pose_signature": target_signature,
                    "ik_failure_count": count,
                }
            )
    return repeated


def main() -> int:
    load_dotenv()
    args = parse_args()
    user_command = " ".join(args.command).strip() or input("Command> ").strip()
    planner_mode = args.planner or __import__("os").environ.get("LLM_MODE", "auto")

    logger = JsonlLogger(args.log_dir)
    logger.write("run_started", {"command": user_command, "planner_mode": planner_mode})
    show_agents = not args.quiet_agents
    cli_section("Run")
    cli_line(f"Command: {user_command}")
    cli_line(f"Planner: {planner_mode}")
    cli_line(f"Log: {logger.path}")

    rclpy.init()
    ros_node = IsaacRosStateManager()
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(ros_node)
    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()

    try:
        snapshot = ros_node.wait_for_initial_snapshot(args.snapshot_timeout)
        logger.write("initial_snapshot", {"snapshot": snapshot})
        if show_agents:
            cli_section("Snapshot")
            cli_line(
                "objects={objects} goals={goals} robots={robots} image={image}".format(
                    objects=len(snapshot.get("objects", [])),
                    goals=len(snapshot.get("goals", [])),
                    robots=",".join(sorted(snapshot.get("robots", {}))),
                    image="yes" if snapshot.get("image_available") else "no",
                )
            )
        if not snapshot.get("objects"):
            logger.write("abort", {"reason": "No objects found in /world/object_markers."})
            print("No objects found in /world/object_markers.")
            return 2

        llm_client = make_llm_client(planner_mode)
        task_agent = TaskAgent(llm_client, logger)
        main_agent = MainAgent(llm_client, logger)
        action_agent = ActionAgent(llm_client, logger)
        critic_agent = CriticAgent(llm_client, logger)
        action_executor = ActionExecutor(
            ros_node,
            logger,
            service_timeout_sec=args.service_timeout,
            motion_timeout_sec=args.motion_timeout,
            pre_grip_delay_sec=args.pre_grip_delay,
        )
        memory = TaskMemory(logger)
        ik_failures: dict[tuple[str, str, str, str, str], int] = {}
        infeasible_robots: dict[tuple[str, str], set[str]] = {}

        tasks = task_agent.create_tasks(user_command, snapshot)
        if show_agents:
            cli_section("Agent Decisions")
            cli_agent("TaskAgent", {"tasks": tasks})
        if not tasks:
            logger.write("abort", {"reason": "TaskAgent produced no valid tasks."})
            print("TaskAgent produced no valid tasks from the current snapshot.")
            return 3
        memory.set_tasks(tasks)

        for iteration in range(1, args.max_iterations + 1):
            if show_agents:
                cli_section(f"Iteration {iteration}")
            snapshot = ros_node.snapshot()
            update_task_completion(memory, snapshot)
            blocked_tasks = infeasible_tasks(memory.tasks)
            if blocked_tasks:
                logger.write("abort", {"reason": "infeasible_tasks", "tasks": blocked_tasks})
                print("Aborted: one or more tasks have no feasible robot assignment.")
                return 9
            decision = main_agent.decide(user_command, memory.tasks, snapshot)
            logger.write("main_decision", {"iteration": iteration, "decision": decision})
            if show_agents:
                cli_agent("MainAgent", decision)

            if decision["status"] == "complete":
                blocked_tasks = infeasible_tasks(memory.tasks)
                if blocked_tasks:
                    logger.write("abort", {"reason": "infeasible_tasks", "tasks": blocked_tasks})
                    print("Aborted: one or more tasks have no feasible robot assignment.")
                    return 9
                print(f"Completed: {decision.get('reason', 'all tasks complete')}")
                return 0
            if decision["status"] == "abort":
                print(f"Aborted: {decision.get('reason', 'MainAgent aborted')}")
                return 4
            if decision["status"] == "replan":
                replanned = replan_tasks(
                    task_agent,
                    user_command,
                    snapshot,
                    memory,
                    logger,
                    infeasible_robots,
                    decision.get("reason", "MainAgent requested replan."),
                )
                if show_agents:
                    cli_agent("TaskRegenerator", {"tasks": replanned})
                continue

            task = choose_task(memory.tasks, decision.get("task_id"))
            if task is None:
                blocked_tasks = infeasible_tasks(memory.tasks)
                if blocked_tasks:
                    logger.write("abort", {"reason": "infeasible_tasks", "tasks": blocked_tasks})
                    print("Aborted: one or more tasks have no feasible robot assignment.")
                    return 9
                print("Completed: no pending tasks remain.")
                return 0

            before_snapshot = ros_node.snapshot()
            plan = action_agent.plan(task, before_snapshot)
            logger.write("action_plan", {"iteration": iteration, "task": task, "plan": plan})
            if show_agents:
                cli_agent("SelectedTask", task)
                cli_agent("ActionAgent", {"actions": plan})
            if not plan:
                failures = memory.record_failure(task["task_id"], "ActionAgent produced no valid actions.")
                if failures >= args.max_task_failures:
                    print(f"Aborted: no valid plan for {task['task_id']}.")
                    return 5
                continue

            results = action_executor.execute_plan(plan, before_snapshot)
            if show_agents:
                cli_agent("ActionExecutor", {"results": results})
            repeated_ik = collect_repeated_ik_failures(
                task, results, ik_failures, threshold=2
            )
            if repeated_ik:
                for repeated_failure in repeated_ik:
                    object_id = repeated_failure["object_id"]
                    goal_id = repeated_failure["goal_id"]
                    robot_id = repeated_failure["robot_id"]
                    infeasible_robots.setdefault((object_id, goal_id), set()).add(robot_id)
                    logger.write(
                        "robot_marked_infeasible",
                        repeated_failure,
                    )
                replanned = replan_tasks(
                    task_agent,
                    user_command,
                    ros_node.snapshot(),
                    memory,
                    logger,
                    infeasible_robots,
                    "Repeated IK failure; trying another feasible robot if available.",
                )
                if show_agents:
                    cli_agent(
                        "IKPolicy",
                        {
                            "repeated_failures": repeated_ik,
                            "replanned_tasks": replanned,
                        },
                    )
                continue
            time.sleep(0.5)
            after_snapshot = ros_node.snapshot()
            update_task_completion(memory, after_snapshot)
            critic = critic_agent.evaluate(task, plan, results, before_snapshot, after_snapshot)
            logger.write("critic_decision", {"iteration": iteration, "critic": critic})
            if show_agents:
                cli_agent("CriticAgent", critic)

            if critic["status"] == "success":
                verification_snapshot = ros_node.snapshot()
                update_task_completion(memory, verification_snapshot)
                if task.get("status") != "completed":
                    logger.write(
                        "critic_success_unverified",
                        {
                            "task": task,
                            "critic": critic,
                            "reason": "Critic reported success but Snapshot does not place object inside goal.",
                        },
                    )
                    failures = memory.record_failure(
                        task["task_id"],
                        "Critic success was not verified by Snapshot.",
                    )
                    if failures >= args.max_task_failures:
                        print(f"Aborted: {task['task_id']} exceeded retry limit.")
                        return 6
                else:
                    memory.reset_failures(task["task_id"])
            elif critic["status"] == "retry":
                failures = memory.record_failure(task["task_id"], critic.get("reason", "retry"))
                if failures >= args.max_task_failures:
                    print(f"Aborted: {task['task_id']} exceeded retry limit.")
                    return 6
            elif critic["status"] == "replan":
                replanned = replan_tasks(
                    task_agent,
                    user_command,
                    ros_node.snapshot(),
                    memory,
                    logger,
                    infeasible_robots,
                    critic.get("reason", "Critic requested replan."),
                )
                if show_agents:
                    cli_agent("TaskRegenerator", {"tasks": replanned})
            elif critic["status"] == "abort":
                print(f"Aborted: {critic.get('reason', 'Critic aborted')}")
                return 7

        print("Aborted: max iterations reached.")
        logger.write("abort", {"reason": "max_iterations_reached"})
        return 8
    finally:
        ros_executor.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        logger.write("run_finished", {"log_file": str(logger.path)})


if __name__ == "__main__":
    raise SystemExit(main())
