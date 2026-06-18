"""LLM-backed agents plus conservative fallback planners."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory import JsonlLogger
from ros import ACTION_GRIP, ACTION_HOMING, ACTION_MOVING, ACTION_PLACING, ACTION_RELEASE


PROMPT_DIR = Path("prompts")
OBJECT_WORDS = ("object", "objects", "cube", "cubes", "capsule", "capsules", "sphere", "spheres")
ALL_OBJECT_PATTERNS = (
    "all object",
    "all objects",
    "every object",
    "every objects",
    "all cube",
    "all cubes",
    "every cube",
    "every cubes",
)
COLOR_MATCH_PATTERNS = (
    "same color",
    "same colors",
    "matching color",
    "matching colors",
    "color match",
    "color matched",
    "match colors",
    "sort by color",
    "sorted by color",
    "respective color",
    "corresponding color",
)
INVALID_COLOR_REPLAN_PATTERNS = (
    "same color",
    "same-color",
    "matching color",
    "matching colors",
    "object color",
    "color mismatch",
    "color conflict",
    "contradict",
    "should go",
    "belongs",
    "belong",
)
COLOR_ONLY_REPLAN_PATTERNS = (
    "same color",
    "same-color",
    "matching color",
    "matching colors",
    "object color",
    "color mismatch",
    "color conflict",
)


@dataclass
class ParsedCommand:
    object_scope: str
    target_goal_id: str | None
    object_color_filter: str | None = None
    near_robot_id: str | None = None
    understood: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_scope": self.object_scope,
            "target_goal_id": self.target_goal_id,
            "object_color_filter": self.object_color_filter,
            "near_robot_id": self.near_robot_id,
            "understood": self.understood,
            "reason": self.reason,
        }


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Agent response must be a JSON object.")
    return value


class LLMClient:
    def complete_json(
        self,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        logger: JsonlLogger,
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_sec = float(os.environ.get("LLM_TIMEOUT_SEC", "45"))
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def complete_json(
        self,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        logger: JsonlLogger,
    ) -> dict[str, Any]:
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        logger.write(
            "llm_request",
            {"agent": agent_name, "model": self.model, "payload": payload},
        )
        data = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
        response_json = json.loads(raw)
        content = response_json["choices"][0]["message"]["content"]
        parsed = extract_json(content)
        logger.write("llm_response", {"agent": agent_name, "response": parsed})
        return parsed


class HeuristicClient(LLMClient):
    """Deterministic fallback for local smoke tests when no LLM key is present."""

    def complete_json(
        self,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        logger: JsonlLogger,
    ) -> dict[str, Any]:
        if agent_name == "TaskAgent":
            response = heuristic_tasks(payload["user_command"], payload["snapshot"])
        elif agent_name == "MainAgent":
            response = heuristic_main(payload)
        elif agent_name == "ActionAgent":
            response = heuristic_actions(payload["task"], payload["snapshot"])
        elif agent_name == "CriticAgent":
            response = heuristic_critic(payload)
        else:
            response = {"status": "abort", "reason": f"Unknown agent {agent_name}"}
        logger.write("llm_heuristic_response", {"agent": agent_name, "response": response})
        return response


def make_llm_client(mode: str) -> LLMClient:
    if mode == "heuristic":
        return HeuristicClient()
    if mode == "openai":
        return OpenAICompatibleClient()
    try:
        return OpenAICompatibleClient()
    except RuntimeError:
        return HeuristicClient()


class AgentBase:
    prompt_file = ""

    def __init__(self, client: LLMClient, logger: JsonlLogger) -> None:
        self.client = client
        self.logger = logger
        self.system_prompt = load_prompt(self.prompt_file)

    @property
    def name(self) -> str:
        return type(self).__name__

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.complete_json(self.name, self.system_prompt, payload, self.logger)


class TaskAgent(AgentBase):
    prompt_file = "task_agent.md"

    def create_tasks(
        self,
        user_command: str,
        snapshot: dict[str, Any],
        replan_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.call(
            {
                "user_command": user_command,
                "snapshot": compact_snapshot(snapshot),
            }
        )
        tasks = response.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError("TaskAgent response must contain tasks: []")
        return validate_tasks(tasks, snapshot, replan_context=replan_context)


class MainAgent(AgentBase):
    prompt_file = "main_agent.md"

    def decide(
        self, user_command: str, tasks: list[dict[str, Any]], snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.call(
            {
                "user_command": user_command,
                "tasks": tasks,
                "snapshot": compact_snapshot(snapshot),
            }
        )
        status = response.get("status")
        if status not in {"continue", "complete", "replan", "abort"}:
            raise ValueError(f"Invalid MainAgent status: {status}")
        return response


class ActionAgent(AgentBase):
    prompt_file = "action_agent.md"

    def plan(self, task: dict[str, Any], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.call({"task": task, "snapshot": compact_snapshot(snapshot)})
        actions = response.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError("ActionAgent response must contain actions: []")
        return validate_actions(actions, snapshot)


class CriticAgent(AgentBase):
    prompt_file = "critic_agent.md"

    def evaluate(
        self,
        task: dict[str, Any],
        plan: list[dict[str, Any]],
        results: list[dict[str, Any]],
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.call(
            {
                "task": task,
                "plan": plan,
                "results": results,
                "before_snapshot": compact_snapshot(before_snapshot),
                "after_snapshot": compact_snapshot(after_snapshot),
            }
        )
        status = response.get("status")
        if status not in {"success", "retry", "replan", "abort"}:
            raise ValueError(f"Invalid CriticAgent status: {status}")
        return response


def compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": snapshot.get("objects", []),
        "goals": snapshot.get("goals", []),
        "robots": snapshot.get("robots", {}),
        "table_center": snapshot.get("table_center"),
        "image_available": snapshot.get("image_available"),
        "image": snapshot.get("image"),
        "allowed_actions": snapshot.get("allowed_actions"),
    }


def build_command_context(
    user_command: str,
    snapshot: dict[str, Any],
    tasks: list[dict[str, Any]] | None = None,
    infeasible: dict[tuple[str, str], set[str]] | None = None,
) -> dict[str, Any]:
    parsed = symbolic_parse_command(user_command, snapshot)
    color_matching_requested = command_requests_color_matching(user_command)
    expected_objects = select_objects(parsed, snapshot) if parsed.understood else []
    goal = (
        None
        if parsed.target_goal_id is None
        else find_item(snapshot.get("goals", []), parsed.target_goal_id)
    )
    expected_pending_object_ids = []
    expected_completed_object_ids = []
    if goal is not None:
        for obj in expected_objects:
            if object_in_goal(obj, goal):
                expected_completed_object_ids.append(obj["id"])
            else:
                expected_pending_object_ids.append(obj["id"])

    task_object_ids = [
        task.get("object_id")
        for task in tasks or []
        if task.get("status") != "completed"
    ]
    task_goal_ids = {
        task.get("goal_id")
        for task in tasks or []
        if task.get("status") != "completed"
    }
    return {
        "parsed_command": parsed.to_dict(),
        "expected_goal_id": parsed.target_goal_id,
        "expected_object_ids": [obj["id"] for obj in expected_objects],
        "expected_pending_object_ids": expected_pending_object_ids,
        "expected_completed_object_ids": expected_completed_object_ids,
        "expected_pending_task_pairs": [
            {"object_id": object_id, "goal_id": parsed.target_goal_id}
            for object_id in expected_pending_object_ids
            if parsed.target_goal_id is not None
        ],
        "robot_candidates_by_pair": robot_candidates_by_pair(
            expected_objects,
            goal,
            snapshot,
            infeasible or {},
        ),
        "color_matching_requested": color_matching_requested,
        "goal_color_is_object_filter": False,
        "object_color_filter_active": parsed.object_color_filter is not None,
        "current_pending_task_object_ids": task_object_ids,
        "missing_pending_task_object_ids": [
            object_id
            for object_id in expected_pending_object_ids
            if object_id not in task_object_ids
        ],
        "unexpected_pending_task_goal_ids": sorted(
            goal_id
            for goal_id in task_goal_ids
            if parsed.target_goal_id is not None and goal_id != parsed.target_goal_id
        ),
        "interpretation_note": (
            "Goal color identifies the destination goal; object color filters only "
            "when parsed_command.object_color_filter is not null. Same-color goal "
            "matching is required only when color_matching_requested is true."
        ),
    }


def validate_tasks(
    tasks: list[dict[str, Any]],
    snapshot: dict[str, Any],
    parsed: ParsedCommand | None = None,
    replan_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    object_ids = {item["id"] for item in snapshot.get("objects", [])}
    goal_ids = {item["id"] for item in snapshot.get("goals", [])}
    robot_ids = set(snapshot.get("robots", {}))
    infeasible = parse_infeasible_robots(replan_context)
    validated = []
    for index, task in enumerate(tasks, start=1):
        object_id = task.get("object_id")
        goal_id = task.get("goal_id")
        robot_id = task.get("robot_id")
        if object_id not in object_ids:
            continue
        if goal_id not in goal_ids:
            continue
        obj = find_item(snapshot.get("objects", []), object_id)
        goal = find_item(snapshot.get("goals", []), goal_id)
        if obj and goal and object_in_goal(obj, goal):
            continue
        avoided_robots = infeasible.get((object_id, goal_id), set())
        status = task.get("status", "pending")
        if robot_id not in robot_ids or robot_id in avoided_robots:
            robot_id = assign_robot_for_task(obj, goal, snapshot, avoid=avoided_robots)
        if robot_id is None:
            status = "infeasible"
        validated.append(
            {
                "task_id": str(task.get("task_id") or f"task_{index}"),
                "object_id": object_id,
                "goal_id": goal_id,
                "robot_id": robot_id,
                "status": status,
                "reason": str(task.get("reason", "")),
            }
        )
    return validated


def task_list_violates_command_context(
    tasks: list[dict[str, Any]], command_context: dict[str, Any]
) -> bool:
    parsed = command_context.get("parsed_command") or {}
    if not parsed.get("understood"):
        return False
    expected_pairs = {
        (pair.get("object_id"), pair.get("goal_id"))
        for pair in command_context.get("expected_pending_task_pairs", [])
    }
    if not expected_pairs:
        return False
    actual_pairs = {
        (task.get("object_id"), task.get("goal_id"))
        for task in tasks
        if task.get("status") != "completed"
    }
    if expected_pairs - actual_pairs:
        return True
    expected_goal_id = command_context.get("expected_goal_id")
    expected_object_ids = set(command_context.get("expected_pending_object_ids", []))
    if expected_goal_id is None:
        return False
    return any(
        task.get("object_id") in expected_object_ids
        and task.get("goal_id") != expected_goal_id
        and task.get("status") != "completed"
        for task in tasks
    )


def repair_main_decision_if_needed(
    decision: dict[str, Any],
    command_context: dict[str, Any],
    tasks: list[dict[str, Any]],
    logger: JsonlLogger,
) -> dict[str, Any]:
    if decision.get("status") != "replan":
        return decision
    if command_context.get("color_matching_requested"):
        return decision
    if command_context.get("missing_pending_task_object_ids"):
        return decision
    if command_context.get("unexpected_pending_task_goal_ids"):
        return decision
    reason = str(decision.get("reason", "")).lower()
    if not reason_mentions_invalid_color_replan(reason):
        return decision
    task = first_pending_expected_task(tasks, command_context)
    if task is None:
        return decision
    repaired = {
        "status": "continue",
        "task_id": task["task_id"],
        "reason": (
            "Runtime repaired MainAgent color-matching replan: pending task matches "
            "the parsed command object-goal map."
        ),
    }
    logger.write(
        "main_decision_fallback",
        {
            "reason": "MainAgent requested replan for color mismatch despite valid task coverage.",
            "original_decision": decision,
            "repaired_decision": repaired,
            "command_context": command_context,
        },
    )
    return repaired


def reason_mentions_invalid_color_replan(reason: str) -> bool:
    if any(pattern in reason for pattern in COLOR_ONLY_REPLAN_PATTERNS):
        return True
    if not any(pattern in reason for pattern in INVALID_COLOR_REPLAN_PATTERNS):
        return False
    return any(goal in reason for goal in ("red goal", "blue goal", "red_goal", "blue_goal"))


def tasks_for_decision(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_fields = {
        "task_id",
        "object_id",
        "goal_id",
        "robot_id",
        "status",
        "completion_reason",
    }
    return [
        {key: value for key, value in task.items() if key in allowed_fields}
        for task in tasks
    ]


def first_pending_expected_task(
    tasks: list[dict[str, Any]], command_context: dict[str, Any]
) -> dict[str, Any] | None:
    expected_pairs = {
        (pair.get("object_id"), pair.get("goal_id"))
        for pair in command_context.get("expected_pending_task_pairs", [])
    }
    for task in tasks:
        if task.get("status", "pending") != "pending":
            continue
        if (task.get("object_id"), task.get("goal_id")) in expected_pairs:
            return task
    return None


def parse_infeasible_robots(
    replan_context: dict[str, Any] | None,
) -> dict[tuple[str, str], set[str]]:
    if not replan_context:
        return {}
    raw = replan_context.get("infeasible_robots", {})
    parsed: dict[tuple[str, str], set[str]] = {}
    if not isinstance(raw, dict):
        return parsed
    for key, robots in raw.items():
        if not isinstance(key, str) or "->" not in key:
            continue
        object_id, goal_id = key.split("->", 1)
        if isinstance(robots, list):
            parsed[(object_id, goal_id)] = {str(robot) for robot in robots}
        elif isinstance(robots, str):
            parsed[(object_id, goal_id)] = {robots}
    return parsed


def robot_candidates_by_pair(
    objects: list[dict[str, Any]],
    goal: dict[str, Any] | None,
    snapshot: dict[str, Any],
    infeasible: dict[tuple[str, str], set[str]],
) -> dict[str, Any]:
    if goal is None:
        return {}
    candidates_by_pair = {}
    for obj in objects:
        if object_in_goal(obj, goal):
            continue
        pair_key = f"{obj['id']}->{goal['id']}"
        avoided = infeasible.get((obj["id"], goal["id"]), set())
        ranked = robot_candidates_for_task(obj, goal, snapshot, avoid=avoided)
        recommended = next(
            (candidate["robot_id"] for candidate in ranked if not candidate["known_infeasible"]),
            None,
        )
        candidates_by_pair[pair_key] = {
            "recommended_robot_id": recommended,
            "candidates": ranked,
            "note": (
                "Lower score is preferred. Score combines distance from robot end-effector "
                "to object and object-to-goal travel; known_infeasible comes from repeated IK failure."
            ),
        }
    return candidates_by_pair


def robot_candidates_for_task(
    obj: dict[str, Any],
    goal: dict[str, Any],
    snapshot: dict[str, Any],
    avoid: set[str] | None = None,
) -> list[dict[str, Any]]:
    avoid = avoid or set()
    candidates = []
    for robot_id in sorted(snapshot.get("robots", {})):
        pick_distance = robot_pick_distance(robot_id, obj, snapshot)
        place_distance = object_to_goal_distance(obj, goal)
        candidates.append(
            {
                "robot_id": robot_id,
                "score": round(pick_distance + place_distance * 0.5, 4),
                "pick_distance_xy": round(pick_distance, 4),
                "object_to_goal_distance_xy": round(place_distance, 4),
                "known_infeasible": robot_id in avoid,
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (candidate["known_infeasible"], candidate["score"]),
    )


def regenerate_tasks(
    user_command: str,
    snapshot: dict[str, Any],
    previous_tasks: list[dict[str, Any]] | None = None,
    infeasible: dict[tuple[str, str], set[str]] | None = None,
    reason: str = "",
) -> list[dict[str, Any]]:
    parsed = symbolic_parse_command(user_command, snapshot)
    tasks = symbolic_tasks_from_command(
        parsed,
        snapshot,
        previous_tasks=previous_tasks,
        infeasible=infeasible,
        reason=reason,
    )
    if tasks:
        return tasks
    return validate_tasks(previous_tasks or [], snapshot, parsed)


def symbolic_parse_command(command: str, snapshot: dict[str, Any]) -> ParsedCommand:
    text = command.lower()
    text = text.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    goal = pick_goal(normalized, snapshot.get("goals", []))
    object_scope = infer_object_scope(normalized)
    near_robot_id = None
    if "left robot" in normalized:
        near_robot_id = "left"
    elif "right robot" in normalized:
        near_robot_id = "right"

    object_color_filter = None
    for color in ("red", "blue"):
        if color_applies_to_object(normalized, color):
            object_color_filter = color
            break

    understood = goal is not None and bool(snapshot.get("objects"))
    return ParsedCommand(
        object_scope=object_scope,
        target_goal_id=None if goal is None else goal["id"],
        object_color_filter=object_color_filter,
        near_robot_id=near_robot_id,
        understood=understood,
        reason="Symbolic parse separates object selection from goal color.",
    )


def command_requests_color_matching(command: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", command.lower().replace("_", " "))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(pattern in normalized for pattern in COLOR_MATCH_PATTERNS)


def infer_object_scope(normalized_command: str) -> str:
    if any(pattern in normalized_command for pattern in ALL_OBJECT_PATTERNS):
        return "all"
    tokens = normalized_command.split()
    has_objectish_plural = any(
        token in {"objects", "cubes", "capsules", "spheres"}
        or token.startswith("objec")
        or token.startswith("objet")
        for token in tokens
    )
    if has_objectish_plural and ("all" in tokens or "every" in tokens):
        return "all"
    # Imperative plural object commands usually mean the full visible set unless a
    # color/object filter narrows the set.
    if has_objectish_plural and "to" in tokens:
        return "all"
    return "single"


def color_applies_to_object(normalized_command: str, color: str) -> bool:
    for word in OBJECT_WORDS:
        if f"{color} {word}" in normalized_command:
            return True
        if f"{color}s {word}" in normalized_command:
            return True
    # "all red objects", "all blue cubes"
    return any(f"all {color} {word}" in normalized_command for word in OBJECT_WORDS)


def symbolic_tasks_from_command(
    parsed: ParsedCommand,
    snapshot: dict[str, Any],
    previous_tasks: list[dict[str, Any]] | None = None,
    infeasible: dict[tuple[str, str], set[str]] | None = None,
    reason: str = "",
) -> list[dict[str, Any]]:
    if not parsed.understood or parsed.target_goal_id is None:
        return []
    goal = find_item(snapshot.get("goals", []), parsed.target_goal_id)
    if goal is None:
        return []

    objects = select_objects(parsed, snapshot)
    previous_by_key = {
        (task.get("object_id"), task.get("goal_id")): task
        for task in previous_tasks or []
    }
    tasks = []
    for index, obj in enumerate(objects, start=1):
        if object_in_goal(obj, goal):
            continue
        existing = previous_by_key.get((obj["id"], goal["id"]), {})
        robot_id = choose_robot_for_task(
            obj,
            goal,
            snapshot,
            avoid=(infeasible or {}).get((obj["id"], goal["id"]), set()),
        )
        status = "infeasible" if robot_id is None else existing.get("status", "pending")
        tasks.append(
            {
                "task_id": str(existing.get("task_id") or f"task_{index}"),
                "object_id": obj["id"],
                "goal_id": goal["id"],
                "robot_id": robot_id,
                "status": status,
                "reason": (
                    reason
                    or f"Symbolic task: selected {obj['id']} for destination {goal['id']}."
                ),
            }
        )
    return tasks


def select_objects(parsed: ParsedCommand, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    objects = list(snapshot.get("objects", []))
    if parsed.object_color_filter is not None:
        objects = [obj for obj in objects if obj.get("color") == parsed.object_color_filter]
    if parsed.near_robot_id is not None and parsed.object_scope != "all":
        return sorted(objects, key=lambda obj: distance_to_robot(obj, parsed.near_robot_id, snapshot))[:1]
    if parsed.object_scope == "all":
        return objects
    return objects[:1]


def validate_actions(actions: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    robot_ids = set(snapshot.get("robots", {}))
    object_ids = {item["id"] for item in snapshot.get("objects", [])}
    goal_ids = {item["id"] for item in snapshot.get("goals", [])}
    allowed = set(snapshot.get("allowed_actions", []))
    validated = []
    for action in actions:
        name = action.get("action")
        robot_id = action.get("robot_id")
        if name not in allowed or robot_id not in robot_ids:
            continue
        normalized = {"action": name, "robot_id": robot_id}
        if name == ACTION_MOVING:
            if action.get("object_id") not in object_ids:
                continue
            normalized["object_id"] = action["object_id"]
        elif name == ACTION_PLACING:
            if action.get("goal_id") not in goal_ids:
                continue
            normalized["goal_id"] = action["goal_id"]
        validated.append(normalized)
    return validated


def heuristic_tasks(command: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    parsed = symbolic_parse_command(command, snapshot)
    symbolic_tasks = symbolic_tasks_from_command(parsed, snapshot)
    if symbolic_tasks:
        return {"tasks": symbolic_tasks, "reason": "Symbolic task generation."}
    command_lower = command.lower()
    objects = snapshot.get("objects", [])
    goals = snapshot.get("goals", [])
    goal = pick_goal(command_lower, goals)
    if goal is None:
        return {"tasks": [], "reason": "No goal matched the command and snapshot."}
    selected_objects = pick_objects(command_lower, objects, snapshot)
    tasks = []
    for index, obj in enumerate(selected_objects, start=1):
        if object_in_goal(obj, goal):
            continue
        robot_id = assign_robot_for_task(obj, goal, snapshot)
        status = "infeasible" if robot_id is None else "pending"
        tasks.append(
            {
                "task_id": f"task_{index}",
                "object_id": obj["id"],
                "goal_id": goal["id"],
                "robot_id": robot_id,
                "status": status,
                "reason": "Heuristic task from command and snapshot IDs.",
            }
        )
    return {"tasks": tasks}


def heuristic_main(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = payload.get("tasks", [])
    infeasible = [task for task in tasks if task.get("status") == "infeasible"]
    if infeasible:
        return {
            "status": "abort",
            "reason": "At least one requested task has no feasible robot assignment.",
        }
    pending = [task for task in tasks if task.get("status", "pending") == "pending"]
    if not pending:
        return {"status": "complete", "reason": "All tasks are completed."}
    return {
        "status": "continue",
        "task_id": pending[0]["task_id"],
        "reason": "Execute the next pending object task.",
    }


def heuristic_actions(task: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    robot_id = task["robot_id"]
    return {
        "actions": [
            {"action": ACTION_MOVING, "robot_id": robot_id, "object_id": task["object_id"]},
            {"action": ACTION_GRIP, "robot_id": robot_id},
            {"action": ACTION_PLACING, "robot_id": robot_id, "goal_id": task["goal_id"]},
            {"action": ACTION_RELEASE, "robot_id": robot_id},
            {"action": ACTION_HOMING, "robot_id": robot_id},
        ],
        "reason": "Standard pick-place sequence using world.py primitives.",
    }


def heuristic_critic(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    failed_results = [result for result in results if not result.get("ok")]
    if any("IK could not solve" in str(result.get("message", "")) for result in failed_results):
        return {
            "status": "replan",
            "reason": "An action failed because IK could not solve the target; try a different robot assignment or plan.",
        }
    if failed_results:
        return {"status": "retry", "reason": "At least one action failed."}
    task = payload["task"]
    after = payload["after_snapshot"]
    obj = find_item(after.get("objects", []), task["object_id"])
    goal = find_item(after.get("goals", []), task["goal_id"])
    if obj and goal and object_in_goal(obj, goal):
        return {"status": "success", "reason": "Object marker is inside the goal zone."}
    return {
        "status": "retry",
        "reason": "Actions succeeded but snapshot does not yet verify task completion.",
    }


def pick_goal(command_lower: str, goals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if "red goal" in command_lower or "red_goal" in command_lower:
        return find_item(goals, "red_goal")
    if "blue goal" in command_lower or "blue_goal" in command_lower:
        return find_item(goals, "blue_goal")
    return goals[0] if goals else None


def pick_objects(
    command_lower: str, objects: list[dict[str, Any]], snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    if "all object" in command_lower or "all objects" in command_lower:
        return objects
    if "red object" in command_lower:
        red = [obj for obj in objects if obj.get("color") == "red"]
        return red[:1]
    if "blue object" in command_lower:
        blue = [obj for obj in objects if obj.get("color") == "blue"]
        return blue[:1]
    if "left robot" in command_lower:
        return sorted(objects, key=lambda obj: distance_to_robot(obj, "left", snapshot))[:1]
    if "right robot" in command_lower:
        return sorted(objects, key=lambda obj: distance_to_robot(obj, "right", snapshot))[:1]
    return objects[:1]


def distance_to_robot(obj: dict[str, Any], robot_id: str, snapshot: dict[str, Any]) -> float:
    robot = snapshot.get("robots", {}).get(robot_id, {})
    pose = robot.get("pose")
    if pose is None:
        return 999.0
    op = obj["pose"]["position"]
    rp = pose["position"]
    return ((op["x"] - rp["x"]) ** 2 + (op["y"] - rp["y"]) ** 2) ** 0.5


def choose_robot_for_task(
    obj: dict[str, Any],
    goal: dict[str, Any],
    snapshot: dict[str, Any],
    avoid: set[str] | None = None,
) -> str | None:
    robot_ids = sorted(snapshot.get("robots", {}))
    avoid = avoid or set()
    candidates = [robot_id for robot_id in robot_ids if robot_id not in avoid]
    if not candidates:
        return None
    return min(candidates, key=lambda robot_id: task_travel_score(robot_id, obj, goal, snapshot))


def assign_robot_for_task(
    obj: dict[str, Any] | None,
    goal: dict[str, Any] | None,
    snapshot: dict[str, Any],
    avoid: set[str] | None = None,
) -> str | None:
    if obj is not None and goal is not None:
        return choose_robot_for_task(obj, goal, snapshot, avoid=avoid)
    avoid = avoid or set()
    if goal is not None and goal.get("robot_id"):
        if goal["robot_id"] in avoid:
            return None
        return goal["robot_id"]
    return next((robot_id for robot_id in sorted(snapshot.get("robots", {})) if robot_id not in avoid), None)


def task_travel_score(
    robot_id: str, obj: dict[str, Any], goal: dict[str, Any], snapshot: dict[str, Any]
) -> float:
    return robot_pick_distance(robot_id, obj, snapshot) + object_to_goal_distance(obj, goal) * 0.5


def robot_pick_distance(robot_id: str, obj: dict[str, Any], snapshot: dict[str, Any]) -> float:
    robot = snapshot.get("robots", {}).get(robot_id, {})
    eef_pose = robot.get("end_effector_pose") or robot.get("pose")
    if eef_pose is None:
        return 100.0
    return xy_distance(eef_pose["position"], obj["pose"]["position"])


def object_to_goal_distance(obj: dict[str, Any], goal: dict[str, Any]) -> float:
    goal_position = (goal.get("service_target_pose") or goal["pose"])["position"]
    return xy_distance(obj["pose"]["position"], goal_position)


def xy_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return ((left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2) ** 0.5


def find_item(items: list[dict[str, Any]], item_id: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("id") == item_id), None)


def goal_robot(goal_id: str, snapshot: dict[str, Any]) -> str:
    goal = find_item(snapshot.get("goals", []), goal_id)
    if goal and goal.get("robot_id"):
        return goal["robot_id"]
    return "left" if goal_id == "red_goal" else "right"


def object_in_goal(obj: dict[str, Any], goal: dict[str, Any]) -> bool:
    op = obj["pose"]["position"]
    gp = goal["pose"]["position"]
    scale = goal.get("scale") or {"x": 0.2, "y": 0.2, "z": 0.05}
    half_x = max(float(scale.get("x", 0.2)) / 2.0, 0.1)
    half_y = max(float(scale.get("y", 0.2)) / 2.0, 0.1)
    return abs(op["x"] - gp["x"]) <= half_x and abs(op["y"] - gp["y"]) <= half_y
