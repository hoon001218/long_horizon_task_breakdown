"""Run logging and lightweight task memory for the ROS2 LLM controller."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class JsonlLogger:
    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"run_{utc_timestamp()}.jsonl"

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            "payload": json_safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class TaskMemory:
    def __init__(self, logger: JsonlLogger) -> None:
        self.logger = logger
        self.tasks: list[dict[str, Any]] = []
        self.failures: dict[str, int] = {}

    def set_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = tasks
        self.logger.write("tasks_created", {"tasks": tasks})

    def pending_tasks(self) -> list[dict[str, Any]]:
        return [task for task in self.tasks if task.get("status", "pending") == "pending"]

    def mark_completed(self, task_id: str, reason: str) -> None:
        for task in self.tasks:
            if task.get("task_id") == task_id:
                task["status"] = "completed"
                task["completion_reason"] = reason
                self.logger.write("task_completed", {"task": task})
                return

    def mark_pending(self, task_id: str, reason: str) -> None:
        for task in self.tasks:
            if task.get("task_id") == task_id:
                task["status"] = "pending"
                task["reopen_reason"] = reason
                task.pop("completion_reason", None)
                self.logger.write("task_reopened", {"task": task})
                return

    def record_failure(self, task_id: str, reason: str) -> int:
        count = self.failures.get(task_id, 0) + 1
        self.failures[task_id] = count
        self.logger.write(
            "task_failure",
            {"task_id": task_id, "failure_count": count, "reason": reason},
        )
        return count

    def reset_failures(self, task_id: str) -> None:
        if task_id in self.failures:
            self.logger.write(
                "task_failures_reset",
                {"task_id": task_id, "previous_count": self.failures[task_id]},
            )
            del self.failures[task_id]
