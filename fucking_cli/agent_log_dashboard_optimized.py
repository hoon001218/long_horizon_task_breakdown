#!/usr/bin/env python3
"""Optimized Streamlit dashboard for run_*.jsonl robot agent logs.

Run:
    streamlit run agent_log_dashboard_optimized.py

or:
    python3 agent_log_dashboard_optimized.py

Default log directory:
    ./logs

This dashboard is read-only. It is tuned for the current log shape where each
JSONL file under ./logs represents one run and contains mixed events from
TaskAgent, MainAgent, ActionAgent, CriticAgent, and the executor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional in small environments
    px = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", APP_DIR / "logs"))
DEFAULT_PATTERN = os.getenv("AGENT_LOG_PATTERN", "run_*.jsonl")

EVENT_COLUMNS = [
    "event_id",
    "run_id",
    "file",
    "line_no",
    "timestamp",
    "time_text",
    "elapsed_sec",
    "event",
    "agent",
    "iteration",
    "task_id",
    "object_id",
    "robot_id",
    "action",
    "status",
    "ok",
    "duration_sec",
    "summary",
    "severity",
    "_payload",
    "_raw",
]

RUN_COLUMNS = [
    "run_id",
    "file",
    "command",
    "planner_mode",
    "start_time",
    "end_time",
    "start_text",
    "duration_sec",
    "events",
    "llm_calls",
    "objects",
    "tasks_created",
    "task_completed",
    "task_failures",
    "critic_success",
    "critic_replan",
    "replans",
    "unchanged_replans",
    "action_results",
    "action_failures",
    "aborts",
    "final_state",
    "health_score",
    "first_issue",
]

TASK_COLUMNS = [
    "run_id",
    "event_id",
    "event",
    "iteration",
    "task_id",
    "object_id",
    "goal_id",
    "robot_id",
    "status",
    "reason",
]

ACTION_COLUMNS = [
    "run_id",
    "event_id",
    "event",
    "iteration",
    "task_id",
    "step_index",
    "robot_id",
    "action",
    "object_id",
    "goal_id",
    "ok",
    "duration_sec",
    "message",
]

ISSUE_COLUMNS = [
    "run_id",
    "event_id",
    "timestamp",
    "severity",
    "event",
    "agent",
    "task_id",
    "object_id",
    "robot_id",
    "action",
    "reason",
]


@dataclass(frozen=True)
class LogRecord:
    event_id: int
    run_id: str
    path: str
    line_no: int
    timestamp: float | None
    event: str
    payload: dict[str, Any]
    raw: dict[str, Any]
    parse_error: str | None = None


def safe_json_dumps(value: Any, max_chars: int = 700) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = str(value)
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def get_in(value: Any, path: Iterable[Any], default: Any = None) -> Any:
    current = value
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def time_text(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def short_path(path: str) -> str:
    return Path(path).name


def load_records(log_dir: Path, pattern: str) -> list[LogRecord]:
    files = sorted(log_dir.glob(pattern))
    records: list[LogRecord] = []
    event_id = 0
    for path in files:
        if not path.is_file():
            continue
        run_id = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                parse_error = None
                try:
                    raw = json.loads(text)
                except Exception as exc:
                    raw = {"event": "parse_error", "payload": {"line": text}, "timestamp": None}
                    parse_error = str(exc)
                payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
                records.append(
                    LogRecord(
                        event_id=event_id,
                        run_id=run_id,
                        path=str(path),
                        line_no=line_no,
                        timestamp=as_float(raw.get("timestamp") or raw.get("time")),
                        event=str(raw.get("event", "unknown")),
                        payload=payload,
                        raw=raw,
                        parse_error=parse_error,
                    )
                )
                event_id += 1
    return records


def infer_agent(event: str, payload: dict[str, Any]) -> str:
    agent = payload.get("agent")
    if isinstance(agent, str) and agent:
        return agent
    if event == "tasks_created" or event == "tasks_replanned":
        return "TaskAgent"
    if event == "main_decision":
        return "MainAgent"
    if event in {"action_plan", "action_result", "pre_grip_delay"}:
        return "ActionAgent"
    if event in {"critic_decision", "critic_success_unverified"}:
        return "CriticAgent"
    if event in {"run_started", "run_finished", "initial_snapshot"}:
        return "Runner"
    return "System"


def task_from_payload(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event == "action_plan":
        return as_dict(payload.get("task"))
    if event == "task_completed":
        return as_dict(payload.get("task"))
    if event == "critic_success_unverified":
        return as_dict(payload.get("task"))
    task = payload.get("task")
    return as_dict(task)


def action_from_payload(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event == "action_result":
        return as_dict(get_in(payload, ["result", "action"]))
    action = payload.get("action")
    return as_dict(action)


def status_from_payload(event: str, payload: dict[str, Any]) -> str:
    if event == "main_decision":
        return str(get_in(payload, ["decision", "status"], ""))
    if event == "critic_decision":
        return str(get_in(payload, ["critic", "status"], ""))
    if event == "llm_response":
        response = as_dict(payload.get("response"))
        status = response.get("status")
        if status:
            return str(status)
    if event == "task_completed":
        return "completed"
    if event == "task_failure":
        return "failed"
    if event == "tasks_replanned":
        return "replanned"
    if event == "replan_unchanged":
        return "unchanged"
    if event == "abort":
        return "aborted"
    if event == "action_result":
        ok = get_in(payload, ["result", "ok"])
        if ok is True:
            return "ok"
        if ok is False:
            return "failed"
    return ""


def severity_for(event: str, status: str, ok: Any) -> str:
    if event == "abort":
        return "critical"
    if event in {"task_failure", "critic_success_unverified", "robot_marked_infeasible"}:
        return "error"
    if event == "action_result" and ok is False:
        return "error"
    if event in {"tasks_replanned", "replan_unchanged"}:
        return "warning"
    if status in {"retry", "replan", "replanned", "unchanged"}:
        return "warning"
    if status in {"success", "complete", "completed", "ok", "continue"}:
        return "good"
    return "neutral"


def event_summary(event: str, payload: dict[str, Any]) -> str:
    if event == "run_started":
        return str(payload.get("command", "run started"))
    if event == "initial_snapshot":
        objects = as_list(get_in(payload, ["snapshot", "objects"]))
        image = get_in(payload, ["snapshot", "image_available"])
        return f"snapshot objects={len(objects)} image={image}"
    if event == "llm_request":
        agent = payload.get("agent", "agent")
        model = payload.get("model", "")
        return f"{agent} request {model}".strip()
    if event == "llm_response":
        agent = payload.get("agent", "agent")
        response = as_dict(payload.get("response"))
        if "tasks" in response:
            return f"{agent} returned {len(as_list(response.get('tasks')))} tasks"
        if "actions" in response:
            return f"{agent} returned {len(as_list(response.get('actions')))} actions"
        status = response.get("status")
        reason = str(response.get("reason", ""))[:140]
        return f"{agent} status={status or 'response'} {reason}".strip()
    if event == "tasks_created":
        return f"created {len(as_list(payload.get('tasks')))} tasks"
    if event == "tasks_replanned":
        reason = str(payload.get("reason", ""))[:160]
        return f"replanned {len(as_list(payload.get('tasks')))} tasks: {reason}"
    if event == "main_decision":
        decision = as_dict(payload.get("decision"))
        return f"{decision.get('status', '')} task={decision.get('task_id', '')}: {str(decision.get('reason', ''))[:160]}"
    if event == "action_plan":
        task = as_dict(payload.get("task"))
        return f"iteration {payload.get('iteration')} plan {len(as_list(payload.get('plan')))} steps for {task.get('task_id', '')}"
    if event == "action_result":
        result = as_dict(payload.get("result"))
        action = as_dict(result.get("action"))
        return f"{action.get('robot_id', '')} {action.get('action', '')} ok={result.get('ok')} {str(result.get('message', ''))[:160]}"
    if event == "critic_decision":
        critic = as_dict(payload.get("critic"))
        return f"{critic.get('status', '')}: {str(critic.get('reason', ''))[:180]}"
    if event == "task_completed":
        task = as_dict(payload.get("task"))
        return f"completed {task.get('task_id', '')} {task.get('object_id', '')}"
    if event == "task_failure":
        return f"failure {payload.get('task_id', '')}: {str(payload.get('reason', ''))[:180]}"
    if event == "critic_success_unverified":
        return f"critic success unverified: {str(payload.get('reason', ''))[:180]}"
    if event == "replan_unchanged":
        return f"unchanged replan: {str(payload.get('reason', ''))[:180]}"
    if event == "abort":
        return f"abort: {payload.get('reason', '')}"
    if event == "robot_marked_infeasible":
        return f"infeasible {payload.get('robot_id', '')} {payload.get('action', '')} {payload.get('object_id', '')}"
    if event == "run_finished":
        return "run finished"
    return safe_json_dumps(payload, max_chars=220)


def build_event_frame(records: list[LogRecord]) -> pd.DataFrame:
    first_by_run: dict[str, float | None] = {}
    for record in records:
        if record.timestamp is not None and record.run_id not in first_by_run:
            first_by_run[record.run_id] = record.timestamp

    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record.payload
        task = task_from_payload(record.event, payload)
        action = action_from_payload(record.event, payload)
        status = status_from_payload(record.event, payload)
        ok = get_in(payload, ["result", "ok"]) if record.event == "action_result" else ""
        severity = severity_for(record.event, status, ok)
        first_ts = first_by_run.get(record.run_id)
        elapsed = record.timestamp - first_ts if record.timestamp is not None and first_ts is not None else None
        rows.append(
            {
                "event_id": record.event_id,
                "run_id": record.run_id,
                "file": short_path(record.path),
                "line_no": record.line_no,
                "timestamp": record.timestamp,
                "time_text": time_text(record.timestamp),
                "elapsed_sec": elapsed,
                "event": record.event,
                "agent": infer_agent(record.event, payload),
                "iteration": payload.get("iteration", ""),
                "task_id": task.get("task_id") or payload.get("task_id") or get_in(payload, ["decision", "task_id"], ""),
                "object_id": task.get("object_id") or action.get("object_id") or payload.get("object_id", ""),
                "robot_id": task.get("robot_id") or action.get("robot_id") or payload.get("robot_id", ""),
                "action": action.get("action") or payload.get("action", ""),
                "status": status,
                "ok": ok,
                "duration_sec": as_float(get_in(payload, ["result", "elapsed_sec"])),
                "summary": event_summary(record.event, payload),
                "severity": severity,
                "_payload": payload,
                "_raw": record.raw,
            }
        )
    df = pd.DataFrame(rows)
    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[EVENT_COLUMNS]


def build_task_frame(records: list[LogRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record.payload
        if record.event in {"tasks_created", "tasks_replanned", "replan_unchanged", "abort"}:
            for task in as_list(payload.get("tasks")):
                task_dict = as_dict(task)
                rows.append(
                    {
                        "run_id": record.run_id,
                        "event_id": record.event_id,
                        "event": record.event,
                        "iteration": payload.get("iteration", ""),
                        "task_id": task_dict.get("task_id", ""),
                        "object_id": task_dict.get("object_id", ""),
                        "goal_id": task_dict.get("goal_id", ""),
                        "robot_id": task_dict.get("robot_id", ""),
                        "status": task_dict.get("status", ""),
                        "reason": task_dict.get("reason", payload.get("reason", "")),
                    }
                )
        elif record.event == "task_completed":
            task_dict = as_dict(payload.get("task"))
            rows.append(
                {
                    "run_id": record.run_id,
                    "event_id": record.event_id,
                    "event": record.event,
                    "iteration": payload.get("iteration", ""),
                    "task_id": task_dict.get("task_id", ""),
                    "object_id": task_dict.get("object_id", ""),
                    "goal_id": task_dict.get("goal_id", ""),
                    "robot_id": task_dict.get("robot_id", ""),
                    "status": task_dict.get("status", "completed"),
                    "reason": task_dict.get("completion_reason", task_dict.get("reason", "")),
                }
            )
        elif record.event == "task_failure":
            rows.append(
                {
                    "run_id": record.run_id,
                    "event_id": record.event_id,
                    "event": record.event,
                    "iteration": payload.get("iteration", ""),
                    "task_id": payload.get("task_id", ""),
                    "object_id": "",
                    "goal_id": "",
                    "robot_id": "",
                    "status": "failed",
                    "reason": payload.get("reason", ""),
                }
            )
    df = pd.DataFrame(rows)
    for column in TASK_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[TASK_COLUMNS]


def build_action_frame(records: list[LogRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_context: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.payload
        if record.event == "action_plan":
            task = as_dict(payload.get("task"))
            active_context[record.run_id] = {
                "iteration": payload.get("iteration", ""),
                "task_id": task.get("task_id", ""),
            }
            for index, action in enumerate(as_list(payload.get("plan"))):
                action_dict = as_dict(action)
                rows.append(
                    {
                        "run_id": record.run_id,
                        "event_id": record.event_id,
                        "event": "planned",
                        "iteration": payload.get("iteration", ""),
                        "task_id": task.get("task_id", ""),
                        "step_index": index,
                        "robot_id": action_dict.get("robot_id", ""),
                        "action": action_dict.get("action", ""),
                        "object_id": action_dict.get("object_id", ""),
                        "goal_id": action_dict.get("goal_id", ""),
                        "ok": "",
                        "duration_sec": "",
                        "message": "",
                    }
                )
        elif record.event == "action_result":
            result = as_dict(payload.get("result"))
            action = as_dict(result.get("action"))
            context = active_context.get(record.run_id, {})
            rows.append(
                {
                    "run_id": record.run_id,
                    "event_id": record.event_id,
                    "event": "executed",
                    "iteration": context.get("iteration", ""),
                    "task_id": context.get("task_id", ""),
                    "step_index": payload.get("index", ""),
                    "robot_id": action.get("robot_id", ""),
                    "action": action.get("action", ""),
                    "object_id": action.get("object_id", ""),
                    "goal_id": action.get("goal_id", ""),
                    "ok": result.get("ok", ""),
                    "duration_sec": result.get("elapsed_sec", ""),
                    "message": result.get("message", ""),
                }
            )
    df = pd.DataFrame(rows)
    for column in ACTION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[ACTION_COLUMNS]


def issue_reason(record: LogRecord) -> str:
    payload = record.payload
    if record.event == "action_result":
        return str(get_in(payload, ["result", "message"], ""))
    if record.event == "critic_decision":
        return str(get_in(payload, ["critic", "reason"], ""))
    return str(payload.get("reason") or payload.get("message") or event_summary(record.event, payload))


def build_issue_frame(records: list[LogRecord], event_df: pd.DataFrame) -> pd.DataFrame:
    by_id = event_df.set_index("event_id", drop=False).to_dict("index") if not event_df.empty else {}
    rows: list[dict[str, Any]] = []
    for record in records:
        event_row = by_id.get(record.event_id, {})
        ok = get_in(record.payload, ["result", "ok"]) if record.event == "action_result" else ""
        status = str(event_row.get("status", ""))
        severity = severity_for(record.event, status, ok)
        include = severity in {"critical", "error"}
        include = include or record.event in {"tasks_replanned", "replan_unchanged"}
        include = include or (record.event == "critic_decision" and status in {"retry", "replan"})
        if not include:
            continue
        rows.append(
            {
                "run_id": record.run_id,
                "event_id": record.event_id,
                "timestamp": time_text(record.timestamp),
                "severity": severity,
                "event": record.event,
                "agent": event_row.get("agent", infer_agent(record.event, record.payload)),
                "task_id": event_row.get("task_id", ""),
                "object_id": event_row.get("object_id", ""),
                "robot_id": event_row.get("robot_id", ""),
                "action": event_row.get("action", ""),
                "reason": issue_reason(record),
            }
        )
    df = pd.DataFrame(rows)
    for column in ISSUE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[ISSUE_COLUMNS]


def initial_object_count(records: list[LogRecord], run_id: str) -> int:
    for record in records:
        if record.run_id == run_id and record.event == "initial_snapshot":
            return len(as_list(get_in(record.payload, ["snapshot", "objects"])))
    return 0


def first_payload_value(records: list[LogRecord], run_id: str, event: str, key: str) -> Any:
    for record in records:
        if record.run_id == run_id and record.event == event:
            return record.payload.get(key)
    return ""


def final_state_for(group: pd.DataFrame, object_count: int) -> str:
    events = Counter(group["event"].tolist())
    statuses = set(str(x) for x in group["status"].tolist())
    completed = int(events.get("task_completed", 0))
    if events.get("abort", 0):
        return "aborted"
    if "complete" in statuses or (object_count > 0 and completed >= object_count):
        return "complete"
    if events.get("run_finished", 0):
        return "finished_partial"
    return "incomplete"


def health_score(row: dict[str, Any]) -> int:
    score = 100
    score -= int(row.get("aborts", 0)) * 40
    score -= int(row.get("task_failures", 0)) * 14
    score -= int(row.get("action_failures", 0)) * 10
    score -= int(row.get("unchanged_replans", 0)) * 10
    score -= int(row.get("replans", 0)) * 5
    if row.get("final_state") not in {"complete"}:
        score -= 15
    return max(0, min(100, score))


def build_run_frame(records: list[LogRecord], event_df: pd.DataFrame, issue_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if event_df.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)

    for run_id, group in event_df.groupby("run_id", sort=True):
        timestamps = [x for x in group["timestamp"].tolist() if pd.notna(x) and x != ""]
        start = min(timestamps) if timestamps else None
        end = max(timestamps) if timestamps else None
        duration = end - start if start is not None and end is not None else None
        events = Counter(group["event"].tolist())
        statuses = Counter(str(x) for x in group["status"].tolist() if x)
        objects = initial_object_count(records, run_id)
        run_issues = issue_df[issue_df["run_id"] == run_id] if not issue_df.empty else pd.DataFrame(columns=ISSUE_COLUMNS)
        first_issue = ""
        if not run_issues.empty:
            first_issue = str(run_issues.iloc[0]["reason"])[:160]
        row = {
            "run_id": run_id,
            "file": group["file"].iloc[0],
            "command": first_payload_value(records, run_id, "run_started", "command"),
            "planner_mode": first_payload_value(records, run_id, "run_started", "planner_mode"),
            "start_time": start,
            "end_time": end,
            "start_text": time_text(start),
            "duration_sec": duration,
            "events": len(group),
            "llm_calls": int(events.get("llm_request", 0) + events.get("llm_response", 0)),
            "objects": objects,
            "tasks_created": int(events.get("tasks_created", 0)),
            "task_completed": int(events.get("task_completed", 0)),
            "task_failures": int(events.get("task_failure", 0) + events.get("critic_success_unverified", 0)),
            "critic_success": int(statuses.get("success", 0)),
            "critic_replan": int(statuses.get("replan", 0)),
            "replans": int(events.get("tasks_replanned", 0)),
            "unchanged_replans": int(events.get("replan_unchanged", 0)),
            "action_results": int(events.get("action_result", 0)),
            "action_failures": int(((group["event"] == "action_result") & (group["ok"] == False)).sum()),  # noqa: E712
            "aborts": int(events.get("abort", 0)),
            "final_state": final_state_for(group, objects),
            "health_score": 0,
            "first_issue": first_issue,
        }
        row["health_score"] = health_score(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    for column in RUN_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[RUN_COLUMNS].sort_values("start_time", na_position="last")


def final_task_state(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame(columns=["run_id", "task_id", "object_id", "goal_id", "robot_id", "latest_status", "last_event", "last_reason"])
    rows: list[dict[str, Any]] = []
    ordered = task_df.sort_values("event_id")
    for (run_id, task_id), group in ordered.groupby(["run_id", "task_id"], dropna=False, sort=True):
        last = group.iloc[-1]
        object_id = next((x for x in group["object_id"].tolist() if x), "")
        goal_id = next((x for x in group["goal_id"].tolist() if x), "")
        robot_id = next((x for x in group["robot_id"].tolist() if x), "")
        rows.append(
            {
                "run_id": run_id,
                "task_id": task_id,
                "object_id": object_id,
                "goal_id": goal_id,
                "robot_id": robot_id,
                "latest_status": last.get("status", ""),
                "last_event": last.get("event", ""),
                "last_reason": last.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def records_by_id(records: list[LogRecord]) -> dict[int, LogRecord]:
    return {record.event_id: record for record in records}


def apply_filters(
    run_df: pd.DataFrame,
    event_df: pd.DataFrame,
    task_df: pd.DataFrame,
    action_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    selected_runs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not selected_runs:
        return run_df.iloc[0:0], event_df.iloc[0:0], task_df.iloc[0:0], action_df.iloc[0:0], issue_df.iloc[0:0]
    return (
        run_df[run_df["run_id"].isin(selected_runs)],
        event_df[event_df["run_id"].isin(selected_runs)],
        task_df[task_df["run_id"].isin(selected_runs)],
        action_df[action_df["run_id"].isin(selected_runs)],
        issue_df[issue_df["run_id"].isin(selected_runs)],
    )


def render_metric_grid(run_df: pd.DataFrame, event_df: pd.DataFrame, issue_df: pd.DataFrame) -> None:
    cols = st.columns(6)
    complete_runs = int((run_df["final_state"] == "complete").sum()) if not run_df.empty else 0
    success_rate = (complete_runs / len(run_df) * 100.0) if len(run_df) else 0.0
    cols[0].metric("Runs", len(run_df))
    cols[1].metric("Complete", f"{success_rate:.0f}%")
    cols[2].metric("Tasks done", int(run_df["task_completed"].sum()) if not run_df.empty else 0)
    cols[3].metric("Action failures", int(run_df["action_failures"].sum()) if not run_df.empty else 0)
    cols[4].metric("Replans", int(run_df["replans"].sum()) if not run_df.empty else 0)
    cols[5].metric("Issues", len(issue_df))


def render_plotly_or_table(fig: Any, fallback: pd.DataFrame) -> None:
    if px is None:
        st.dataframe(fallback, use_container_width=True, hide_index=True)
    else:
        st.plotly_chart(fig, use_container_width=True)


def render_overview(run_df: pd.DataFrame, event_df: pd.DataFrame, issue_df: pd.DataFrame) -> None:
    render_metric_grid(run_df, event_df, issue_df)

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Run Health")
        table_cols = [
            "run_id",
            "final_state",
            "health_score",
            "duration_sec",
            "objects",
            "task_completed",
            "action_failures",
            "replans",
            "aborts",
            "first_issue",
        ]
        st.dataframe(
            run_df[table_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "health_score": st.column_config.ProgressColumn("health", min_value=0, max_value=100),
                "duration_sec": st.column_config.NumberColumn("duration", format="%.1f s"),
            },
        )

    with right:
        st.subheader("Outcome Mix")
        outcome = run_df["final_state"].value_counts().reset_index()
        outcome.columns = ["final_state", "runs"]
        if px is not None and not outcome.empty:
            fig = px.pie(outcome, values="runs", names="final_state", hole=0.45)
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(outcome, use_container_width=True, hide_index=True)

    st.subheader("Duration vs Reliability")
    if px is not None and not run_df.empty:
        fig = px.scatter(
            run_df,
            x="duration_sec",
            y="health_score",
            color="final_state",
            size="events",
            hover_data=["run_id", "task_completed", "action_failures", "replans", "first_issue"],
        )
        fig.update_layout(height=420, xaxis_title="Duration (sec)", yaxis_title="Health score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Event Volume")
    if not event_df.empty:
        counts = event_df.groupby(["run_id", "event"]).size().reset_index(name="count")
        if px is not None:
            fig = px.bar(counts, x="run_id", y="count", color="event", title=None)
            fig.update_layout(height=420, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(counts, use_container_width=True, hide_index=True)


def render_run_deep_dive(
    run_df: pd.DataFrame,
    event_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    all_run_ids: list[str],
) -> None:
    if not all_run_ids:
        st.info("No run selected.")
        return
    default_index = max(0, len(all_run_ids) - 1)
    run_id = st.selectbox("Run", all_run_ids, index=default_index)
    run_row = run_df[run_df["run_id"] == run_id]
    events = event_df[event_df["run_id"] == run_id].sort_values("event_id")
    issues = issue_df[issue_df["run_id"] == run_id]

    if run_row.empty:
        st.info("The selected run is outside the current filter.")
        return
    row = run_row.iloc[0]
    cols = st.columns(5)
    cols[0].metric("State", row["final_state"])
    cols[1].metric("Health", int(row["health_score"]))
    cols[2].metric("Duration", f"{float(row['duration_sec'] or 0):.1f}s")
    cols[3].metric("Completed", f"{int(row['task_completed'])}/{int(row['objects'])}")
    cols[4].metric("Issues", len(issues))
    st.caption(str(row["command"]))

    st.subheader("Timeline")
    if px is not None and not events.empty:
        fig = px.scatter(
            events,
            x="elapsed_sec",
            y="agent",
            color="severity",
            symbol="event",
            hover_data=["event_id", "event", "status", "task_id", "robot_id", "action", "summary"],
        )
        fig.update_traces(marker={"size": 11})
        fig.update_layout(height=470, xaxis_title="Elapsed seconds")
        st.plotly_chart(fig, use_container_width=True)

    if not issues.empty:
        st.subheader("Risk Rail")
        st.dataframe(issues, use_container_width=True, hide_index=True)

    st.subheader("Event Sequence")
    display_cols = ["event_id", "elapsed_sec", "agent", "event", "severity", "status", "task_id", "robot_id", "action", "summary"]
    st.dataframe(
        events[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={"elapsed_sec": st.column_config.NumberColumn("elapsed", format="%.2f s")},
    )


def render_tasks(task_df: pd.DataFrame) -> None:
    final_df = final_task_state(task_df)
    st.subheader("Latest Task State")
    st.dataframe(final_df, use_container_width=True, hide_index=True)

    if not task_df.empty:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Task Events")
            event_counts = task_df.groupby(["event", "status"]).size().reset_index(name="count")
            st.dataframe(event_counts, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Objects")
            obj_counts = task_df[task_df["object_id"] != ""].groupby(["object_id", "status"]).size().reset_index(name="count")
            if px is not None and not obj_counts.empty:
                fig = px.bar(obj_counts, x="object_id", y="count", color="status")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(obj_counts, use_container_width=True, hide_index=True)

    st.subheader("Task Log")
    st.dataframe(task_df.sort_values(["run_id", "event_id"]), use_container_width=True, hide_index=True)


def render_actions(action_df: pd.DataFrame) -> None:
    executed = action_df[action_df["event"] == "executed"].copy()
    planned = action_df[action_df["event"] == "planned"].copy()
    cols = st.columns(5)
    cols[0].metric("Planned", len(planned))
    cols[1].metric("Executed", len(executed))
    cols[2].metric("Failed", int((executed["ok"] == False).sum()) if not executed.empty else 0)  # noqa: E712
    cols[3].metric("Robots", executed["robot_id"].nunique() if not executed.empty else 0)
    cols[4].metric("Action types", executed["action"].nunique() if not executed.empty else 0)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Executed Actions")
        counts = executed.groupby(["action", "ok"]).size().reset_index(name="count") if not executed.empty else pd.DataFrame()
        if px is not None and not counts.empty:
            fig = px.bar(counts, x="action", y="count", color="ok")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(counts, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Executor Time")
        duration_df = executed.copy()
        duration_df["duration_sec"] = pd.to_numeric(duration_df["duration_sec"], errors="coerce")
        duration_df = duration_df.dropna(subset=["duration_sec"])
        if px is not None and not duration_df.empty:
            fig = px.box(duration_df, x="action", y="duration_sec", color="robot_id", points="all")
            fig.update_layout(height=380, yaxis_title="Seconds")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(duration_df, use_container_width=True, hide_index=True)

    failures = executed[executed["ok"] == False] if not executed.empty else pd.DataFrame(columns=ACTION_COLUMNS)  # noqa: E712
    if not failures.empty:
        st.subheader("Failed Action Results")
        st.dataframe(failures, use_container_width=True, hide_index=True)

    st.subheader("Action Log")
    st.dataframe(action_df.sort_values(["run_id", "event_id", "step_index"]), use_container_width=True, hide_index=True)


def render_issues(issue_df: pd.DataFrame) -> None:
    if issue_df.empty:
        st.success("No warnings or failures in the selected runs.")
        return
    st.subheader("Issue Summary")
    cols = st.columns(4)
    cols[0].metric("Critical", int((issue_df["severity"] == "critical").sum()))
    cols[1].metric("Errors", int((issue_df["severity"] == "error").sum()))
    cols[2].metric("Warnings", int((issue_df["severity"] == "warning").sum()))
    cols[3].metric("Runs affected", issue_df["run_id"].nunique())

    counts = issue_df.groupby(["run_id", "severity", "event"]).size().reset_index(name="count")
    if px is not None and not counts.empty:
        fig = px.bar(counts, x="run_id", y="count", color="event", facet_row="severity")
        fig.update_layout(height=520, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Issue Log")
    st.dataframe(issue_df, use_container_width=True, hide_index=True)


def render_payload_inspector(records: list[LogRecord], event_df: pd.DataFrame) -> None:
    if event_df.empty:
        st.info("No events to inspect.")
        return
    record_map = records_by_id(records)
    label_df = event_df.sort_values("event_id").copy()
    label_df["label"] = label_df.apply(
        lambda row: f"#{row['event_id']} {row['run_id']} {row['agent']}:{row['event']} {str(row['summary'])[:80]}",
        axis=1,
    )
    labels = label_df["label"].tolist()
    label_to_id = dict(zip(labels, label_df["event_id"].tolist()))
    selected = st.selectbox("Event", labels, index=len(labels) - 1)
    event_id = int(label_to_id[selected])
    record = record_map.get(event_id)
    if record is None:
        st.warning("Record not found.")
        return
    st.caption(f"{short_path(record.path)} line {record.line_no}")
    st.json(record.raw, expanded=False)


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def relaunch_with_streamlit_if_needed() -> None:
    if running_under_streamlit():
        return
    if os.environ.get("AGENT_DASHBOARD_NO_AUTO_STREAMLIT") == "1":
        return
    script_path = Path(__file__).resolve()
    command = [sys.executable, "-m", "streamlit", "run", str(script_path), *sys.argv[1:]]
    print("This is a Streamlit app. Launching it with:")
    print("  " + " ".join(command))
    raise SystemExit(subprocess.call(command))


def main() -> None:
    st.set_page_config(page_title="Optimized Agent Run Dashboard", layout="wide")
    st.title("Optimized Agent Run Dashboard")
    st.caption("Run-level analytics for JSONL logs in ./logs.")

    with st.sidebar:
        st.header("Input")
        log_dir_text = st.text_input("Log directory", value=str(DEFAULT_LOG_DIR))
        pattern = st.text_input("File pattern", value=DEFAULT_PATTERN)
        max_runs = st.number_input("Recent runs", min_value=1, max_value=200, value=50, step=1)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_sec = st.number_input("Refresh interval seconds", min_value=2, max_value=60, value=5, step=1)
        if st.button("Refresh now"):
            st.cache_data.clear()

    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{int(refresh_sec)}'>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def cached_load(path_text: str, glob_pattern: str, run_limit: int) -> tuple[list[LogRecord], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        log_dir = Path(path_text).expanduser()
        records = load_records(log_dir, glob_pattern)
        if records:
            run_order = []
            seen = set()
            for record in records:
                if record.run_id not in seen:
                    seen.add(record.run_id)
                    run_order.append(record.run_id)
            keep = set(run_order[-int(run_limit) :])
            records = [record for record in records if record.run_id in keep]
            records = [
                LogRecord(idx, r.run_id, r.path, r.line_no, r.timestamp, r.event, r.payload, r.raw, r.parse_error)
                for idx, r in enumerate(records)
            ]
        events = build_event_frame(records)
        tasks = build_task_frame(records)
        actions = build_action_frame(records)
        issues = build_issue_frame(records, events)
        runs = build_run_frame(records, events, issues)
        return records, runs, events, tasks, actions, issues

    log_dir = Path(log_dir_text).expanduser()
    if not log_dir.exists():
        st.error(f"Log directory does not exist: {log_dir}")
        st.stop()

    records, run_df, event_df, task_df, action_df, issue_df = cached_load(str(log_dir), pattern, int(max_runs))
    if run_df.empty:
        st.warning(f"No log records found under {log_dir} with pattern {pattern}.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        all_runs = run_df["run_id"].tolist()
        default_runs = all_runs[-min(len(all_runs), 15) :]
        selected_runs = st.multiselect("Runs", all_runs, default=default_runs)
        states = sorted(run_df["final_state"].dropna().unique().tolist())
        selected_states = st.multiselect("Final states", states, default=states)
        severities = sorted(event_df["severity"].dropna().unique().tolist())
        selected_severities = st.multiselect("Event severity", severities, default=severities)
        text_query = st.text_input("Search", value="")

    if selected_states:
        selected_runs = [r for r in selected_runs if run_df.loc[run_df["run_id"] == r, "final_state"].iloc[0] in selected_states]

    filtered_runs, filtered_events, filtered_tasks, filtered_actions, filtered_issues = apply_filters(
        run_df, event_df, task_df, action_df, issue_df, selected_runs
    )
    if selected_severities:
        filtered_events = filtered_events[filtered_events["severity"].isin(selected_severities)]
    if text_query.strip():
        q = text_query.strip().lower()
        filtered_events = filtered_events[
            filtered_events.apply(
                lambda row: q
                in " | ".join(
                    str(row.get(col, ""))
                    for col in ["run_id", "event", "agent", "task_id", "object_id", "robot_id", "action", "summary"]
                ).lower(),
                axis=1,
            )
        ]
        matching_ids = set(filtered_events["run_id"].tolist())
        filtered_runs = filtered_runs[filtered_runs["run_id"].isin(matching_ids)]
        filtered_tasks = filtered_tasks[filtered_tasks["run_id"].isin(matching_ids)]
        filtered_actions = filtered_actions[filtered_actions["run_id"].isin(matching_ids)]
        filtered_issues = filtered_issues[filtered_issues["run_id"].isin(matching_ids)]

    tabs = st.tabs(["Overview", "Run Detail", "Tasks", "Actions", "Issues", "Payload"])
    with tabs[0]:
        render_overview(filtered_runs, filtered_events, filtered_issues)
    with tabs[1]:
        render_run_deep_dive(filtered_runs, filtered_events, filtered_issues, filtered_runs["run_id"].tolist())
    with tabs[2]:
        render_tasks(filtered_tasks)
    with tabs[3]:
        render_actions(filtered_actions)
    with tabs[4]:
        render_issues(filtered_issues)
    with tabs[5]:
        render_payload_inspector(records, filtered_events)


if __name__ == "__main__":
    relaunch_with_streamlit_if_needed()
    main()
