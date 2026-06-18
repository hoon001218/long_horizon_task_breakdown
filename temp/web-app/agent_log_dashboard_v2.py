#!/usr/bin/env python3
"""Streamlit dashboard for cli_multi_agent_chatgpt.py agent JSONL logs.

Run:
    pip install streamlit pandas plotly
    streamlit run agent_log_dashboard.py

Default log directory:
    ./logs/agents
or set:
    AGENT_LOG_DIR=/path/to/logs/agents streamlit run agent_log_dashboard.py

The dashboard is intentionally read-only. It does not import ROS2 and does not
modify cli_multi_agent_chatgpt.py. It consumes the JSONL records written by
AgentLogger: {"time": ..., "event": ..., "payload": ...}.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency in minimal installs
    px = None


DEFAULT_LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", Path(__file__).resolve().parent / "logs" / "agents"))

GOAL_TERMS: dict[str, tuple[str, ...]] = {
    "red": (
        "red_goal",
        "red goal",
        "red target",
        "red_goal:1",
        "빨간 목표",
        "빨강 목표",
        "빨간색 목표",
        "빨간 목표점",
        "빨강 목표점",
    ),
    "blue": (
        "blue_goal",
        "blue goal",
        "blue target",
        "blue_goal:2",
        "파란 목표",
        "파랑 목표",
        "파란색 목표",
        "파란 목표점",
        "파랑 목표점",
    ),
}

OBJECT_ID_RE = re.compile(r"\b(?:cube|sphere|capsule)_\d+:\d+\b")
ACTION_TERMS = ("Moving", "Centering", "Placing", "Grip", "Release", "Homing")
TROUBLE_STATUSES = {
    "success",
    "pending",
    "retry",
    "replan_task",
    "replan_all",
    "emergency_recover",
    "complete",
}
MAIN_DECISIONS = {"continue", "retry", "replan_task", "replan_all", "emergency_recover", "complete"}

# DataFrame columns that the dashboard expects even when no log records exist.
# Keeping these columns prevents KeyError failures on an empty log directory.
SUMMARY_COLUMNS = [
    "event_id",
    "agent",
    "event",
    "time",
    "timestamp",
    "path",
    "line_no",
    "command",
    "explicit_goal",
    "task",
    "tasks_count",
    "status_decision",
    "message",
    "robot_id",
    "action",
    "target_object_id",
    "goal_mentions",
    "summary",
    "parse_error",
    "_payload",
    "_raw",
    "explicit_run_id",
    "session_id",
    "session_command",
    "session_explicit_goal",
    "elapsed_sec",
]

VIOLATION_COLUMNS = [
    "event_id",
    "session_id",
    "agent",
    "event",
    "timestamp",
    "expected_goal",
    "unexpected_goal",
    "task",
    "summary",
    "reason",
]


@dataclass(frozen=True)
class LogRecord:
    event_id: int
    agent: str
    path: str
    line_no: int
    time: float | None
    event: str
    payload: dict[str, Any]
    raw: dict[str, Any]
    parse_error: str | None = None


def safe_json_dumps(value: Any, max_chars: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = str(value)
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
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


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json_dumps(value, max_chars=2000)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def detect_single_explicit_goal(command: str | None) -> str | None:
    if not command:
        return None
    mentioned = [color for color, terms in GOAL_TERMS.items() if contains_any(command, terms)]
    if len(mentioned) == 1:
        return mentioned[0]
    return None


def goal_mentions(text: str | None) -> set[str]:
    if not text:
        return set()
    return {color for color, terms in GOAL_TERMS.items() if contains_any(text, terms)}


def find_object_ids(value: Any) -> list[str]:
    found = OBJECT_ID_RE.findall(as_text(value))
    return sorted(set(found))


def load_jsonl_logs(log_dir: Path, include_patterns: list[str] | None = None) -> list[LogRecord]:
    patterns = include_patterns or ["*.log"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(log_dir.glob(pattern)))
    files = sorted(set(files))
    records: list[LogRecord] = []
    event_id = 0
    for path in files:
        agent = path.stem
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parse_error = None
                    try:
                        raw = json.loads(line)
                    except Exception as exc:
                        raw = {"time": None, "event": "parse_error", "payload": {"line": line}}
                        parse_error = str(exc)
                    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
                    event = str(raw.get("event", "unknown"))
                    t = raw.get("time")
                    try:
                        t_float = float(t) if t is not None else None
                    except Exception:
                        t_float = None
                    records.append(
                        LogRecord(
                            event_id=event_id,
                            agent=agent,
                            path=str(path),
                            line_no=line_no,
                            time=t_float,
                            event=event,
                            payload=payload,
                            raw=raw,
                            parse_error=parse_error,
                        )
                    )
                    event_id += 1
        except FileNotFoundError:
            continue
    records.sort(key=lambda item: ((item.time if item.time is not None else float("inf")), item.event_id))
    return [record.__class__(idx, record.agent, record.path, record.line_no, record.time, record.event, record.payload, record.raw, record.parse_error) for idx, record in enumerate(records)]


def extract_command(payload: dict[str, Any]) -> str | None:
    candidates = [
        get_in(payload, ["user", "user_command"]),
        get_in(payload, ["request", "user_command"]),
        get_in(payload, ["user_command"]),
        get_in(payload, ["payload", "user_command"]),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def extract_parsed(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = payload.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def extract_task(payload: dict[str, Any]) -> str | None:
    candidates = [
        get_in(payload, ["user", "task"]),
        get_in(payload, ["task"]),
        get_in(payload, ["request", "task"]),
        get_in(payload, ["parsed", "task"]),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def extract_tasks(payload: dict[str, Any]) -> list[str]:
    parsed = extract_parsed(payload)
    candidates = [
        parsed.get("tasks"),
        get_in(payload, ["user", "tasks"]),
        get_in(payload, ["tasks"]),
    ]
    for item in candidates:
        if isinstance(item, list):
            return [str(x) for x in item]
    return []


def extract_action(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        get_in(payload, ["user", "action_result", "action"]),
        get_in(payload, ["user", "action"]),
        get_in(payload, ["action"]),
        get_in(payload, ["parsed", "action"]),
    ]
    for item in candidates:
        if isinstance(item, dict):
            return item
    return {}


def extract_status_or_decision(event: str, payload: dict[str, Any]) -> str:
    if event == "error":
        return "error"
    parsed = extract_parsed(payload)
    for key in ("status", "decision"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            return value
    # MainAgent request may contain existing troubleshooter reports.
    report_status = get_in(payload, ["status"])
    if isinstance(report_status, str):
        return report_status
    return ""


def extract_message(payload: dict[str, Any]) -> str:
    parsed = extract_parsed(payload)
    candidates = [
        parsed.get("message"),
        parsed.get("notes"),
        payload.get("error"),
        get_in(payload, ["user", "feedback"]),
        get_in(payload, ["user", "action_result", "message"]),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    raw = payload.get("raw")
    if isinstance(raw, str):
        return raw.strip()
    return ""


def summarize_record(record: LogRecord) -> dict[str, Any]:
    payload = record.payload
    parsed = extract_parsed(payload)
    task = extract_task(payload)
    tasks = extract_tasks(payload)
    action = extract_action(payload)
    command = extract_command(payload)
    message = extract_message(payload)
    status = extract_status_or_decision(record.event, payload)

    action_name = action.get("action") if isinstance(action.get("action"), str) else ""
    robot_id = action.get("robot_id") if isinstance(action.get("robot_id"), str) else ""
    target_object_id = action.get("target_object_id") if isinstance(action.get("target_object_id"), str) else ""
    if not target_object_id:
        ids = find_object_ids(task or tasks or payload)
        target_object_id = ", ".join(ids[:4])

    goal_text_parts = []
    if task:
        goal_text_parts.append(task)
    if tasks:
        goal_text_parts.extend(tasks)
    if message:
        goal_text_parts.append(message)
    goals = sorted(goal_mentions(" | ".join(goal_text_parts)))

    if record.time is not None:
        dt = datetime.fromtimestamp(record.time)
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    else:
        timestamp = ""

    if record.event == "request":
        summary_bits = []
        if command:
            summary_bits.append(f"command={command}")
        if task:
            summary_bits.append(f"task={task}")
        if action_name:
            summary_bits.append(f"action={robot_id}:{action_name}:{target_object_id}")
        if tasks:
            summary_bits.append(f"tasks={len(tasks)}")
        summary = " | ".join(summary_bits) or "request"
    elif record.event == "response":
        if "tasks" in parsed:
            summary = f"tasks={len(tasks)}"
            if parsed.get("notes"):
                summary += f" | {parsed.get('notes')}"
        elif status:
            summary = f"{status} | {message}"
        else:
            summary = message or "response"
    elif record.event == "error":
        summary = message or "error"
    else:
        summary = message or record.event

    return {
        "event_id": record.event_id,
        "agent": record.agent,
        "event": record.event,
        "time": record.time,
        "timestamp": timestamp,
        "path": record.path,
        "line_no": record.line_no,
        "command": command or "",
        "explicit_goal": detect_single_explicit_goal(command),
        "task": task or "",
        "tasks_count": len(tasks),
        "status_decision": status,
        "message": message,
        "robot_id": robot_id,
        "action": action_name,
        "target_object_id": target_object_id,
        "goal_mentions": ",".join(goals),
        "summary": summary,
        "parse_error": record.parse_error or "",
    }


def ensure_dashboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame that always has the columns used by the UI."""
    if df is None or df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for column in SUMMARY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def assign_sessions(df: pd.DataFrame, gap_sec: float) -> pd.DataFrame:
    df = ensure_dashboard_columns(df)
    if df.empty:
        return df
    df = df.sort_values(["time", "event_id"], na_position="last").copy()
    explicit_run = []
    for _, row in df.iterrows():
        record_payload = row.get("_payload", {})
        raw = row.get("_raw", {})
        run_id = None
        if isinstance(raw, dict):
            run_id = raw.get("run_id") or raw.get("trace_id") or raw.get("session_id")
        if run_id is None and isinstance(record_payload, dict):
            run_id = record_payload.get("run_id") or get_in(record_payload, ["user", "run_id"])
        explicit_run.append(str(run_id) if run_id else "")
    df["explicit_run_id"] = explicit_run

    session_ids: list[str] = []
    current = -1
    last_time: float | None = None
    active_explicit = ""
    for _, row in df.iterrows():
        row_run = str(row.get("explicit_run_id", ""))
        t = row.get("time")
        t_float = float(t) if pd.notna(t) else None
        new_session = False
        if row_run:
            if row_run != active_explicit:
                new_session = True
                active_explicit = row_run
        elif current < 0:
            new_session = True
        elif t_float is not None and last_time is not None and t_float - last_time > gap_sec:
            new_session = True
        if new_session:
            current += 1
        session_ids.append(row_run or f"session_{current:03d}")
        if t_float is not None:
            last_time = t_float
    df["session_id"] = session_ids

    # Propagate first command/explicit goal inside each inferred session.
    session_command: dict[str, str] = {}
    session_goal: dict[str, str] = {}
    for session_id, group in df.groupby("session_id", sort=False):
        commands = [c for c in group["command"].tolist() if isinstance(c, str) and c]
        command = commands[0] if commands else ""
        session_command[session_id] = command
        session_goal[session_id] = detect_single_explicit_goal(command) or ""
    df["session_command"] = df["session_id"].map(session_command).fillna("")
    df["session_explicit_goal"] = df["session_id"].map(session_goal).fillna("")

    first_time = df.groupby("session_id")["time"].transform("min")
    df["elapsed_sec"] = df["time"] - first_time
    return df


def detect_violations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = ensure_dashboard_columns(df)
    for _, row in df.iterrows():
        explicit_goal = row.get("session_explicit_goal") or row.get("explicit_goal") or ""
        if explicit_goal not in GOAL_TERMS:
            continue
        other_goals = [color for color in GOAL_TERMS if color != explicit_goal]
        search_text = " | ".join(
            str(row.get(col, ""))
            for col in ("summary", "task", "message", "goal_mentions")
        )
        for other in other_goals:
            if contains_any(search_text, GOAL_TERMS[other]):
                rows.append(
                    {
                        "event_id": row.get("event_id", ""),
                        "session_id": row.get("session_id", ""),
                        "agent": row.get("agent", ""),
                        "event": row.get("event", ""),
                        "timestamp": row.get("timestamp", ""),
                        "expected_goal": explicit_goal,
                        "unexpected_goal": other,
                        "task": row.get("task", ""),
                        "summary": row.get("summary", ""),
                        "reason": f"User command explicitly targets {explicit_goal}_goal, but this record mentions {other}_goal.",
                    }
                )
    return pd.DataFrame(rows, columns=VIOLATION_COLUMNS)


def build_dataframe(records: list[LogRecord], session_gap_sec: float) -> pd.DataFrame:
    rows = []
    for record in records:
        row = summarize_record(record)
        row["_payload"] = record.payload
        row["_raw"] = record.raw
        rows.append(row)
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS) if not rows else pd.DataFrame(rows)
    df = assign_sessions(df, session_gap_sec)
    return ensure_dashboard_columns(df)


def get_record_by_id(records: list[LogRecord], event_id: int) -> LogRecord | None:
    for record in records:
        if record.event_id == event_id:
            return record
    return None


def objects_by_id(snapshot: Any) -> dict[str, dict[str, Any]]:
    objects = get_in(snapshot, ["objects"], [])
    if not isinstance(objects, list):
        return {}
    result = {}
    for obj in objects:
        if isinstance(obj, dict) and obj.get("id"):
            result[str(obj["id"])] = obj
    return result


def pose_xyz(obj: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(obj, dict):
        return None
    pos = get_in(obj, ["pose", "position"])
    if not isinstance(pos, dict):
        pos = obj.get("position") if isinstance(obj.get("position"), dict) else None
    if not isinstance(pos, dict):
        return None
    try:
        return {"x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.0)), "z": float(pos.get("z", 0.0))}
    except Exception:
        return None


def object_pose_deltas(snapshot_before: Any, snapshot_after: Any) -> pd.DataFrame:
    before = objects_by_id(snapshot_before)
    after = objects_by_id(snapshot_after)
    rows = []
    for object_id in sorted(set(before) | set(after)):
        b = pose_xyz(before.get(object_id))
        a = pose_xyz(after.get(object_id))
        if b is None or a is None:
            continue
        rows.append(
            {
                "object_id": object_id,
                "before_x": b["x"],
                "before_y": b["y"],
                "before_z": b["z"],
                "after_x": a["x"],
                "after_y": a["y"],
                "after_z": a["z"],
                "dx": a["x"] - b["x"],
                "dy": a["y"] - b["y"],
                "dz": a["z"] - b["z"],
            }
        )
    return pd.DataFrame(rows)


def build_graphviz_dot(df: pd.DataFrame, max_nodes: int = 80) -> str:
    if df.empty:
        return "digraph G { empty [label=\"No events\"]; }"
    view = df.sort_values(["time", "event_id"], na_position="last").head(max_nodes)
    lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [shape=box, style=rounded, fontsize=10];",
    ]
    previous_name = None
    for _, row in view.iterrows():
        node_name = f"n{int(row['event_id'])}"
        label = f"#{int(row['event_id'])} {row['agent']}:{row['event']}\\n{str(row.get('summary',''))[:70]}"
        status = str(row.get("status_decision", ""))
        color = "black"
        if status in {"retry", "replan_task", "replan_all", "emergency_recover", "error"}:
            color = "red"
        elif status in {"pending"}:
            color = "orange"
        elif status in {"success", "continue", "complete"}:
            color = "green"
        lines.append(f'{node_name} [label={json.dumps(label)}, color="{color}"];')
        if previous_name is not None:
            lines.append(f"{previous_name} -> {node_name};")
        previous_name = node_name
    lines.append("}")
    return "\n".join(lines)


def render_payload(record: LogRecord | None) -> None:
    if record is None:
        st.info("Select an event to inspect its payload.")
        return
    st.markdown(f"**Event #{record.event_id} — `{record.agent}` / `{record.event}` / line {record.line_no}**")
    st.json(record.raw, expanded=False)

    user_payload = record.payload.get("user") if isinstance(record.payload.get("user"), dict) else {}
    if user_payload:
        before = user_payload.get("snapshot_before")
        after = user_payload.get("snapshot_after")
        if before and after:
            deltas = object_pose_deltas(before, after)
            if not deltas.empty:
                st.subheader("Object pose deltas in selected event")
                st.dataframe(deltas, use_container_width=True, hide_index=True)


def running_under_streamlit() -> bool:
    """True when this file is being executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def relaunch_with_streamlit_if_needed() -> None:
    """Allow `python3 agent_log_dashboard.py` to work by re-launching Streamlit."""
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
    st.set_page_config(page_title="Multi-Agent Robot Log Dashboard", layout="wide")
    st.title("Multi-Agent Robot Log Dashboard")
    st.caption("Read-only dashboard for cli_multi_agent_chatgpt.py JSONL agent logs.")

    with st.sidebar:
        st.header("Input")
        log_dir_text = st.text_input("Log directory", value=str(DEFAULT_LOG_DIR))
        log_dir = Path(log_dir_text).expanduser()
        session_gap_sec = st.number_input("Session gap seconds", min_value=10, max_value=3600, value=180, step=10)
        tail_events = st.number_input("Max loaded events", min_value=50, max_value=50000, value=5000, step=50)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_sec = st.number_input("Refresh interval seconds", min_value=1, max_value=60, value=3, step=1)
        if st.button("Refresh now"):
            st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def cached_load(path_text: str, gap: float, max_events: int) -> tuple[list[LogRecord], pd.DataFrame]:
        loaded_records = load_jsonl_logs(Path(path_text).expanduser())
        if max_events and len(loaded_records) > max_events:
            loaded_records = loaded_records[-int(max_events):]
            loaded_records = [record.__class__(idx, record.agent, record.path, record.line_no, record.time, record.event, record.payload, record.raw, record.parse_error) for idx, record in enumerate(loaded_records)]
        return loaded_records, build_dataframe(loaded_records, gap)

    if auto_refresh:
        # Streamlit reruns on interaction; this meta refresh keeps a live-ish log viewer without custom components.
        st.markdown(f"<meta http-equiv='refresh' content='{int(refresh_sec)}'>", unsafe_allow_html=True)

    if not log_dir.exists():
        st.error(f"Log directory does not exist: {log_dir}")
        st.stop()

    records, df = cached_load(str(log_dir), float(session_gap_sec), int(tail_events))
    if df.empty:
        st.warning(f"No JSONL log records found under: {log_dir}")
        st.stop()

    violations = detect_violations(df)

    with st.sidebar:
        st.header("Filters")
        sessions = ["<all>"] + sorted(df["session_id"].dropna().unique().tolist())
        selected_session = st.selectbox("Session", sessions, index=max(0, len(sessions) - 1) if len(sessions) > 1 else 0)
        agents = ["<all>"] + sorted(df["agent"].dropna().unique().tolist())
        selected_agents = st.multiselect("Agents", agents[1:], default=agents[1:])
        events = ["<all>"] + sorted(df["event"].dropna().unique().tolist())
        selected_events = st.multiselect("Events", events[1:], default=events[1:])
        status_values = sorted([x for x in df["status_decision"].dropna().unique().tolist() if x])
        selected_status = st.multiselect("Status / decision", status_values, default=status_values)
        text_query = st.text_input("Search text", value="")

    filtered = df.copy()
    if selected_session != "<all>":
        filtered = filtered[filtered["session_id"] == selected_session]
    if selected_agents:
        filtered = filtered[filtered["agent"].isin(selected_agents)]
    if selected_events:
        filtered = filtered[filtered["event"].isin(selected_events)]
    if selected_status:
        filtered = filtered[(filtered["status_decision"].isin(selected_status)) | (filtered["status_decision"] == "")]
    if text_query.strip():
        q = text_query.strip().lower()
        mask = filtered.apply(
            lambda row: q in " | ".join(str(row.get(col, "")) for col in ["summary", "task", "message", "command", "target_object_id"]).lower(),
            axis=1,
        )
        filtered = filtered[mask]

    metric_cols = st.columns(6)
    metric_cols[0].metric("Events", len(filtered))
    metric_cols[1].metric("Sessions", filtered["session_id"].nunique())
    metric_cols[2].metric("Agents", filtered["agent"].nunique())
    metric_cols[3].metric("Requests", int((filtered["event"] == "request").sum()))
    metric_cols[4].metric("Responses", int((filtered["event"] == "response").sum()))
    if selected_session == "<all>":
        selected_violations = violations
    elif "session_id" in violations.columns:
        selected_violations = violations[violations["session_id"] == selected_session]
    else:
        selected_violations = pd.DataFrame(columns=VIOLATION_COLUMNS)
    metric_cols[5].metric("Violations", len(selected_violations))

    if selected_session != "<all>":
        session_rows = df[df["session_id"] == selected_session]
        command = session_rows["session_command"].iloc[0] if not session_rows.empty else ""
        explicit_goal = session_rows["session_explicit_goal"].iloc[0] if not session_rows.empty else ""
        st.info(f"Selected session: `{selected_session}` | command: {command or 'unknown'} | explicit goal: {explicit_goal or 'none/ambiguous'}")

    tab_overview, tab_timeline, tab_events, tab_tasks, tab_violations, tab_graph, tab_inspector = st.tabs(
        ["Overview", "Timeline", "Events", "Tasks & decisions", "Violations", "Flow graph", "Payload inspector"]
    )

    with tab_overview:
        st.subheader("Status / decision counts")
        counts = filtered["status_decision"].replace("", "(none)").value_counts().reset_index()
        counts.columns = ["status_decision", "count"]
        left, right = st.columns([1, 2])
        with left:
            st.dataframe(counts, use_container_width=True, hide_index=True)
        with right:
            if px is not None and not counts.empty:
                fig = px.bar(counts, x="status_decision", y="count", title="Status / decision distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Install plotly to render charts: pip install plotly")

        st.subheader("Agent event counts")
        agent_counts = filtered.groupby(["agent", "event"]).size().reset_index(name="count")
        st.dataframe(agent_counts, use_container_width=True, hide_index=True)

    with tab_timeline:
        st.subheader("Event timeline")
        if px is not None:
            plot_df = filtered.dropna(subset=["time"]).copy()
            if plot_df.empty:
                st.warning("No timestamped events available.")
            else:
                fig = px.scatter(
                    plot_df,
                    x="elapsed_sec" if selected_session != "<all>" else "time",
                    y="agent",
                    color="status_decision",
                    symbol="event",
                    hover_data={
                        "event_id": True,
                        "timestamp": True,
                        "summary": True,
                        "task": True,
                        "action": True,
                        "target_object_id": True,
                        "time": False,
                    },
                    title="Agent event timeline",
                )
                fig.update_traces(marker={"size": 11})
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Install plotly to render timeline: pip install plotly")
        st.dataframe(
            filtered[["event_id", "timestamp", "session_id", "agent", "event", "status_decision", "summary"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_events:
        st.subheader("Filtered event table")
        visible_cols = [
            "event_id",
            "timestamp",
            "session_id",
            "agent",
            "event",
            "status_decision",
            "robot_id",
            "action",
            "target_object_id",
            "goal_mentions",
            "summary",
        ]
        st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)
        csv = filtered[visible_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download filtered events as CSV", csv, file_name="agent_events_filtered.csv", mime="text/csv")

    with tab_tasks:
        st.subheader("TaskAgent task outputs")
        task_response_rows = filtered[(filtered["agent"] == "task") & (filtered["event"] == "response") & (filtered["tasks_count"] > 0)]
        expanded_rows = []
        for _, row in task_response_rows.iterrows():
            record = get_record_by_id(records, int(row["event_id"]))
            if record is None:
                continue
            for idx, task in enumerate(extract_tasks(record.payload), start=1):
                expanded_rows.append(
                    {
                        "event_id": row["event_id"],
                        "session_id": row["session_id"],
                        "timestamp": row["timestamp"],
                        "task_no": idx,
                        "task": task,
                        "goal_mentions": ",".join(sorted(goal_mentions(task))),
                    }
                )
        st.dataframe(pd.DataFrame(expanded_rows), use_container_width=True, hide_index=True)

        st.subheader("Main / Troubleshooter decisions")
        decision_rows = filtered[filtered["status_decision"].isin(TROUBLE_STATUSES | MAIN_DECISIONS | {"error"})]
        st.dataframe(
            decision_rows[["event_id", "timestamp", "session_id", "agent", "event", "status_decision", "task", "action", "target_object_id", "message"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_violations:
        st.subheader("Detected semantic violations")
        display_violations = selected_violations
        if display_violations.empty:
            st.success("No explicit-goal contradiction detected in the loaded logs.")
        else:
            st.error(f"Detected {len(display_violations)} potential violation(s).")
            st.dataframe(display_violations, use_container_width=True, hide_index=True)

        st.caption(
            "Current rule: if a session command explicitly mentions exactly one goal color, task/message records that mention another goal color are flagged. "
            "This is intentionally conservative and should be treated as a debugging signal, not a formal proof."
        )

    with tab_graph:
        st.subheader("Sequential flow graph")
        if selected_session == "<all>":
            st.warning("Select a single session in the sidebar for a readable graph.")
        else:
            graph_df = df[df["session_id"] == selected_session]
            dot = build_graphviz_dot(graph_df, max_nodes=120)
            try:
                st.graphviz_chart(dot, use_container_width=True)
            except Exception as exc:
                st.code(dot, language="dot")
                st.warning(f"Graphviz rendering failed: {exc}")

    with tab_inspector:
        st.subheader("Payload inspector")
        ids = filtered["event_id"].tolist()
        if not ids:
            st.info("No event is selected by the current filters.")
        else:
            default_index = len(ids) - 1
            selected_event = st.selectbox("Event ID", ids, index=default_index, format_func=lambda x: f"#{x}")
            render_payload(get_record_by_id(records, int(selected_event)))


if __name__ == "__main__":
    relaunch_with_streamlit_if_needed()
    main()
