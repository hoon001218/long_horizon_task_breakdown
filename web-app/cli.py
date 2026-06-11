#!/usr/bin/env python3
"""CLI utility to call OpenAI Chat API using the project's prompt.md templates.

Usage examples:
  python web-app/cli.py --command "Pick up the red cube" \
      --robot-specs '{"robots": []}' --env-state '{"objects": []}'

This script reads `web-app/prompt.md`, extracts the system prompt and the user
prompt template, fills placeholders, and sends a chat request to the OpenAI
Chat Completions API configured via .env (OPENAI_API_KEY, OPENAI_MODEL,
OPENAI_API_URL).

Design notes:
- The user prompt is constructed with labeled sections (`ROBOT_SPECS`,
  `ENV_STATE`, `EXECUTION_HISTORY`, `HIGH_LEVEL_COMMAND`) so that the parts
  other than `HIGH_LEVEL_COMMAND` can be easily parsed/forwarded to ROS later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROMPT_MD = BASE_DIR / "prompt.md"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")


def parse_prompt_md(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    # Find markers
    sys_marker = "<!-- System prompt -->"
    user_marker = "<!-- User prompt -->"
    if sys_marker in text and user_marker in text:
        sys_start = text.index(sys_marker) + len(sys_marker)
        user_start = text.index(user_marker)
        system_block = text[sys_start:user_start].strip()
        user_block = text[user_start + len(user_marker) :].strip()
        # Normalize indentation
        system_block = textwrap.dedent(system_block).strip()
        user_block = textwrap.dedent(user_block).strip()
        return system_block, user_block

    raise RuntimeError(f"Could not parse {path}; expected markers not found.")


def make_user_message(template: str, robot_specs: str, env_state: str, execution_history: str, user_command: str) -> str:
    # Ensure JSON-like inputs are pretty-printed for readability and downstream parsing
    def _pretty(s: str) -> str:
        if not s:
            return "{}"
        # If s is a path to a file, read it
        p = Path(s)
        if p.exists():
            try:
                return json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
            except Exception:
                return p.read_text(encoding="utf-8")
        try:
            parsed = json.loads(s)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return s

    sections = (
        "ROBOT_SPECS:\n" + _pretty(robot_specs),
        "ENV_STATE:\n" + _pretty(env_state),
        "EXECUTION_HISTORY:\n" + _pretty(execution_history),
        "HIGH_LEVEL_COMMAND:\n" + user_command,
    )

    # The template may already contain placeholders like {robot_specs}; attempt safe formatting
    try:
        filled = template.format(
            robot_specs=_pretty(robot_specs),
            env_state=_pretty(env_state),
            execution_history=_pretty(execution_history),
            user_command=user_command,
        )
    except Exception:
        # Fallback: concatenate labeled sections so they remain machine-parsable
        filled = "\n\n".join(sections)

    # Always append the labeled sections to make ROS integration easier
    filled += "\n\n---\n\n" + "\n\n".join(sections)
    return filled


def post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI for OpenAI chat using project prompt.md")
    parser.add_argument("--command", required=True, help="High-level user command (the {user_command} placeholder)")
    parser.add_argument("--robot-specs", default="", help="JSON string or path for robot specs (will be included in ROBOT_SPECS)")
    parser.add_argument("--env-state", default="", help="JSON string or path for environment state (will be included in ENV_STATE)")
    parser.add_argument("--history", default="", help="JSON string or path for execution history")
    parser.add_argument("--model", default=OPENAI_MODEL, help="Override model to use")
    parser.add_argument("--api-url", default=OPENAI_API_URL, help="Override OpenAI API URL")
    parser.add_argument("--json-output", action="store_true", help="Print assistant response as raw JSON")

    args = parser.parse_args(argv)

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY (or CHATGPT_API_KEY) is required in environment (.env).", file=sys.stderr)
        return 2

    try:
        system_prompt, user_template = parse_prompt_md(PROMPT_MD)
    except Exception as exc:
        print("Error reading prompt.md:", exc, file=sys.stderr)
        return 3

    user_message = make_user_message(user_template, args.robot_specs, args.env_state, args.history, args.command)

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        resp = post_json(args.api_url, payload, headers)
    except Exception as exc:
        print("API request failed:", exc, file=sys.stderr)
        return 4

    # For OpenAI Chat Completions, the top-level response has `choices` with `message`.
    choices = resp.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or choices[0].get("delta") or {}
        content = message.get("content") if isinstance(message, dict) else None
    else:
        content = resp.get("response") or resp.get("message")

    if args.json_output:
        print(json.dumps(resp, ensure_ascii=False))
    else:
        print("--- Assistant Response ---")
        print(content or json.dumps(resp, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
