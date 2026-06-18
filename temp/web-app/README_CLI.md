CLI for OpenAI prompt.md
========================

This CLI loads `web-app/prompt.md`, fills placeholders, and calls the OpenAI Chat API.

Setup
-----

1. Ensure `.env` contains `OPENAI_API_KEY` (or `CHATGPT_API_KEY`) and optionally `OPENAI_MODEL` and `OPENAI_API_URL`.

2. Install dependencies (only `python-dotenv` is required):

```bash
pip install python-dotenv
```

Usage
-----

```bash
python web-app/cli.py --command "Pick up the red cube" \
  --robot-specs '{"robots": []}' --env-state '{"objects": []}'
```

Notes
-----
- The CLI appends labeled sections (`ROBOT_SPECS`, `ENV_STATE`, `EXECUTION_HISTORY`, `HIGH_LEVEL_COMMAND`) so the non-command parts can be forwarded to a ROS bridge later.
