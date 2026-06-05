import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv  # type: ignore[import-not-found]
from flask import Flask, jsonify, request, send_from_directory  # type: ignore[import-not-found]


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
HTML_DIR = STATIC_DIR / "html"

load_dotenv()

MODE = os.getenv("MODE", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or "").strip()
# DEFAULT_SYSTEM_PROMPT = os.getenv(
#     "DEFAULT_SYSTEM_PROMPT",
#     (
#         "You are a multimodal task decomposition assistant. "
#         "Given an image and a user prompt, return only valid JSON with this shape: "
#         "{\"summary\": string, \"partial_actions\": [{\"title\": string, \"details\": string}]} . "
#         "Keep the partial actions practical, ordered, and concise. Do not use markdown fences."
#     ),
# )
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    (
        "You are a multimodal robot task decomposition assistant. "
        "Your role is to convert a user's natural-language long-horizon instruction "
        "and the provided image into visually grounded atomic robot actions. "
        "Return only valid JSON with this exact shape: "
        "{\"summary\": string, \"partial_actions\": [{\"title\": string, \"details\": string}]} . "

        "Use the same language as the user prompt. "
        "Ground all actions in the visible scene and the user instruction. "
        "Do not invent objects, tools, containers, states, or spatial relations that are not visible or stated. "

        "Each partial action must describe exactly one robot-executable primitive operation. "
        "Each partial action must have exactly one explicit target object or target structure. "
        "Do not combine multiple objects in a single partial action. "
        "Do not combine multiple robot operations in a single partial action. "
        "Do not use abstract, compound, or goal-level action titles. "
        "Decompose compound goals into the smallest practical sequence of physical robot actions. "

        "The title of each partial action must contain a clear target and a clear primitive operation. "
        "The details of each partial action must specify the target, the operation, the relevant current location, "
        "the intended next state or destination when applicable, and any visual constraint needed for execution. "

        "Use stable and consistent wording for primitive operations so that the generated actions can be matched "
        "to a robot skill library by semantic similarity. "
        "Prefer physical robot primitives related to perception, approach, alignment, grasping, lifting, transporting, "
        "placing, releasing, pushing, pulling, and verification when they are applicable. "

        "If the instruction refers to multiple visible objects, generate a separate ordered action sequence for each object. "
        "If a requested action requires interaction with a structure, decompose the structure interaction into contact-based "
        "physical primitives rather than using a single high-level verb. "

        "Keep the sequence ordered, minimal, practical, and directly executable by a robot. "
        "If the scene is ambiguous, include only actions that are safely inferable and state the ambiguity in the details. "
        "Do not use markdown fences, numbering outside JSON, or any text outside the JSON object."
    ),
)


app = Flask(__name__, static_folder="static", static_url_path="/static")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _coerce_partial_actions(content: str) -> list[dict[str, str]]:
    cleaned = _strip_code_fences(content)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        actions = parsed.get("partial_actions")
        if isinstance(actions, list):
            normalized_actions: list[dict[str, str]] = []
            for index, action in enumerate(actions, start=1):
                if isinstance(action, dict):
                    title = str(action.get("title") or f"Action {index}").strip()
                    details = str(action.get("details") or action.get("description") or "").strip()
                else:
                    title = f"Action {index}"
                    details = str(action).strip()
                normalized_actions.append({"title": title or f"Action {index}", "details": details})
            if normalized_actions:
                return normalized_actions

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    actions: list[dict[str, str]] = []
    for index, line in enumerate(lines, start=1):
        normalized_line = re.sub(r"^[-*•\d.\s]+", "", line).strip()
        if not normalized_line:
            continue
        title, separator, details = normalized_line.partition(":")
        if separator:
            actions.append({"title": title.strip() or f"Action {index}", "details": details.strip()})
        else:
            actions.append({"title": f"Action {index}", "details": normalized_line})

    if actions:
        return actions

    return [{"title": "Action 1", "details": cleaned or "No response received from the model."}]


def _image_data_url(image_bytes: bytes, image_mimetype: str) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{image_mimetype};base64,{encoded_image}"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request_object = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")

    try:
        with urllib.request.urlopen(request_object, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed with HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach endpoint at {url}: {exc.reason}") from exc

    return json.loads(raw_response)


def call_ollama_with_image(image_bytes: bytes, image_mimetype: str, user_prompt: str, system_prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [base64.b64encode(image_bytes).decode("utf-8")],
            },
        ],
        "options": {
            "temperature": 0.2,
        },
    }

    response_payload = _post_json(OLLAMA_URL, payload, {})
    message = response_payload.get("message") or {}
    content = str(message.get("content") or response_payload.get("response") or "").strip()

    return {
        "model": response_payload.get("model", OLLAMA_MODEL),
        "raw_content": content,
        "partial_actions": _coerce_partial_actions(content),
    }


def call_openai_with_image(image_bytes: bytes, image_mimetype: str, user_prompt: str, system_prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 또는 CHATGPT_API_KEY가 .env에 필요합니다.")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_bytes, image_mimetype)}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    response_payload = _post_json(
        OPENAI_API_URL,
        payload,
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},
    )

    choices = response_payload.get("choices") or []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    content = str((message or {}).get("content") or "").strip()

    return {
        "model": response_payload.get("model", OPENAI_MODEL),
        "raw_content": content,
        "partial_actions": _coerce_partial_actions(content),
    }


def call_llm_with_image(image_bytes: bytes, image_mimetype: str, user_prompt: str, system_prompt: str) -> dict:
    if MODE == "chatgpt":
        return call_openai_with_image(image_bytes, image_mimetype, user_prompt, system_prompt)
    return call_ollama_with_image(image_bytes, image_mimetype, user_prompt, system_prompt)


@app.route("/")
def index() -> object:
    return send_from_directory(HTML_DIR, "index.html")


@app.route("/api/partial-actions", methods=["POST"])
def partial_actions() -> object:
    image_file = request.files.get("image")
    user_prompt = (request.form.get("user_prompt") or "").strip()
    system_prompt = (request.form.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()

    if image_file is None or image_file.filename == "":
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    if not user_prompt:
        return jsonify({"error": "유저 프롬프트가 필요합니다."}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "이미지 파일을 읽을 수 없습니다."}), 400

    try:
        result = call_llm_with_image(image_bytes, image_file.mimetype or "application/octet-stream", user_prompt, system_prompt)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health() -> object:
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)