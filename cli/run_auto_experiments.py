#!/usr/bin/env python3
"""Run repeated autonomous experiments with rosbag and top-camera MP4 capture."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

ROOT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT_DIR / "main.py"
RESET_SCRIPT = ROOT_DIR / "reset_world.py"

IMAGE_TOPIC = "/world/top_camera/image_raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "experiment_records"
MODE_AUTO_INPUT = "auto"
MODE_COMMAND = "command"
MODE_PROMPT = "prompt"
MODE = os.environ.get("AUTO_EXPERIMENT_MODE", MODE_AUTO_INPUT).strip().lower()
PREDEFINED_COMMANDS = (
    "Move all objects to blue goal.",
)
DEFAULT_COMMAND = PREDEFINED_COMMANDS[0]
DEFAULT_RUN_TIMEOUT_SEC = 15 * 60
DEFAULT_VIDEO_FPS = 8.0

ACTIVE_PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESSES: dict[int, tuple[str, subprocess.Popen]] = {}


def unique_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_{uuid.uuid4().hex[:10]}"


def make_process_env() -> dict[str, str]:
    return os.environ.copy()


def register_process(name: str, process: subprocess.Popen) -> subprocess.Popen:
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES[process.pid] = (name, process)
    return process


def unregister_process(process: subprocess.Popen) -> None:
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.pop(process.pid, None)


def stop_active_processes() -> None:
    with ACTIVE_PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES.values())
    for name, process in processes:
        stop_process(process, name)


def start_process(command: list[str], log_path: Path | None = None) -> subprocess.Popen:
    stdout = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=make_process_env(),
        stdout=stdout or subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    if stdout is not None:
        setattr(process, "_automation_log_file", stdout)
    return register_process(command[0], process)


def close_process_log(process: subprocess.Popen) -> None:
    tee_thread = getattr(process, "_automation_tee_thread", None)
    if tee_thread is not None:
        try:
            tee_thread.join(timeout=2.0)
        except Exception:
            pass
    log_file = getattr(process, "_automation_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass


def tee_process_output(process: subprocess.Popen, log_file, prefix: str) -> None:
    if process.stdout is None:
        return
    try:
        for line in process.stdout:
            text = f"{prefix}{line}" if prefix else line
            print(text, end="", flush=True)
            log_file.write(line)
            log_file.flush()
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass


def stop_process(
    process: subprocess.Popen,
    name: str,
    sigint_timeout_sec: float = 10.0,
    terminate_timeout_sec: float = 5.0,
) -> None:
    if process.poll() is not None:
        close_process_log(process)
        unregister_process(process)
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except ProcessLookupError:
        unregister_process(process)
        return
    try:
        process.wait(timeout=sigint_timeout_sec)
        close_process_log(process)
        unregister_process(process)
        return
    except subprocess.TimeoutExpired:
        print(f"{name}: SIGINT timeout; terminating.", file=sys.stderr)

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        unregister_process(process)
        return
    try:
        process.wait(timeout=terminate_timeout_sec)
        close_process_log(process)
        unregister_process(process)
        return
    except subprocess.TimeoutExpired:
        print(f"{name}: SIGTERM timeout; killing.", file=sys.stderr)

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        unregister_process(process)
        return
    process.wait(timeout=terminate_timeout_sec)
    close_process_log(process)
    unregister_process(process)


def create_video_writer(path: Path, fps: float, size: tuple[int, int]):
    errors: list[str] = []
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
            True,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
        errors.append(codec)
    raise RuntimeError(
        f"Could not open MP4 writer for {path}; tried {', '.join(errors)}"
    )


def image_msg_to_bgr(message: RosImage) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")

    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channels = 4 if "a" in encoding else 3
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_min = height * step
    if raw.size < expected_min:
        raise ValueError(
            f"Image data shorter than expected: got={raw.size}, expected={expected_min}"
        )

    rows = raw[:expected_min].reshape((height, step))
    pixels = rows[:, : width * channels].reshape((height, width, channels))
    if encoding == "bgr8":
        return np.ascontiguousarray(pixels)
    if encoding == "rgb8":
        return np.ascontiguousarray(pixels[:, :, ::-1])
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)


class ImageRecorder(Node):
    def __init__(self, output_path: Path, fps: float) -> None:
        super().__init__(f"auto_experiment_image_recorder_{uuid.uuid4().hex[:8]}")
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self.codec = ""
        self.frame_count = 0
        self.error: Exception | None = None
        self.lock = threading.Lock()
        self.create_subscription(RosImage, IMAGE_TOPIC, self._on_image, 10)

    def _on_image(self, message: RosImage) -> None:
        try:
            frame = image_msg_to_bgr(message)
            height, width = frame.shape[:2]
            with self.lock:
                if self.writer is None:
                    self.writer, self.codec = create_video_writer(
                        self.output_path, self.fps, (width, height)
                    )
                self.writer.write(frame)
                self.frame_count += 1
        except Exception as exc:
            self.error = exc
            self.get_logger().error(f"Failed to record image frame: {exc}")

    def close(self) -> None:
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None


class SpinThread:
    def __init__(self, node: Node) -> None:
        self.node = node
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while rclpy.ok() and not self.stop_event.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)


def start_rosbag(bag_path: Path, log_path: Path) -> subprocess.Popen:
    command = [
        "ros2",
        "bag",
        "record",
        "-a",
        "-x",
        f"^{IMAGE_TOPIC}$",
        "-o",
        str(bag_path),
    ]
    return start_process(command, log_path)


def run_main(
    command: str | None,
    log_path: Path,
    planner: str,
    max_iterations: int,
) -> subprocess.Popen:
    env = make_process_env()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process_command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--planner",
        planner,
        "--max-iterations",
        str(max_iterations),
    ]
    if command:
        process_command.append(command)
    process = subprocess.Popen(
        process_command,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    setattr(process, "_automation_log_file", log_file)
    tee_thread = threading.Thread(
        target=tee_process_output,
        args=(process, log_file, "[cli] "),
        daemon=True,
    )
    setattr(process, "_automation_tee_thread", tee_thread)
    tee_thread.start()
    return register_process("main.py", process)


def call_reset_world(timeout_sec: float, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(RESET_SCRIPT), "--timeout", str(timeout_sec)],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        register_process("reset_world", process)
        try:
            return int(process.wait(timeout=timeout_sec + 5.0))
        except subprocess.TimeoutExpired:
            stop_process(process, "reset_world")
            return int(process.returncode or 130)
        finally:
            unregister_process(process)


def wait_for_main(process: subprocess.Popen, timeout_sec: float) -> str:
    try:
        process.wait(timeout=timeout_sec)
        close_process_log(process)
        unregister_process(process)
        return "completed"
    except subprocess.TimeoutExpired:
        stop_process(process, "main.py")
        return "timeout"


def run_once(
    run_dir: Path,
    run_id: str,
    command: str | None,
    timeout_sec: float,
    video_fps: float,
    reset_timeout_sec: float,
    planner: str,
    max_iterations: int,
) -> dict[str, object]:
    bag_path = run_dir / run_id
    video_path = run_dir / f"{run_id}.mp4"
    logs_dir = run_dir / "logs"

    recorder: ImageRecorder | None = None
    spinner: SpinThread | None = None
    bag_process: subprocess.Popen | None = None
    main_process: subprocess.Popen | None = None

    status = "unknown"
    main_returncode: int | None = None
    run_error = ""
    try:
        recorder = ImageRecorder(video_path, video_fps)
        spinner = SpinThread(recorder)
        spinner.start()
        bag_process = start_rosbag(bag_path, logs_dir / "rosbag.log")
        time.sleep(1.0)
        if bag_process.poll() is not None:
            raise RuntimeError(
                f"ros2 bag exited early with code {bag_process.returncode}; "
                f"see {logs_dir / 'rosbag.log'}"
            )
        main_process = run_main(
            command,
            logs_dir / "main.log",
            planner=planner,
            max_iterations=max_iterations,
        )
        status = wait_for_main(main_process, timeout_sec)
        main_returncode = main_process.returncode
    except Exception as exc:
        status = "error"
        run_error = str(exc)
    finally:
        if main_process is not None:
            stop_process(main_process, "main.py")
        if bag_process is not None:
            stop_process(bag_process, "ros2 bag")
        if spinner is not None:
            spinner.stop()
        if recorder is not None:
            recorder.close()
            recorder.destroy_node()

    reset_error = ""
    try:
        reset_returncode: int | None = call_reset_world(
            reset_timeout_sec, logs_dir / "reset_world.log"
        )
    except Exception as exc:
        reset_returncode = None
        reset_error = str(exc)
    if recorder is not None and recorder.error is not None:
        status = f"{status}_image_error"

    return {
        "run_id": run_id,
        "command": command or "",
        "status": status,
        "main_returncode": main_returncode,
        "reset_returncode": reset_returncode,
        "run_error": run_error,
        "reset_error": reset_error,
        "bag_path": str(bag_path),
        "video_path": str(video_path),
        "video_frames": 0 if recorder is None else recorder.frame_count,
        "video_codec": "" if recorder is None else recorder.codec,
    }


def command_for_run(
    mode: str,
    run_number: int,
    command: str | None,
    predefined_commands: list[str],
) -> str | None:
    if mode == MODE_PROMPT:
        return None
    if mode == MODE_COMMAND:
        if not command:
            raise ValueError("--command is required when --mode command is used.")
        return command
    if mode == MODE_AUTO_INPUT:
        commands = predefined_commands or [DEFAULT_COMMAND]
        return commands[(run_number - 1) % len(commands)]
    raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatedly run main.py while recording rosbag and MP4 video."
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_AUTO_INPUT, MODE_COMMAND, MODE_PROMPT],
        default=MODE if MODE in {MODE_AUTO_INPUT, MODE_COMMAND, MODE_PROMPT} else MODE_AUTO_INPUT,
        help=(
            "Input mode for main.py. "
            "'auto' cycles through predefined commands, "
            "'command' uses --command, and 'prompt' lets main.py ask Command>. "
            f"Default: {MODE_AUTO_INPUT}"
        ),
    )
    parser.add_argument(
        "--auto-command",
        action="append",
        dest="auto_commands",
        default=None,
        help=(
            "Command used in --mode auto. Can be repeated; runs cycle through "
            "the provided commands. Default: predefined command list."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of experiment runs. Use 0 for an infinite loop. Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for all experiment outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Command passed to main.py when --mode command is used.",
    )
    parser.add_argument(
        "--planner",
        choices=["auto", "openai", "heuristic"],
        default="auto",
        help="Planner mode passed to main.py. Default: auto",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max closed-loop iterations passed to main.py. Default: 20",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_RUN_TIMEOUT_SEC,
        help=f"Seconds before main.py is stopped with SIGINT. Default: {DEFAULT_RUN_TIMEOUT_SEC:g}",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help="FPS written into the MP4 file. Default: 8",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for /world/reset after each run. Default: 30",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    run_number = 0
    try:
        while args.runs == 0 or run_number < args.runs:
            run_number += 1
            run_id = unique_run_id()
            run_dir = args.output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            command = command_for_run(
                args.mode,
                run_number,
                args.command,
                args.auto_commands or list(PREDEFINED_COMMANDS),
            )
            print(f"[{run_id}] starting mode={args.mode} command={command or '<prompt>'}")
            result = run_once(
                run_dir=run_dir,
                run_id=run_id,
                command=command,
                timeout_sec=max(float(args.timeout), 1.0),
                video_fps=max(float(args.video_fps), 0.1),
                reset_timeout_sec=max(float(args.reset_timeout), 1.0),
                planner=args.planner,
                max_iterations=max(int(args.max_iterations), 1),
            )
            summary_path = run_dir / "summary.json"
            summary_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{run_id}] {result['status']} "
                f"frames={result['video_frames']} reset={result['reset_returncode']}"
            )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        stop_active_processes()
        return 130
    finally:
        stop_active_processes()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
