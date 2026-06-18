#!/usr/bin/env python3
"""Convert WebM video files by reading and writing frames with OpenCV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one or more .webm files without audio using OpenCV."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .webm file(s). Shell globs such as *.webm are supported.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path. Only valid when converting a single input file.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--fourcc",
        help=(
            "FourCC codec for output. "
            "Default: XVID/MJPG/DIVX for .avi, or avc1/H264/mp4v for .mp4."
        ),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Write every Nth frame. Default: 2, which halves the frame rate.",
    )
    return parser.parse_args()


def output_path_for(input_path: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser()
    return input_path.with_suffix(".avi")


def get_video_property(capture, prop_id: int, default: float) -> float:
    value = capture.get(prop_id)
    if value and value > 0:
        return value
    return default


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def codec_candidates(output_path: Path, fourcc_text: str | None) -> list[str]:
    if fourcc_text:
        return [fourcc_text]
    if output_path.suffix.lower() == ".mp4":
        return ["avc1", "H264", "mp4v"]
    return ["XVID", "MJPG", "DIVX"]


def create_video_writer(
    output_path: Path,
    fps: float,
    size: tuple[int, int],
    codecs: list[str],
):
    errors = []
    for codec in codecs:
        if len(codec) != 4:
            errors.append(f"{codec}: FourCC must be exactly 4 characters")
            continue

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
            True,
        )
        if writer.isOpened():
            return writer, codec

        writer.release()
        errors.append(f"{codec}: writer did not open")

    raise RuntimeError("Could not create output video. Tried " + ", ".join(errors))


def validate_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Output video was created but cannot be opened: {path}")

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Output video was created but no frame can be read: {path}")
    finally:
        capture.release()


class ProgressBar:
    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.current = 0
        self.started_at = time.monotonic()
        self.last_draw_at = 0.0
        self.tqdm_bar = None

        if tqdm is not None:
            self.tqdm_bar = tqdm(total=total or None, unit="frame", desc=label)

    def update(self, count: int = 1) -> None:
        self.current += count
        if self.tqdm_bar is not None:
            self.tqdm_bar.update(count)
            return

        now = time.monotonic()
        if now - self.last_draw_at < 0.1:
            return

        self.last_draw_at = now
        elapsed = max(now - self.started_at, 0.001)
        fps = self.current / elapsed

        if self.total > 0:
            percent = min(self.current / self.total, 1.0)
            filled = int(percent * 30)
            bar = "#" * filled + "-" * (30 - filled)
            message = (
                f"\r{self.label}: [{bar}] {percent * 100:6.2f}% "
                f"({self.current}/{self.total} frames, {fps:.1f} fps)"
            )
        else:
            message = f"\r{self.label}: {self.current} frames, {fps:.1f} fps"

        print(message, end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            return

        self.update(0)
        print(file=sys.stderr)


def convert_webm_to_mp4(
    input_path: Path,
    output_path: Path,
    *,
    force: bool,
    fourcc_text: str | None,
    frame_step: int,
) -> tuple[int, str]:
    if input_path.suffix.lower() != ".webm":
        raise ValueError(f"Input is not a .webm file: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path} (use --force to overwrite)"
        )
    if frame_step <= 0:
        raise ValueError("--frame-step must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    temp_output_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    source_fps = get_video_property(capture, cv2.CAP_PROP_FPS, 30.0)
    output_fps = max(source_fps / frame_step, 1.0)
    source_width = int(get_video_property(capture, cv2.CAP_PROP_FRAME_WIDTH, 0))
    source_height = int(get_video_property(capture, cv2.CAP_PROP_FRAME_HEIGHT, 0))
    width = make_even(source_width)
    height = make_even(source_height)
    total_frames = int(get_video_property(capture, cv2.CAP_PROP_FRAME_COUNT, 0))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video size: {input_path}")

    try:
        writer, selected_codec = create_video_writer(
            temp_output_path,
            output_fps,
            (width, height),
            codec_candidates(output_path, fourcc_text),
        )
    except RuntimeError:
        capture.release()
        temp_output_path.unlink(missing_ok=True)
        raise

    frame_count = 0
    written_count = 0
    target_frames = (total_frames + frame_step - 1) // frame_step if total_frames else 0
    progress = ProgressBar(target_frames, input_path.name)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_write = frame_count % frame_step == 0
            frame_count += 1
            if not should_write:
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
            written_count += 1
            progress.update()
    finally:
        progress.close()
        capture.release()
        writer.release()

    try:
        if written_count == 0:
            raise RuntimeError(f"No frames were read from input video: {input_path}")

        validate_video(temp_output_path)
        temp_output_path.replace(output_path)
    except RuntimeError:
        temp_output_path.unlink(missing_ok=True)
        raise

    return written_count, selected_codec


def main() -> int:
    args = parse_args()

    if cv2 is None:
        print("Error: OpenCV is not installed. Install it with: pip install opencv-python", file=sys.stderr)
        return 1

    if args.output and len(args.inputs) != 1:
        print("Error: --output can only be used with one input file.", file=sys.stderr)
        return 1

    for raw_input in args.inputs:
        input_path = Path(raw_input).expanduser()
        output_path = output_path_for(input_path, args.output)
        try:
            frame_count, selected_codec = convert_webm_to_mp4(
                input_path,
                output_path,
                force=args.force,
                fourcc_text=args.fourcc,
                frame_step=args.frame_step,
            )
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(
            f"Wrote {frame_count} frames with {selected_codec}: "
            f"{input_path} -> {output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
