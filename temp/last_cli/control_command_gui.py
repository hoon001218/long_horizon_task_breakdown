"""Tkinter GUI client for testing the Auto-mode ControlCommand service.

Run after building and sourcing the workspace that contains custom_msgs:

    source install/setup.bash
    /usr/bin/python3 control_command_gui.py
"""

import queue
import threading
import tkinter as tk
import argparse
from dataclasses import dataclass
from tkinter import messagebox, ttk

import rclpy
from custom_msgs.srv import ControlCommand
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

MARKER_TOPIC = "/world/object_markers"
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}
TABLE_CENTER_TARGET = (0.6, 0.0, 0.46)
GOAL_TARGETS = {
    "left": {
        "label": "red_goal",
        "position": (0.272, 0.228, 0.466),
    },
    "right": {
        "label": "blue_goal",
        "position": (0.928, -0.228, 0.466),
    },
}

# End-effector orientation is fixed to be vertical to the table.
# This quaternion makes the EEF local z-axis point along world -Z.
VERTICAL_EEF_ORIENTATION_XYZW = (1.0, 0.0, 0.0, 0.0)

GUI_REFRESH_MS = 150
SERVICE_WAIT_TIMEOUT_SEC = 1.0

ACTION_MOVING = "Moving"
ACTION_CENTERING = "Centering"
ACTION_PLACING = "Placing"
ACTION_GRIP = "Grip"
ACTION_RELEASE = "Release"
ACTION_HOMING = "Homing"


@dataclass


class MarkerTarget:
    key: tuple[str, int]
    label: str
    pose: Pose


class ControlCommandNode(Node):
    def __init__(self, robot_id: str) -> None:
        super().__init__(f"control_command_gui_{robot_id}")
        self.robot_id = robot_id
        self.control_service_topic = CONTROL_SERVICE_TOPICS[robot_id]
        self._targets_lock = threading.Lock()
        self._targets: dict[tuple[str, int], MarkerTarget] = {}
        self.events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()

        self.create_subscription(MarkerArray, MARKER_TOPIC, self._marker_callback, 10)
        self.client = self.create_client(ControlCommand, self.control_service_topic)

    def _marker_callback(self, message: MarkerArray) -> None:
        targets: dict[tuple[str, int], MarkerTarget] = {}
        for marker in message.markers:
            if marker.ns in {"table", "red_goal", "blue_goal"}:
                continue
            key = (marker.ns, marker.id)
            label = (
                f"{marker.ns}[{marker.id}]  "
                f"x={marker.pose.position.x:.3f}, "
                f"y={marker.pose.position.y:.3f}, "
                f"z={marker.pose.position.z:.3f}"
            )
            targets[key] = MarkerTarget(key=key, label=label, pose=marker.pose)

        with self._targets_lock:
            self._targets = targets

    def get_targets(self) -> list[MarkerTarget]:
        with self._targets_lock:
            return sorted(self._targets.values(), key=lambda target: target.label)

    def send_command(self, action: str, target: MarkerTarget | None = None) -> None:
        if not self.client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            self.events.put(
                ("error", f"Service not available: {self.control_service_topic}")
            )
            return

        request = ControlCommand.Request()
        request.action = action
        if target is not None:
            request.target_pose.position = Point(
                x=target.pose.position.x,
                y=target.pose.position.y,
                z=target.pose.position.z
                + 0.0,  # Lift the target slightly above the table
            )
        request.target_pose.orientation.x = VERTICAL_EEF_ORIENTATION_XYZW[0]
        request.target_pose.orientation.y = VERTICAL_EEF_ORIENTATION_XYZW[1]
        request.target_pose.orientation.z = VERTICAL_EEF_ORIENTATION_XYZW[2]
        request.target_pose.orientation.w = VERTICAL_EEF_ORIENTATION_XYZW[3]

        future = self.client.call_async(request)
        future.add_done_callback(lambda done: self._service_done(action, done))
        self.events.put(("info", f"Sent {action}"))

    def _service_done(self, action: str, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.events.put(("error", f"{action} failed: {exc}"))
            return

        level = "info" if response.success else "error"
        self.events.put((level, f"{action}: {response.message}"))


class ControlCommandGui:
    def __init__(self, root: tk.Tk, node: ControlCommandNode) -> None:
        self.root = root
        self.node = node
        self.targets: list[MarkerTarget] = []
        self.targets_by_label: dict[str, MarkerTarget] = {}
        self.selected_key: tuple[str, int] | None = None
        self.center_target = self.make_center_target()
        self.goal_target = self.make_goal_target()

        self.root.title(f"ControlCommand GUI - {node.robot_id}")
        self.root.geometry("720x280")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.target_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Waiting for MarkerArray...")

        main = ttk.Frame(root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="Target").grid(row=0, column=0, sticky="w")
        self.target_combo = ttk.Combobox(
            main,
            textvariable=self.target_var,
            state="readonly",
            height=10,
        )
        self.target_combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        self.target_combo.bind("<<ComboboxSelected>>", self.on_target_selected)

        button_bar = ttk.Frame(main)
        button_bar.grid(row=2, column=0, sticky="ew")
        for index in range(6):
            button_bar.columnconfigure(index, weight=1)

        ttk.Button(button_bar, text="Move", command=self.move).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_bar, text="Centering", command=self.centering).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(button_bar, text="Place", command=self.place_goal).grid(
            row=0, column=2, sticky="ew", padx=6
        )
        ttk.Button(
            button_bar, text="Grip", command=lambda: self.send_simple(ACTION_GRIP)
        ).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(
            button_bar,
            text="Release",
            command=lambda: self.send_simple(ACTION_RELEASE),
        ).grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Button(
            button_bar, text="Home", command=lambda: self.send_simple(ACTION_HOMING)
        ).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        ttk.Separator(main).grid(row=3, column=0, sticky="ew", pady=14)
        ttk.Label(main, textvariable=self.status_var).grid(row=4, column=0, sticky="w")

        self.refresh()

    def selected_target(self) -> MarkerTarget | None:
        target = self.targets_by_label.get(self.target_var.get())
        if target is not None:
            self.selected_key = target.key
        return target

    def on_target_selected(self, _event=None) -> None:
        self.selected_target()

    def move(self) -> None:
        target = self.selected_target()
        if target is None:
            messagebox.showwarning("Move", "Select a MarkerArray target first.")
            return
        self.node.send_command(ACTION_MOVING, target)

    def centering(self) -> None:
        self.node.send_command(ACTION_CENTERING, self.center_target)

    def place_goal(self) -> None:
        self.node.send_command(ACTION_PLACING, self.goal_target)

    def send_simple(self, action: str) -> None:
        self.node.send_command(action)

    def refresh(self) -> None:
        self.refresh_targets()
        self.refresh_events()
        self.root.after(GUI_REFRESH_MS, self.refresh)

    def refresh_targets(self) -> None:
        previous_key = self.selected_key
        self.targets = self.node.get_targets()
        labels = [target.label for target in self.targets]
        self.targets_by_label = {target.label: target for target in self.targets}
        self.target_combo["values"] = labels

        selected_target = next(
            (target for target in self.targets if target.key == previous_key),
            None,
        )
        if selected_target is not None:
            self.target_var.set(selected_target.label)
        elif labels:
            self.target_var.set(labels[0])
            self.selected_key = self.targets[0].key
        else:
            self.target_var.set("")
            self.selected_key = None

        if labels and self.status_var.get() == "Waiting for MarkerArray...":
            self.status_var.set(f"{len(labels)} targets received.")

    def make_center_target(self) -> MarkerTarget:
        center_pose = Pose()
        center_pose.position = Point(
            x=TABLE_CENTER_TARGET[0],
            y=TABLE_CENTER_TARGET[1],
            z=TABLE_CENTER_TARGET[2],
        )
        center_target = MarkerTarget(
            key=("table_center", -1),
            label=(
                "table_center  "
                f"x={TABLE_CENTER_TARGET[0]:.3f}, "
                f"y={TABLE_CENTER_TARGET[1]:.3f}, "
                f"z={TABLE_CENTER_TARGET[2]:.3f}"
            ),
            pose=center_pose,
        )
        return center_target

    def make_goal_target(self) -> MarkerTarget:
        goal = GOAL_TARGETS[self.node.robot_id]
        position = goal["position"]
        goal_pose = Pose()
        goal_pose.position = Point(x=position[0], y=position[1], z=position[2])
        return MarkerTarget(
            key=(goal["label"], -2),
            label=(
                f"{goal['label']}  "
                f"x={position[0]:.3f}, "
                f"y={position[1]:.3f}, "
                f"z={position[2]:.3f}"
            ),
            pose=goal_pose,
        )

    def refresh_events(self) -> None:
        while True:
            try:
                level, text = self.node.events.get_nowait()
            except queue.Empty:
                return
            prefix = "Error: " if level == "error" else ""
            self.status_var.set(prefix + text)

    def close(self) -> None:
        self.root.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI client for one Franka ControlCommand service."
    )
    parser.add_argument(
        "--robot",
        choices=sorted(CONTROL_SERVICE_TOPICS),
        default="left",
        help="Franka robot to control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ControlCommandNode(args.robot)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    ControlCommandGui(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
