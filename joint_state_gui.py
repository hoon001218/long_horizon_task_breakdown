"""Tkinter GUI publisher for Franka JointState commands.

Run after sourcing a ROS 2 environment:

    python3 joint_state_gui.py

Then run Isaac Sim in another terminal:

    ./isaac-sim/python.sh world.py

The GUI publishes sensor_msgs/msg/JointState to /joint_states. The joint names
match the Franka Panda USD used in world.py.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float
    upper: float
    home: float
    resolution: float = 0.001


FRANKA_JOINTS = [
    JointSpec("panda_joint1", -2.8973, 2.8973, 0.0),
    JointSpec("panda_joint2", -1.7628, 1.7628, -0.785),
    JointSpec("panda_joint3", -2.8973, 2.8973, 0.0),
    JointSpec("panda_joint4", -3.0718, -0.0698, -2.356),
    JointSpec("panda_joint5", -2.8973, 2.8973, 0.0),
    JointSpec("panda_joint6", -0.0175, 3.7525, 1.571),
    JointSpec("panda_joint7", -2.8973, 2.8973, 0.785),
    JointSpec("panda_finger_joint1", 0.0, 0.04, 0.04, 0.0005),
    JointSpec("panda_finger_joint2", 0.0, 0.04, 0.04, 0.0005),
]

PRESETS = {
    "home": [joint.home for joint in FRANKA_JOINTS],
    "zero": [0.0, 0.0, 0.0, -0.0698, 0.0, 0.0, 0.0, 0.04, 0.04],
    "ready": [0.0, -0.6, 0.0, -2.2, 0.0, 1.8, 0.8, 0.04, 0.04],
    "gripper_open": [None, None, None, None, None, None, None, 0.04, 0.04],
    "gripper_closed": [None, None, None, None, None, None, None, 0.0, 0.0],
}


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class JointStatePublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("franka_joint_state_gui")
        self.publisher = self.create_publisher(JointState, topic, 10)

    def publish_positions(self, names: list[str], positions: list[float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = names
        message.position = positions
        self.publisher.publish(message)


class JointStateGui:
    def __init__(
        self, root: tk.Tk, node: JointStatePublisher, topic: str, publish_hz: float
    ) -> None:
        self.root = root
        self.node = node
        self.topic = topic
        self.publish_hz = max(1.0, publish_hz)
        self.publish_enabled = tk.BooleanVar(value=True)
        self.joint_vars = [tk.DoubleVar(value=joint.home) for joint in FRANKA_JOINTS]
        self.value_labels: list[ttk.Label] = []

        self.root.title("Franka JointState Publisher")
        self.root.minsize(820, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build()
        self._publish_loop()

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        title = ttk.Label(
            main, text="Franka JointState Publisher", font=("", 15, "bold")
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        topic_label = ttk.Label(
            main, text=f"Topic: {self.topic}   Rate: {self.publish_hz:.1f} Hz"
        )
        topic_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        for index, joint in enumerate(FRANKA_JOINTS, start=2):
            label = ttk.Label(main, text=joint.name, width=22)
            label.grid(row=index, column=0, sticky="w", padx=(0, 8), pady=3)

            scale = tk.Scale(
                main,
                from_=joint.lower,
                to=joint.upper,
                resolution=joint.resolution,
                orient=tk.HORIZONTAL,
                showvalue=False,
                variable=self.joint_vars[index - 2],
                command=lambda _value, idx=index - 2: self._update_value_label(idx),
            )
            scale.grid(row=index, column=1, sticky="ew", pady=3)

            value_label = ttk.Label(main, text="", width=12, anchor="e")
            value_label.grid(row=index, column=2, sticky="e", padx=(8, 0), pady=3)
            self.value_labels.append(value_label)
            self._update_value_label(index - 2)

        controls = ttk.Frame(main)
        controls.grid(
            row=len(FRANKA_JOINTS) + 2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(14, 0),
        )
        controls.columnconfigure(6, weight=1)

        ttk.Checkbutton(controls, text="Publish", variable=self.publish_enabled).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(
            controls, text="Home", command=lambda: self._apply_preset("home")
        ).grid(row=0, column=1, padx=4)
        ttk.Button(
            controls, text="Zero", command=lambda: self._apply_preset("zero")
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            controls, text="Ready", command=lambda: self._apply_preset("ready")
        ).grid(row=0, column=3, padx=4)
        ttk.Button(
            controls, text="Open", command=lambda: self._apply_preset("gripper_open")
        ).grid(row=0, column=4, padx=4)
        ttk.Button(
            controls, text="Close", command=lambda: self._apply_preset("gripper_closed")
        ).grid(row=0, column=5, padx=4)

        self.status = ttk.Label(controls, text="Publishing", anchor="e")
        self.status.grid(row=0, column=6, sticky="e")

    def _update_value_label(self, index: int) -> None:
        value = self.joint_vars[index].get()
        self.value_labels[index].configure(text=f"{value:+.4f}")

    def _positions(self) -> list[float]:
        positions = []
        for joint, variable in zip(FRANKA_JOINTS, self.joint_vars):
            positions.append(clamp(variable.get(), joint.lower, joint.upper))
        return positions

    def _apply_preset(self, name: str) -> None:
        for index, value in enumerate(PRESETS[name]):
            if value is None:
                continue
            joint = FRANKA_JOINTS[index]
            self.joint_vars[index].set(clamp(value, joint.lower, joint.upper))
            self._update_value_label(index)
        self._publish_once()

    def _publish_once(self) -> None:
        if not self.publish_enabled.get():
            self.status.configure(text="Paused")
            return

        names = [joint.name for joint in FRANKA_JOINTS]
        self.node.publish_positions(names, self._positions())
        self.status.configure(text="Publishing")

    def _publish_loop(self) -> None:
        self._publish_once()
        rclpy.spin_once(self.node, timeout_sec=0.0)
        interval_ms = int(1000.0 / self.publish_hz)
        self.root.after(interval_ms, self._publish_loop)

    def close(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI JointState publisher for the Isaac Sim Franka scene."
    )
    parser.add_argument(
        "--topic",
        default="/joint_states",
        help="JointState topic to publish. Default: /joint_states",
    )
    parser.add_argument(
        "--rate", type=float, default=30.0, help="Publish rate in Hz. Default: 30"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()

    node = JointStatePublisher(args.topic)
    root = tk.Tk()
    JointStateGui(root, node, args.topic, args.rate)
    root.mainloop()


if __name__ == "__main__":
    main()
