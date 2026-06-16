"""Isaac Sim 5.0 standalone scene synced through ROS-TCP-Endpoint.

Run this file with Isaac Sim's bundled Python, for example:

    ./isaac-sim/python.sh world.py

The script creates two fixed Franka robots, a cuboid table, and randomized small
objects on the table. Isaac Sim connects as a TCP client to an already-running
ROS-TCP-Endpoint server and:

* Manual mode subscribes to ``sensor_msgs/msg/JointState`` on ``/joint_states``;
* Auto mode serves ``custom_msgs/srv/ControlCommand`` and publishes ``/joint_states``;
* both modes publish ``visualization_msgs/msg/MarkerArray`` and a top-view RGB image.
"""

from isaacsim import SimulationApp

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass

# Object count controls.
NUM_CUBES = 5
NUM_CAPSULES = 0
NUM_SPHERES = 0
OBJECT_SIZE_SCALE = 0.5
OBJECT_SPAWN_DELAY_SEC = 1.0

MARKER_TOPIC = "/world/object_markers"
MARKER_FRAME_ID = "world"
MARKER_PUBLISH_HZ = 15.0
RGB_CAMERA_TOPIC = "/world/top_camera/image_raw"
RGB_CAMERA_PUBLISH_HZ = 8.0
RGB_CAMERA_RESOLUTION = (1024, 768)
CAMERA_POSE_TOPIC = "/world/top_camera/pose"
ROS_TCP_HOST = "127.0.0.1"
ROS_TCP_PORT = 10000
JOINT_STATE_TOPICS = {
    "left": "/franka_left/joint_states",
    "right": "/franka_right/joint_states",
}
ROBOT_POSE_TOPICS = {
    "left": "/franka_left/pose",
    "right": "/franka_right/pose",
}
EEF_POSE_TOPICS = {
    "left": "/franka_left/end_effector_pose",
    "right": "/franka_right/end_effector_pose",
}
JOINT_STATE_PUBLISH_HZ = 30.0
CONTROL_SERVICE_TOPICS = {
    "left": "/franka_left/control_command",
    "right": "/franka_right/control_command",
}
CONTROL_SERVICE_TYPE = "custom_msgs/ControlCommand"
CONTROL_SERVICE_TIMEOUT = 5.0
ROS_TCP_RECONNECT_PERIOD_SEC = 3.0

ACTION_MOVING = "Moving"
ACTION_CENTERING = "Centering"
ACTION_PLACING = "Placing"
ACTION_GRIP = "Grip"
ACTION_REALEASE = "Release"
ACTION_HOMING = "Homing"

CONTROL_MODE = "auto"  # "manual" or "auto"
HEADLESS = False
RANDOM_SEED = None
TEST_STEPS = 0

HOME_JOINT_POSITIONS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
GRIPPER_OPEN_POSITIONS = [0.04, 0.04]
GRIPPER_CLOSED_POSITIONS = [0.0, 0.0]
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
AUTO_MOVE_POSITION_TOLERANCE = 0.008
AUTO_MOVE_ORIENTATION_TOLERANCE = 0.07
AUTO_MOVE_DWELL_SEC = 0.5

OBJECT_COLORS = {
    "red": (0.95, 0.05, 0.05),
    "blue": (0.05, 0.25, 0.95),
}

TOP_CAMERA_PRIM_PATH = "/World/TopCamera"
TOP_CAMERA_POSITION = (0.6, 0.0, 1.18)
TOP_CAMERA_EULER_DEGREES = (0.0, 90.0, 0.0)
TOP_CAMERA_FOCAL_LENGTH = 12.0

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": HEADLESS})

import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import (
    DynamicCapsule,
    DynamicCuboid,
    DynamicSphere,
    FixedCuboid,
)
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils import viewports
from isaacsim.core.utils.numpy.rotations import (
    euler_angles_to_quats,
    rot_matrices_to_quats,
)
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.kinematics_solver import (
    KinematicsSolver as FrankaKinematicsSolver,
)
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux

FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
ROBOT_CONFIGS = {
    "left": {
        "prim_path": "/World/FrankaLeft",
        "name": "franka_left",
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
    },
    "right": {
        "prim_path": "/World/FrankaRight",
        "name": "franka_right",
        "position": np.array([1.2, 0.0, 0.0]),
        "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
    },
}

TABLE_DIMS = np.array([0.9, 0.7, 0.08])
TABLE_CENTER = np.array([0.6, 0.0, 0.38])
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_DIMS[2] / 2.0
TABLE_RIM_THICKNESS = 0.016
TABLE_RIM_HEIGHT = 0.0165
TABLE_RIM_COLOR = np.array([0.38, 0.30, 0.22])
GOAL_SIDE_LENGTH = 0.200
GOAL_DIMS = np.array([GOAL_SIDE_LENGTH, GOAL_SIDE_LENGTH, 0.008])
GOAL_RIM_THICKNESS = 0.012
GOAL_RIM_HEIGHT = 0.0105
GOAL_TARGET_Z = TABLE_TOP_Z + GOAL_RIM_HEIGHT + 0.035
GOAL_CORNER_MARGIN = 0.045
GOAL_ZONES = {
    "left": {
        "name": "red_goal",
        "center": np.array(
            [
                TABLE_CENTER[0]
                - TABLE_DIMS[0] / 2.0
                + GOAL_CORNER_MARGIN
                + GOAL_SIDE_LENGTH / 2.0,
                TABLE_CENTER[1]
                + TABLE_DIMS[1] / 2.0
                - GOAL_CORNER_MARGIN
                - GOAL_SIDE_LENGTH / 2.0,
                TABLE_TOP_Z + GOAL_DIMS[2] / 2.0,
            ]
        ),
        "color": np.array(OBJECT_COLORS["red"]),
    },
    "right": {
        "name": "blue_goal",
        "center": np.array(
            [
                TABLE_CENTER[0]
                + TABLE_DIMS[0] / 2.0
                - GOAL_CORNER_MARGIN
                - GOAL_SIDE_LENGTH / 2.0,
                TABLE_CENTER[1]
                - TABLE_DIMS[1] / 2.0
                + GOAL_CORNER_MARGIN
                + GOAL_SIDE_LENGTH / 2.0,
                TABLE_TOP_Z + GOAL_DIMS[2] / 2.0,
            ]
        ),
        "color": np.array(OBJECT_COLORS["blue"]),
    },
}
OBJECT_LINEAR_DAMPING = 1.0

ROBOT_WORKSPACE_X_RANGE = (0.32, 0.84)
ROBOT_WORKSPACE_Y_RANGE = (-0.26, 0.26)
MIN_OBJECT_CLEARANCE = 0.035
MAX_SPAWN_ATTEMPTS = 1000

TABLE_STATIC_FRICTION = 2.2
TABLE_DYNAMIC_FRICTION = 1.8
OBJECT_STATIC_FRICTION = 1.8
OBJECT_DYNAMIC_FRICTION = 1.4
GROUND_STATIC_FRICTION = 1.4
GROUND_DYNAMIC_FRICTION = 1.2
USE_DEFAULT_FRICTION = False


@dataclass
class SpawnedObject:
    name: str
    shape: str
    prim: object
    color: np.ndarray
    marker_scale: np.ndarray


@dataclass
class JointStateCommand:
    names: list[str]
    positions: list[float]
    velocities: list[float]
    efforts: list[float]


@dataclass
class PoseCommand:
    position: np.ndarray
    orientation_xyzw: np.ndarray


@dataclass
class MotionPhase:
    position: np.ndarray
    orientation_wxyz: np.ndarray | None
    dwell_after: float = 0.0
    apply_home_on_arrival: bool = False


@dataclass
class ControlCommandRequest:
    action: str
    target_pose: PoseCommand


@dataclass
class ControlCommandResponse:
    success: bool
    message: str


@dataclass
class PendingServiceRequest:
    service_topic: str
    srv_id: int
    request: ControlCommandRequest
    done_event: threading.Event
    response: ControlCommandResponse | None = None


@dataclass
class SpawnSpec:
    shape: str
    size: float
    radius: float
    height: float
    footprint_radius: float
    color: np.ndarray
    orientation: np.ndarray
    z: float


class RosTopic:
    @staticmethod
    def normalize(raw_topic_name: str) -> str:
        if not raw_topic_name:
            raise ValueError("ROS topic name must not be empty.")
        return (
            raw_topic_name if raw_topic_name.startswith("/") else f"/{raw_topic_name}"
        )


class QuaternionUtils:
    @staticmethod
    def random_yaw(rng: np.random.Generator) -> np.ndarray:
        yaw = float(rng.uniform(-np.pi, np.pi))
        half_yaw = yaw * 0.5
        return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])

    @staticmethod
    def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ]
        )

    @staticmethod
    def random_horizontal_capsule(rng: np.random.Generator) -> np.ndarray:
        yaw = QuaternionUtils.random_yaw(rng)
        pitch_90 = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
        return QuaternionUtils.multiply(yaw, pitch_90)

    @staticmethod
    def wxyz_to_xyzw(orientation: np.ndarray) -> tuple[float, float, float, float]:
        return (
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
            float(orientation[0]),
        )

    @staticmethod
    def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-6:
            return None
        x, y, z, w = quaternion / norm
        return np.array([w, x, y, z], dtype=np.float64)

    @staticmethod
    def angular_distance_wxyz(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = left / max(float(np.linalg.norm(left)), 1.0e-9)
        right_norm = right / max(float(np.linalg.norm(right)), 1.0e-9)
        dot = float(np.clip(abs(np.dot(left_norm, right_norm)), -1.0, 1.0))
        return 2.0 * float(np.arccos(dot))


class CustomScene:
    def __init__(self, seed: int | None) -> None:
        self.seed = seed
        self.world = World(stage_units_in_meters=1.0)
        self.frankas: dict[str, Robot] = {}
        self.top_camera: Camera | None = None
        self.object_material: PhysicsMaterial | None = None
        self.spawned_objects: list[SpawnedObject] = []

    def build(self, assets_root_path: str) -> None:
        self.add_ground()
        self.frankas = self.add_frankas(assets_root_path)
        self.add_table()
        self.add_lighting()
        self.top_camera = self.add_top_camera()

    def initialize_sensors(self) -> None:
        if self.top_camera is not None:
            self.top_camera.initialize()
            self.apply_top_camera_lens()

    def add_ground(self) -> None:
        if USE_DEFAULT_FRICTION:
            self.world.scene.add_default_ground_plane()
            return
        self.world.scene.add_default_ground_plane(
            static_friction=GROUND_STATIC_FRICTION,
            dynamic_friction=GROUND_DYNAMIC_FRICTION,
            restitution=0.0,
        )

    @property
    def franka(self) -> Robot | None:
        return self.frankas.get("left")

    def add_frankas(self, assets_root_path: str) -> dict[str, Robot]:
        robots: dict[str, Robot] = {}
        for robot_id, config in ROBOT_CONFIGS.items():
            robot_prim = add_reference_to_stage(
                usd_path=assets_root_path + FRANKA_USD_PATH,
                prim_path=config["prim_path"],
            )
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")
            robots[robot_id] = self.world.scene.add(
                Robot(
                    prim_path=config["prim_path"],
                    name=config["name"],
                    position=config["position"],
                    orientation=config["orientation"],
                )
            )
            robots[robot_id].set_world_pose(
                position=config["position"], orientation=config["orientation"]
            )
        return robots

    def create_physics_material(
        self, name: str, static_friction: float, dynamic_friction: float
    ) -> PhysicsMaterial:
        return PhysicsMaterial(
            prim_path=f"/World/PhysicsMaterials/{name}",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
        )

    def table_limited_workspace(
        self, footprint_radius: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        spawn_dims = TABLE_DIMS[:2] * 0.8
        table_x_min = TABLE_CENTER[0] - spawn_dims[0] / 2.0 + footprint_radius
        table_x_max = TABLE_CENTER[0] + spawn_dims[0] / 2.0 - footprint_radius
        table_y_min = TABLE_CENTER[1] - spawn_dims[1] / 2.0 + footprint_radius
        table_y_max = TABLE_CENTER[1] + spawn_dims[1] / 2.0 - footprint_radius
        x_range = (
            table_x_min,
            table_x_max,
        )
        y_range = (
            table_y_min,
            table_y_max,
        )
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise RuntimeError(
                "Object footprint is too large for the configured robot workspace and table bounds."
            )
        return x_range, y_range

    def sample_non_overlapping_xy(
        self,
        rng: np.random.Generator,
        footprint_radius: float,
        occupied: list[tuple[np.ndarray, float]],
    ) -> np.ndarray:
        x_range, y_range = self.table_limited_workspace(footprint_radius)
        for _ in range(MAX_SPAWN_ATTEMPTS):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            if self.overlaps_goal_zone(xy, footprint_radius):
                continue
            if all(
                np.linalg.norm(xy - center)
                >= footprint_radius + other_radius + MIN_OBJECT_CLEARANCE
                for center, other_radius in occupied
            ):
                occupied.append((xy, footprint_radius))
                return xy
        raise RuntimeError(
            "Could not sample a non-overlapping object pose. Reduce object counts or sizes."
        )

    def overlaps_goal_zone(self, xy: np.ndarray, footprint_radius: float) -> bool:
        clearance = footprint_radius + MIN_OBJECT_CLEARANCE
        for zone in GOAL_ZONES.values():
            center = zone["center"][:2]
            half_extents = GOAL_DIMS[:2] / 2.0 + clearance
            if np.all(np.abs(xy - center) <= half_extents):
                return True
        return False

    def create_spawn_specs(self, rng: np.random.Generator) -> list[SpawnSpec]:
        specs: list[SpawnSpec] = []

        def object_color() -> np.ndarray:
            color_name = "red" if rng.random() < 0.5 else "blue"
            return np.array(OBJECT_COLORS[color_name], dtype=np.float64)

        for index in range(NUM_CUBES):
            size = float(rng.uniform(0.045, 0.065) * OBJECT_SIZE_SCALE)
            specs.append(
                SpawnSpec(
                    shape="cube",
                    size=size,
                    radius=0.0,
                    height=0.0,
                    footprint_radius=np.sqrt(2.0) * size / 2.0,
                    color=object_color(),
                    orientation=QuaternionUtils.random_yaw(rng),
                    z=TABLE_TOP_Z + size / 2.0 + 0.003,
                )
            )
        for index in range(NUM_CAPSULES):
            radius = float(rng.uniform(0.020, 0.027) * OBJECT_SIZE_SCALE)
            height = float(rng.uniform(0.055, 0.085) * OBJECT_SIZE_SCALE)
            specs.append(
                SpawnSpec(
                    shape="capsule",
                    size=0.0,
                    radius=radius,
                    height=height,
                    footprint_radius=height / 2.0 + radius,
                    color=object_color(),
                    orientation=QuaternionUtils.random_horizontal_capsule(rng),
                    z=TABLE_TOP_Z + radius + 0.004,
                )
            )
        for index in range(NUM_SPHERES):
            radius = float(rng.uniform(0.028, 0.04) * OBJECT_SIZE_SCALE)
            specs.append(
                SpawnSpec(
                    shape="sphere",
                    size=0.0,
                    radius=radius,
                    height=0.0,
                    footprint_radius=radius,
                    color=object_color(),
                    orientation=QuaternionUtils.random_yaw(rng),
                    z=TABLE_TOP_Z + radius + 0.004,
                )
            )
        rng.shuffle(specs)
        return specs

    def add_table(self) -> None:
        table_material = None
        self.object_material = None
        if not USE_DEFAULT_FRICTION:
            table_material = self.create_physics_material(
                "table_high_friction", TABLE_STATIC_FRICTION, TABLE_DYNAMIC_FRICTION
            )
            self.object_material = self.create_physics_material(
                "object_high_friction", OBJECT_STATIC_FRICTION, OBJECT_DYNAMIC_FRICTION
            )

        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=TABLE_CENTER,
                scale=TABLE_DIMS,
                size=1.0,
                color=np.array([0.55, 0.42, 0.30]),
                **(
                    {}
                    if table_material is None
                    else {"physics_material": table_material}
                ),
            )
        )
        self.add_table_rims(table_material)
        self.add_goal_zones(table_material)

    def add_table_rims(self, table_material: PhysicsMaterial | None) -> None:
        material_kwargs = (
            {} if table_material is None else {"physics_material": table_material}
        )
        rim_z = TABLE_TOP_Z + TABLE_RIM_HEIGHT / 2.0
        rim_specs = [
            (
                "x_min",
                np.array(
                    [
                        TABLE_CENTER[0]
                        - TABLE_DIMS[0] / 2.0
                        + TABLE_RIM_THICKNESS / 2.0,
                        TABLE_CENTER[1],
                        rim_z,
                    ]
                ),
                np.array([TABLE_RIM_THICKNESS, TABLE_DIMS[1], TABLE_RIM_HEIGHT]),
            ),
            (
                "x_max",
                np.array(
                    [
                        TABLE_CENTER[0]
                        + TABLE_DIMS[0] / 2.0
                        - TABLE_RIM_THICKNESS / 2.0,
                        TABLE_CENTER[1],
                        rim_z,
                    ]
                ),
                np.array([TABLE_RIM_THICKNESS, TABLE_DIMS[1], TABLE_RIM_HEIGHT]),
            ),
            (
                "y_min",
                np.array(
                    [
                        TABLE_CENTER[0],
                        TABLE_CENTER[1]
                        - TABLE_DIMS[1] / 2.0
                        + TABLE_RIM_THICKNESS / 2.0,
                        rim_z,
                    ]
                ),
                np.array([TABLE_DIMS[0], TABLE_RIM_THICKNESS, TABLE_RIM_HEIGHT]),
            ),
            (
                "y_max",
                np.array(
                    [
                        TABLE_CENTER[0],
                        TABLE_CENTER[1]
                        + TABLE_DIMS[1] / 2.0
                        - TABLE_RIM_THICKNESS / 2.0,
                        rim_z,
                    ]
                ),
                np.array([TABLE_DIMS[0], TABLE_RIM_THICKNESS, TABLE_RIM_HEIGHT]),
            ),
        ]
        for suffix, position, scale in rim_specs:
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/TableRims/{suffix}",
                    name=f"table_rim_{suffix}",
                    position=position,
                    scale=scale,
                    size=1.0,
                    color=TABLE_RIM_COLOR,
                    **material_kwargs,
                )
            )

    def add_goal_zones(self, table_material: PhysicsMaterial | None) -> None:
        material_kwargs = (
            {} if table_material is None else {"physics_material": table_material}
        )
        for robot_id, zone in GOAL_ZONES.items():
            center = zone["center"]
            name = zone["name"]
            color = zone["color"]
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/GoalZones/{name}",
                    name=name,
                    position=center,
                    scale=GOAL_DIMS,
                    size=1.0,
                    color=color,
                    **material_kwargs,
                )
            )
            self.add_goal_rims(robot_id, name, center, table_material)

    def add_goal_rims(
        self,
        robot_id: str,
        goal_name: str,
        goal_center: np.ndarray,
        table_material: PhysicsMaterial | None,
    ) -> None:
        material_kwargs = (
            {} if table_material is None else {"physics_material": table_material}
        )
        rim_z = TABLE_TOP_Z + GOAL_RIM_HEIGHT / 2.0 + GOAL_DIMS[2]
        rim_specs = [
            (
                "left",
                np.array(
                    [
                        goal_center[0] - GOAL_DIMS[0] / 2.0 + GOAL_RIM_THICKNESS / 2.0,
                        goal_center[1],
                        rim_z,
                    ]
                ),
                np.array([GOAL_RIM_THICKNESS, GOAL_DIMS[1], GOAL_RIM_HEIGHT]),
            ),
            (
                "right",
                np.array(
                    [
                        goal_center[0] + GOAL_DIMS[0] / 2.0 - GOAL_RIM_THICKNESS / 2.0,
                        goal_center[1],
                        rim_z,
                    ]
                ),
                np.array([GOAL_RIM_THICKNESS, GOAL_DIMS[1], GOAL_RIM_HEIGHT]),
            ),
            (
                "front",
                np.array(
                    [
                        goal_center[0],
                        goal_center[1] - GOAL_DIMS[1] / 2.0 + GOAL_RIM_THICKNESS / 2.0,
                        rim_z,
                    ]
                ),
                np.array([GOAL_DIMS[0], GOAL_RIM_THICKNESS, GOAL_RIM_HEIGHT]),
            ),
            (
                "back",
                np.array(
                    [
                        goal_center[0],
                        goal_center[1] + GOAL_DIMS[1] / 2.0 - GOAL_RIM_THICKNESS / 2.0,
                        rim_z,
                    ]
                ),
                np.array([GOAL_DIMS[0], GOAL_RIM_THICKNESS, GOAL_RIM_HEIGHT]),
            ),
        ]
        for suffix, position, scale in rim_specs:
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/GoalZoneRims/{goal_name}_{suffix}",
                    name=f"{robot_id}_goal_rim_{suffix}",
                    position=position,
                    scale=scale,
                    size=1.0,
                    color=TABLE_RIM_COLOR,
                    **material_kwargs,
                )
            )

    def spawn_objects(self) -> None:
        if self.spawned_objects:
            return

        rng = np.random.default_rng(self.seed)
        spawned_objects: list[SpawnedObject] = []
        occupied: list[tuple[np.ndarray, float]] = []
        shape_counts = {"cube": 0, "capsule": 0, "sphere": 0}
        for spec in self.create_spawn_specs(rng):
            shape_counts[spec.shape] += 1
            object_index = shape_counts[spec.shape]
            xy = self.sample_non_overlapping_xy(rng, spec.footprint_radius, occupied)
            position = np.array([xy[0], xy[1], spec.z])
            prim, marker_scale = self.add_object_from_spec(
                spec, object_index, position, self.object_material
            )
            spawned_objects.append(
                SpawnedObject(
                    name=f"{spec.shape}_{object_index}",
                    shape=spec.shape,
                    prim=prim,
                    color=spec.color,
                    marker_scale=marker_scale,
                )
            )
        self.spawned_objects.extend(spawned_objects)
        carb.log_info(
            f"Spawned {len(spawned_objects)} table objects after robot settling delay."
        )

    def add_lighting(self) -> None:
        stage = get_current_stage()

        softbox = UsdLux.RectLight.Define(stage, Sdf.Path("/World/TopCameraSoftbox"))
        softbox.CreateIntensityAttr(900.0)
        softbox.CreateWidthAttr(1.25)
        softbox.CreateHeightAttr(1.0)
        softbox.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(softbox.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(TOP_CAMERA_POSITION[0], TOP_CAMERA_POSITION[1], 1.08)
        )

        fill = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/ColorFillDomeLight"))
        fill.CreateIntensityAttr(350.0)
        fill.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    def add_top_camera(self) -> Camera:
        camera = self.world.scene.add(
            Camera(
                prim_path=TOP_CAMERA_PRIM_PATH,
                name="top_camera",
                frequency=int(RGB_CAMERA_PUBLISH_HZ),
                resolution=RGB_CAMERA_RESOLUTION,
                position=np.array(TOP_CAMERA_POSITION, dtype=np.float64),
                orientation=euler_angles_to_quats(
                    np.array(TOP_CAMERA_EULER_DEGREES), degrees=True
                ),
            )
        )
        self.apply_top_camera_lens()
        return camera

    def apply_top_camera_lens(self) -> None:
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(TOP_CAMERA_PRIM_PATH)
        if not prim.IsValid():
            carb.log_warn(f"Top camera prim not found: {TOP_CAMERA_PRIM_PATH}")
            return

        focal_attr = prim.GetAttribute("focalLength")
        if not focal_attr.IsValid():
            focal_attr = prim.CreateAttribute(
                "focalLength", Sdf.ValueTypeNames.Float, False
            )
        focal_attr.Set(float(TOP_CAMERA_FOCAL_LENGTH))

        clipping_attr = prim.GetAttribute("clippingRange")
        if clipping_attr.IsValid():
            clipping_attr.Set(Gf.Vec2f(0.01, 5.0))

        carb.log_info(
            f"Set {TOP_CAMERA_PRIM_PATH}.focalLength to {TOP_CAMERA_FOCAL_LENGTH}"
        )

    def add_object_from_spec(
        self,
        spec: SpawnSpec,
        object_index: int,
        position: np.ndarray,
        object_material: PhysicsMaterial | None,
    ) -> tuple[object, np.ndarray]:
        material_kwargs = (
            {} if object_material is None else {"physics_material": object_material}
        )
        if spec.shape == "cube":
            prim = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Objects/Cube_{object_index}",
                    name=f"cube_{object_index}",
                    position=position,
                    orientation=spec.orientation,
                    scale=np.array([spec.size, spec.size, spec.size]),
                    size=1.0,
                    color=spec.color,
                    **material_kwargs,
                    mass=0.05,
                )
            )
            self.apply_object_damping(prim)
            return prim, np.array([spec.size, spec.size, spec.size])
        if spec.shape == "capsule":
            prim = self.world.scene.add(
                DynamicCapsule(
                    prim_path=f"/World/Objects/Capsule_{object_index}",
                    name=f"capsule_{object_index}",
                    position=position,
                    orientation=spec.orientation,
                    radius=spec.radius,
                    height=spec.height,
                    color=spec.color,
                    **material_kwargs,
                    mass=0.04,
                )
            )
            self.apply_object_damping(prim)
            return prim, np.array([2.0 * spec.radius, 2.0 * spec.radius, spec.height])
        prim = self.world.scene.add(
            DynamicSphere(
                prim_path=f"/World/Objects/Sphere_{object_index}",
                name=f"sphere_{object_index}",
                position=position,
                orientation=spec.orientation,
                radius=spec.radius,
                color=spec.color,
                **material_kwargs,
                mass=0.04,
            )
        )
        self.apply_object_damping(prim)
        return prim, np.array([2.0 * spec.radius, 2.0 * spec.radius, 2.0 * spec.radius])

    def apply_object_damping(self, dynamic_prim: object) -> None:
        rigid_prim = dynamic_prim.prim
        if rigid_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(rigid_prim)
        else:
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        rigid_body_api.CreateLinearDampingAttr().Set(float(OBJECT_LINEAR_DAMPING))


class CdrWriter:
    """Small ROS2 CDR writer for the message subset used by this scene."""

    def __init__(self) -> None:
        self.buffer = bytearray(b"\x00\x01\x00\x00")

    def align(self, alignment: int) -> None:
        cdr_offset = len(self.buffer) - 4
        self.buffer.extend(b"\x00" * ((-cdr_offset) % alignment))

    def write_bool(self, value: bool) -> None:
        self.buffer.extend(struct.pack("<?", value))

    def write_uint8(self, value: int) -> None:
        self.buffer.extend(struct.pack("<B", value))

    def write_int32(self, value: int) -> None:
        self.align(4)
        self.buffer.extend(struct.pack("<i", value))

    def write_uint32(self, value: int) -> None:
        self.align(4)
        self.buffer.extend(struct.pack("<I", value))

    def write_float32(self, value: float) -> None:
        self.align(4)
        self.buffer.extend(struct.pack("<f", value))

    def write_float64(self, value: float) -> None:
        self.align(8)
        self.buffer.extend(struct.pack("<d", value))

    def write_string(self, value: str) -> None:
        data = value.encode("utf-8")
        self.write_uint32(len(data) + 1)
        self.buffer.extend(data)
        self.buffer.extend(b"\x00")

    def write_string_sequence(self, values: list[str]) -> None:
        self.write_uint32(len(values))
        for value in values:
            self.write_string(value)

    def write_float64_sequence(self, values: list[float]) -> None:
        self.write_uint32(len(values))
        if values:
            self.align(8)
            self.buffer.extend(struct.pack(f"<{len(values)}d", *values))

    def write_header(self, sec: int, nanosec: int, frame_id: str) -> None:
        self.write_int32(sec)
        self.write_uint32(nanosec)
        self.write_string(frame_id)

    def write_vector3(self, vector: np.ndarray) -> None:
        self.write_float64(float(vector[0]))
        self.write_float64(float(vector[1]))
        self.write_float64(float(vector[2]))

    def write_quaternion_xyzw(self, quaternion: np.ndarray) -> None:
        self.write_float64(float(quaternion[0]))
        self.write_float64(float(quaternion[1]))
        self.write_float64(float(quaternion[2]))
        self.write_float64(float(quaternion[3]))

    def write_color(self, color: np.ndarray, alpha: float) -> None:
        self.write_float32(float(color[0]))
        self.write_float32(float(color[1]))
        self.write_float32(float(color[2]))
        self.write_float32(alpha)

    def write_uint8_sequence(self, data: bytes = b"") -> None:
        self.write_uint32(len(data))
        self.buffer.extend(data)

    def to_bytes(self) -> bytes:
        return bytes(self.buffer)


class CdrReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 4 if len(data) >= 4 else 0

    def align(self, alignment: int) -> None:
        cdr_offset = self.offset - 4
        self.offset += (-cdr_offset) % alignment

    def require(self, size: int) -> None:
        if self.offset + size > len(self.data):
            raise ValueError("ROS2 CDR payload ended unexpectedly.")

    def read_int32(self) -> int:
        self.align(4)
        self.require(4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_uint32(self) -> int:
        self.align(4)
        self.require(4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_float64(self) -> float:
        self.align(8)
        self.require(8)
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return value

    def read_string(self) -> str:
        length = self.read_uint32()
        self.require(length)
        raw = self.data[self.offset : self.offset + length]
        self.offset += length
        return raw.rstrip(b"\x00").decode("utf-8")

    def read_string_sequence(self) -> list[str]:
        count = self.read_uint32()
        return [self.read_string() for _ in range(count)]

    def read_float64_sequence(self) -> list[float]:
        count = self.read_uint32()
        if count == 0:
            return []
        self.align(8)
        self.require(count * 8)
        values = list(struct.unpack_from(f"<{count}d", self.data, self.offset))
        self.offset += count * 8
        return values


class RosMessageCodec:
    @staticmethod
    def deserialize_joint_state(data: bytes) -> JointStateCommand:
        reader = CdrReader(data)
        reader.read_int32()
        reader.read_uint32()
        reader.read_string()
        names = reader.read_string_sequence()
        positions = reader.read_float64_sequence()
        velocities = reader.read_float64_sequence()
        efforts = reader.read_float64_sequence()
        return JointStateCommand(names, positions, velocities, efforts)

    @staticmethod
    def deserialize_control_command_request(data: bytes) -> ControlCommandRequest:
        reader = CdrReader(data)
        action = reader.read_string()
        position = np.array(
            [reader.read_float64(), reader.read_float64(), reader.read_float64()],
            dtype=np.float64,
        )
        orientation_xyzw = np.array(
            [
                reader.read_float64(),
                reader.read_float64(),
                reader.read_float64(),
                reader.read_float64(),
            ],
            dtype=np.float64,
        )
        return ControlCommandRequest(action, PoseCommand(position, orientation_xyzw))

    @staticmethod
    def serialize_control_command_response(response: ControlCommandResponse) -> bytes:
        writer = CdrWriter()
        writer.write_bool(response.success)
        writer.write_string(response.message)
        return writer.to_bytes()

    @staticmethod
    def serialize_joint_state(
        names: list[str],
        positions: list[float],
        velocities: list[float],
        efforts: list[float],
        frame_id: str,
    ) -> bytes:
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1_000_000_000)
        writer = CdrWriter()
        writer.write_header(sec, nanosec, frame_id)
        writer.write_string_sequence(names)
        writer.write_float64_sequence(positions)
        writer.write_float64_sequence(velocities)
        writer.write_float64_sequence(efforts)
        return writer.to_bytes()

    @staticmethod
    def serialize_pose_stamped(
        position: np.ndarray, orientation_wxyz: np.ndarray, frame_id: str
    ) -> bytes:
        qx, qy, qz, qw = QuaternionUtils.wxyz_to_xyzw(orientation_wxyz)
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1_000_000_000)
        writer = CdrWriter()
        writer.write_header(sec, nanosec, frame_id)
        writer.write_float64(float(position[0]))
        writer.write_float64(float(position[1]))
        writer.write_float64(float(position[2]))
        writer.write_float64(qx)
        writer.write_float64(qy)
        writer.write_float64(qz)
        writer.write_float64(qw)
        return writer.to_bytes()

    @staticmethod
    def serialize_rgb_image(image: np.ndarray, frame_id: str) -> bytes:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected RGB image with shape HxWx3+, got {rgb.shape}.")
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32)
            if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        height, width = rgb.shape[:2]
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1_000_000_000)
        writer = CdrWriter()
        writer.write_header(sec, nanosec, frame_id)
        writer.write_uint32(int(height))
        writer.write_uint32(int(width))
        writer.write_string("rgb8")
        writer.write_uint8(0)
        writer.write_uint32(int(width * 3))
        writer.write_uint8_sequence(rgb.tobytes())
        return writer.to_bytes()


class MarkerArrayCodec:
    @staticmethod
    def serialize(objects: list[SpawnedObject], frame_id: str) -> bytes:
        writer = CdrWriter()
        fixed_marker_count = 1 + len(GOAL_ZONES)
        writer.write_uint32(len(objects) + fixed_marker_count)
        MarkerArrayCodec.write_table_marker(writer, frame_id)
        MarkerArrayCodec.write_goal_markers(writer, frame_id)
        for marker_id, spawned_object in enumerate(objects):
            MarkerArrayCodec.write_marker(
                writer, marker_id + fixed_marker_count, frame_id, spawned_object
            )
        return writer.to_bytes()

    @staticmethod
    def write_table_marker(writer: CdrWriter, frame_id: str) -> None:
        MarkerArrayCodec.write_marker_fields(
            writer=writer,
            frame_id=frame_id,
            namespace="table",
            marker_id=0,
            marker_type=1,
            position=TABLE_CENTER,
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=TABLE_DIMS,
            color=np.array([0.55, 0.42, 0.30]),
            alpha=0.45,
        )

    @staticmethod
    def write_goal_markers(writer: CdrWriter, frame_id: str) -> None:
        for marker_id, zone in enumerate(GOAL_ZONES.values(), start=1):
            MarkerArrayCodec.write_marker_fields(
                writer=writer,
                frame_id=frame_id,
                namespace=zone["name"],
                marker_id=marker_id,
                marker_type=1,
                position=zone["center"],
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                scale=GOAL_DIMS,
                color=zone["color"],
                alpha=0.65,
            )

    @staticmethod
    def write_marker(
        writer: CdrWriter, marker_id: int, frame_id: str, spawned_object: SpawnedObject
    ) -> None:
        position, orientation = spawned_object.prim.get_world_pose()
        qx, qy, qz, qw = QuaternionUtils.wxyz_to_xyzw(orientation)
        marker_type = 1 if spawned_object.shape == "cube" else 2
        MarkerArrayCodec.write_marker_fields(
            writer=writer,
            frame_id=frame_id,
            namespace=spawned_object.name,
            marker_id=marker_id,
            marker_type=marker_type,
            position=position,
            orientation_xyzw=(qx, qy, qz, qw),
            scale=spawned_object.marker_scale,
            color=spawned_object.color,
            alpha=0.9,
        )

    @staticmethod
    def write_marker_fields(
        writer: CdrWriter,
        frame_id: str,
        namespace: str,
        marker_id: int,
        marker_type: int,
        position: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        scale: np.ndarray,
        color: np.ndarray,
        alpha: float,
    ) -> None:
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1_000_000_000)
        qx, qy, qz, qw = orientation_xyzw
        writer.write_header(sec, nanosec, frame_id)
        writer.write_string(namespace)
        writer.write_int32(marker_id)
        writer.write_int32(marker_type)
        writer.write_int32(0)
        writer.write_float64(float(position[0]))
        writer.write_float64(float(position[1]))
        writer.write_float64(float(position[2]))
        writer.write_float64(qx)
        writer.write_float64(qy)
        writer.write_float64(qz)
        writer.write_float64(qw)
        writer.write_vector3(scale)
        writer.write_color(color, alpha)
        writer.write_int32(0)
        writer.write_uint32(0)
        writer.write_bool(False)
        writer.write_uint32(0)
        writer.write_uint32(0)
        writer.write_string("")
        writer.write_header(0, 0, "")
        writer.write_string("")
        writer.write_uint8_sequence()
        writer.write_uint32(0)
        writer.write_string("")
        writer.write_string("")
        writer.write_string("")
        writer.write_uint8_sequence()
        writer.write_bool(False)


class RosTcpClientBase:
    def __init__(
        self,
        host: str,
        port: int,
        control_mode: str,
        joint_topics: dict[str, str],
        robot_pose_topics: dict[str, str],
        eef_pose_topics: dict[str, str],
        marker_topic: str,
        image_topic: str,
        camera_pose_topic: str,
        frame_id: str,
        publish_hz: float,
        image_publish_hz: float,
        joint_state_publish_hz: float,
        control_service_topics: dict[str, str],
        control_service_type: str,
        objects: list[SpawnedObject],
    ) -> None:
        self.host = host
        self.port = port
        self.control_mode = control_mode.lower()
        self.joint_topics = {
            robot_id: RosTopic.normalize(topic)
            for robot_id, topic in joint_topics.items()
        }
        self.robot_pose_topics = {
            robot_id: RosTopic.normalize(topic)
            for robot_id, topic in robot_pose_topics.items()
        }
        self.eef_pose_topics = {
            robot_id: RosTopic.normalize(topic)
            for robot_id, topic in eef_pose_topics.items()
        }
        self.joint_topic = next(iter(self.joint_topics.values()))
        self.marker_topic = RosTopic.normalize(marker_topic)
        self.image_topic = RosTopic.normalize(image_topic)
        self.camera_pose_topic = RosTopic.normalize(camera_pose_topic)
        self.control_service_topics = {
            robot_id: RosTopic.normalize(topic)
            for robot_id, topic in control_service_topics.items()
        }
        self.control_service_topic_to_robot = {
            topic: robot_id for robot_id, topic in self.control_service_topics.items()
        }
        self.control_service_type = control_service_type
        self.objects = objects
        self.frame_id = frame_id
        self.publish_period = 1.0 / max(publish_hz, 0.1)
        self.image_publish_period = 1.0 / max(image_publish_hz, 0.1)
        self.joint_state_publish_period = 1.0 / max(joint_state_publish_hz, 0.1)
        self.last_publish_time = 0.0
        self.last_image_publish_time = 0.0
        self.last_pose_publish_time = 0.0
        self.last_joint_state_publish_times = {
            robot_id: 0.0 for robot_id in self.joint_topics
        }
        self.socket: socket.socket | None = None
        self.connected = False
        self.connecting = False
        self.next_reconnect_time = 0.0
        self.send_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.service_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None
        self.latest_joint_state: JointStateCommand | None = None
        self.pending_service_request_id: int | None = None
        self.pending_service_requests: list[PendingServiceRequest] = []
        self.logged_first_publish = False
        self.logged_first_camera_image = False
        self.logged_first_joint_state = False
        self.logged_first_joint_state_publish = False
        self.logged_disconnect = False
        self.try_connect(force=True)

    @staticmethod
    def pack_packet(destination: str, payload: bytes) -> bytes:
        destination_bytes = destination.encode("utf-8")
        return (
            struct.pack("<I", len(destination_bytes))
            + destination_bytes
            + struct.pack("<I", len(payload))
            + payload
        )

    @staticmethod
    def pack_syscommand(command: str, params: dict) -> bytes:
        payload = json.dumps(params).encode("utf-8") + b"\x00"
        return RosTcpClientBase.pack_packet(command, payload)

    @staticmethod
    def recvall(sock: socket.socket, size: int) -> bytes:
        data = bytearray(size)
        view = memoryview(data)
        pos = 0
        while pos < size:
            count = sock.recv_into(view[pos:], size - pos)
            if count == 0:
                raise IOError("ROS-TCP-Endpoint closed the connection.")
            pos += count
        return bytes(data)

    def try_connect(self, force: bool = False) -> bool:
        if self.stop_event.is_set():
            return False
        if self.connected and self.socket is not None:
            return True

        now = time.monotonic()
        if not force and now < self.next_reconnect_time:
            return False
        if self.connecting:
            return False

        self.connecting = True
        self.next_reconnect_time = now + ROS_TCP_RECONNECT_PERIOD_SEC
        try:
            sock = socket.create_connection((self.host, self.port), timeout=1.0)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.socket = sock
            self.connected = True
            self.logged_disconnect = False
            self.pending_service_request_id = None

            self.register_ros_interfaces()
            self.send_syscommand(
                "__publish",
                {
                    "topic": self.marker_topic,
                    "message_name": "visualization_msgs/MarkerArray",
                    "queue_size": 10,
                    "latch": False,
                },
            )
            self.send_syscommand(
                "__publish",
                {
                    "topic": self.image_topic,
                    "message_name": "sensor_msgs/Image",
                    "queue_size": 2,
                    "latch": False,
                },
            )
            for pose_topic in self.robot_pose_topics.values():
                self.register_pose_publisher(pose_topic)
            for pose_topic in self.eef_pose_topics.values():
                self.register_pose_publisher(pose_topic)
            self.register_pose_publisher(self.camera_pose_topic)
            if not self.connected or self.socket is not sock:
                raise ConnectionError(
                    "ROS-TCP-Endpoint disconnected during registration."
                )

            self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
            self.reader_thread.start()

            carb.log_info(
                f"Connected to ROS-TCP-Endpoint at {self.host}:{self.port}; "
                f"mode={self.control_mode}, joint_topics={self.joint_topics}, "
                f"marker_topic={self.marker_topic}, image_topic={self.image_topic}"
            )
            return True
        except Exception as exc:
            self.close_socket()
            carb.log_warn(
                f"ROS-TCP-Endpoint unavailable at {self.host}:{self.port}: {exc}. "
                f"Retrying every {ROS_TCP_RECONNECT_PERIOD_SEC:.1f}s."
            )
            return False
        finally:
            self.connecting = False

    def mark_disconnected(self) -> None:
        self.connected = False
        self.pending_service_request_id = None
        self.close_socket()
        self.next_reconnect_time = time.monotonic() + ROS_TCP_RECONNECT_PERIOD_SEC

    def close_socket(self) -> None:
        sock = self.socket
        self.socket = None
        self.connected = False
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    def send_syscommand(self, command: str, params: dict) -> None:
        self.send_raw(self.pack_syscommand(command, params))

    def send_packet(self, destination: str, payload: bytes) -> None:
        self.send_raw(self.pack_packet(destination, payload))

    def register_pose_publisher(self, topic: str) -> None:
        self.send_syscommand(
            "__publish",
            {
                "topic": topic,
                "message_name": "geometry_msgs/PoseStamped",
                "queue_size": 10,
                "latch": False,
            },
        )

    def send_raw(self, packet: bytes) -> None:
        if self.socket is None and not self.try_connect():
            return
        with self.send_lock:
            if self.socket is None:
                return
            try:
                self.socket.sendall(packet)
            except Exception as exc:
                if not self.stop_event.is_set():
                    carb.log_error(f"ROS-TCP-Endpoint send failed: {exc}")
                    self.mark_disconnected()

    def reader_loop(self) -> None:
        sock = self.socket
        if sock is None:
            return
        try:
            while not self.stop_event.is_set():
                destination_length = struct.unpack("<I", self.recvall(sock, 4))[0]
                destination = (
                    self.recvall(sock, destination_length)
                    .decode("utf-8")
                    .rstrip("\x00")
                )
                payload_length = struct.unpack("<I", self.recvall(sock, 4))[0]
                payload = self.recvall(sock, payload_length)

                if self.pending_service_request_id is not None:
                    self.handle_service_payload(destination, payload)
                elif destination.startswith("__"):
                    self.log_syscommand(destination, payload)
                else:
                    self.handle_topic_message(destination, payload)
        except Exception as exc:
            if not self.stop_event.is_set() and not self.logged_disconnect:
                carb.log_error(f"ROS-TCP-Endpoint reader stopped: {exc}")
                self.logged_disconnect = True
            if not self.stop_event.is_set() and self.socket is sock:
                self.mark_disconnected()

    def log_syscommand(self, command: str, payload: bytes) -> None:
        try:
            text = payload.rstrip(b"\x00").decode("utf-8")
            data = json.loads(text) if text else {}
        except Exception:
            data = {}

        if command == "__handshake":
            carb.log_info(f"ROS-TCP-Endpoint handshake: {data}")
        elif command == "__error":
            carb.log_error(f"ROS-TCP-Endpoint error: {data}")
            text = str(data.get("text", ""))
            if "Unknown service class" in text and self.control_service_type in text:
                carb.log_error(
                    "ROS-TCP-Endpoint cannot import custom_msgs.srv.ControlCommand. "
                    "Rebuild custom_msgs, source this workspace before starting the endpoint, "
                    "and restart ROS-TCP-Endpoint."
                )
        elif command == "__warn":
            carb.log_warn(f"ROS-TCP-Endpoint warning: {data}")
        elif command == "__request":
            self.pending_service_request_id = int(data.get("srv_id", -1))

    def publish_if_due(self) -> None:
        if not self.try_connect():
            return
        now = time.monotonic()
        if now - self.last_publish_time < self.publish_period:
            return
        self.last_publish_time = now
        self.publish()

    def publish(self) -> None:
        payload = MarkerArrayCodec.serialize(self.objects, self.frame_id)
        self.send_packet(self.marker_topic, payload)
        if not self.logged_first_publish:
            carb.log_info(
                f"Published first MarkerArray with {len(self.objects)} objects through ROS-TCP-Endpoint"
            )
            self.logged_first_publish = True

    def publish_camera_if_due(self, camera: Camera | None) -> None:
        if camera is None or not self.try_connect():
            return
        now = time.monotonic()
        if now - self.last_image_publish_time < self.image_publish_period:
            return
        self.last_image_publish_time = now

        try:
            image = camera.get_rgb()
            if image is None:
                return
            payload = RosMessageCodec.serialize_rgb_image(image, self.frame_id)
        except Exception as exc:
            carb.log_warn(f"Skipping top camera publish: {exc}")
            return

        self.send_packet(self.image_topic, payload)
        if not self.logged_first_camera_image:
            height, width = np.asarray(image).shape[:2]
            carb.log_info(
                f"Published first top camera RGB image {width}x{height} on {self.image_topic}"
            )
            self.logged_first_camera_image = True

    def publish_pose_if_due(
        self,
        robots: dict[str, Robot],
        eef_poses: dict[str, tuple[np.ndarray, np.ndarray]],
        camera: Camera | None,
    ) -> None:
        if not self.try_connect():
            return
        now = time.monotonic()
        if now - self.last_pose_publish_time < self.publish_period:
            return
        self.last_pose_publish_time = now

        for robot_id, robot in robots.items():
            topic = self.robot_pose_topics.get(robot_id)
            if topic is None:
                continue
            position, orientation = robot.get_world_pose()
            payload = RosMessageCodec.serialize_pose_stamped(
                np.array(position, dtype=np.float64),
                np.array(orientation, dtype=np.float64),
                self.frame_id,
            )
            self.send_packet(topic, payload)

        for robot_id, (position, orientation) in eef_poses.items():
            topic = self.eef_pose_topics.get(robot_id)
            if topic is None:
                continue
            payload = RosMessageCodec.serialize_pose_stamped(
                position, orientation, self.frame_id
            )
            self.send_packet(topic, payload)

        if camera is not None:
            position, orientation = camera.get_world_pose()
            payload = RosMessageCodec.serialize_pose_stamped(
                np.array(position, dtype=np.float64),
                np.array(orientation, dtype=np.float64),
                self.frame_id,
            )
            self.send_packet(self.camera_pose_topic, payload)

    def publish_joint_state_if_due(self, robots: dict[str, Robot]) -> None:
        self.try_connect()
        return

    def consume_latest_joint_state(self) -> JointStateCommand | None:
        with self.state_lock:
            command = self.latest_joint_state
            self.latest_joint_state = None
        return command

    def handle_service_payload(self, destination: str, payload: bytes) -> None:
        srv_id = self.pending_service_request_id
        self.pending_service_request_id = None
        if srv_id is None:
            return
        if destination not in self.control_service_topic_to_robot:
            response = ControlCommandResponse(
                False, f"Unexpected service payload destination: {destination}"
            )
            self.send_service_response(destination, srv_id, response)
            return

        try:
            request = RosMessageCodec.deserialize_control_command_request(payload)
        except Exception as exc:
            self.send_service_response(
                destination,
                srv_id,
                ControlCommandResponse(False, f"Failed to parse request: {exc}"),
            )
            return

        pending = PendingServiceRequest(destination, srv_id, request, threading.Event())
        with self.service_lock:
            self.pending_service_requests.append(pending)

        if not pending.done_event.wait(CONTROL_SERVICE_TIMEOUT):
            pending.response = ControlCommandResponse(
                False, "ControlCommand handling timed out."
            )
        self.send_service_response(
            destination,
            srv_id,
            pending.response or ControlCommandResponse(False, "No response generated."),
        )

    def pop_service_requests(
        self, service_topic: str | None = None
    ) -> list[PendingServiceRequest]:
        normalized_topic = (
            None if service_topic is None else RosTopic.normalize(service_topic)
        )
        with self.service_lock:
            if normalized_topic is None:
                requests = self.pending_service_requests
                self.pending_service_requests = []
            else:
                requests = [
                    request
                    for request in self.pending_service_requests
                    if request.service_topic == normalized_topic
                ]
                self.pending_service_requests = [
                    request
                    for request in self.pending_service_requests
                    if request.service_topic != normalized_topic
                ]
        return requests

    def send_service_response(
        self, service_topic: str, srv_id: int, response: ControlCommandResponse
    ) -> None:
        packet = b"".join(
            [
                self.pack_syscommand("__response", {"srv_id": srv_id}),
                self.pack_packet(
                    service_topic,
                    RosMessageCodec.serialize_control_command_response(response),
                ),
            ]
        )
        self.send_raw(packet)

    def register_ros_interfaces(self) -> None:
        raise NotImplementedError

    def handle_topic_message(self, destination: str, payload: bytes) -> None:
        return

    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            self.send_packet("", b"")
        except Exception:
            pass
        self.close_socket()


class ManualRosClient(RosTcpClientBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(control_mode="manual", **kwargs)

    def register_ros_interfaces(self) -> None:
        self.send_syscommand(
            "__subscribe",
            {
                "topic": self.joint_topics["left"],
                "message_name": "sensor_msgs/JointState",
            },
        )

    def handle_topic_message(self, destination: str, payload: bytes) -> None:
        if destination != self.joint_topics["left"]:
            return
        command = RosMessageCodec.deserialize_joint_state(payload)
        with self.state_lock:
            self.latest_joint_state = command
        if not self.logged_first_joint_state:
            carb.log_info(
                f"Received first JointState from ROS-TCP-Endpoint on {self.joint_topics['left']}"
            )
            self.logged_first_joint_state = True


class AutoRosClient(RosTcpClientBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(control_mode="auto", **kwargs)

    def register_ros_interfaces(self) -> None:
        for joint_topic in self.joint_topics.values():
            self.send_syscommand(
                "__publish",
                {
                    "topic": joint_topic,
                    "message_name": "sensor_msgs/JointState",
                    "queue_size": 10,
                    "latch": False,
                },
            )
        for service_topic in self.control_service_topics.values():
            self.send_syscommand(
                "__unity_service",
                {
                    "topic": service_topic,
                    "message_name": self.control_service_type,
                },
            )

    def publish_joint_state_if_due(self, robots: dict[str, Robot]) -> None:
        if not self.try_connect():
            return
        now = time.monotonic()
        for robot_id, robot in robots.items():
            if robot_id not in self.joint_topics:
                continue
            last_publish = self.last_joint_state_publish_times.get(robot_id, 0.0)
            if now - last_publish < self.joint_state_publish_period:
                continue
            self.last_joint_state_publish_times[robot_id] = now
            positions = robot.get_joint_positions()
            velocities = robot.get_joint_velocities()
            if positions is None:
                continue
            if velocities is None:
                velocities = []
            payload = RosMessageCodec.serialize_joint_state(
                list(robot.dof_names),
                [float(value) for value in positions],
                [float(value) for value in velocities],
                [],
                self.frame_id,
            )
            self.send_packet(self.joint_topics[robot_id], payload)
            if not self.logged_first_joint_state_publish:
                carb.log_info(
                    "Published first JointState through ROS-TCP-Endpoint "
                    f"on {self.joint_topics[robot_id]}"
                )
                self.logged_first_joint_state_publish = True


class BaseController:
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.controller = robot.get_articulation_controller()

    def step(self, ros_client: RosTcpClientBase) -> None:
        raise NotImplementedError


class ManualController(BaseController):
    def __init__(self, robot: Robot) -> None:
        super().__init__(robot)
        self.unknown_joint_names: set[str] = set()

    def step(self, ros_client: RosTcpClientBase) -> None:
        command = ros_client.consume_latest_joint_state()
        self.apply(command)

    def apply(self, command: JointStateCommand | None) -> None:
        if command is None or len(command.positions) == 0:
            return

        positions = np.array(command.positions, dtype=np.float64)
        joint_indices = None

        if command.names:
            selected_positions: list[float] = []
            selected_indices: list[int] = []
            for name, position in zip(command.names, command.positions):
                try:
                    selected_indices.append(int(self.robot.get_dof_index(name)))
                    selected_positions.append(float(position))
                except Exception:
                    if name not in self.unknown_joint_names:
                        carb.log_warn(
                            f"Ignoring JointState name not found in Franka DOFs: {name}"
                        )
                        self.unknown_joint_names.add(name)

            if not selected_positions:
                return
            positions = np.array(selected_positions, dtype=np.float64)
            joint_indices = np.array(selected_indices, dtype=np.int32)

        self.controller.apply_action(
            ArticulationAction(joint_positions=positions, joint_indices=joint_indices)
        )


class AutoController(BaseController):
    def __init__(self, robot: Robot, service_topic: str) -> None:
        super().__init__(robot)
        self.service_topic = RosTopic.normalize(service_topic)
        self.kinematics_solver = FrankaKinematicsSolver(robot)
        self.gripper_joint_indices = np.array(
            [self.robot.get_dof_index(name) for name in GRIPPER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.motion_phases: list[MotionPhase] = []
        self.phase_dwell_until: float | None = None
        self.gripper_target_positions = np.array(
            GRIPPER_OPEN_POSITIONS, dtype=np.float64
        )
        self.sync_kinematics_base_pose()
        self.home_ee_position, home_ee_rotation = self.compute_home_ee_pose()
        self.home_ee_orientation = rot_matrices_to_quats(home_ee_rotation)
        self.home_applied = False

    def sync_kinematics_base_pose(self) -> None:
        base_position, base_orientation = self.robot.get_world_pose()
        self.kinematics_solver.get_kinematics_solver().set_robot_base_pose(
            np.array(base_position, dtype=np.float64),
            np.array(base_orientation, dtype=np.float64),
        )

    def compute_current_ee_pose(
        self, position_only: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        self.sync_kinematics_base_pose()
        return self.kinematics_solver.compute_end_effector_pose(
            position_only=position_only
        )

    def compute_ik_action(
        self,
        target_position: np.ndarray,
        target_orientation_wxyz: np.ndarray | None,
    ) -> tuple[ArticulationAction, bool]:
        self.sync_kinematics_base_pose()
        return self.kinematics_solver.compute_inverse_kinematics(
            target_position,
            target_orientation_wxyz,
            position_tolerance=AUTO_MOVE_POSITION_TOLERANCE,
            orientation_tolerance=(
                AUTO_MOVE_ORIENTATION_TOLERANCE
                if target_orientation_wxyz is not None
                else None
            ),
        )

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self.compute_current_ee_pose(position_only=False)
        orientation = rot_matrices_to_quats(rotation)
        return np.array(position, dtype=np.float64), np.array(
            orientation, dtype=np.float64
        )

    def step(self, ros_client: RosTcpClientBase) -> None:
        self.process_service_requests(ros_client)
        self.step_motion()

    def process_service_requests(self, ros_tcp_client: RosTcpClientBase) -> None:
        for pending in ros_tcp_client.pop_service_requests(self.service_topic):
            pending.response = self.dispatch(pending.request)
            pending.done_event.set()

    def dispatch(self, request: ControlCommandRequest) -> ControlCommandResponse:
        action = request.action.strip()
        try:
            if action == ACTION_MOVING:
                return self.move(request.target_pose, ACTION_MOVING)
            if action == ACTION_CENTERING:
                return self.move(request.target_pose, ACTION_CENTERING)
            if action == ACTION_PLACING:
                return self.move(request.target_pose, ACTION_PLACING)
            if action == ACTION_GRIP:
                return self.grip()
            if action == ACTION_REALEASE:
                return self.release()
            if action == ACTION_HOMING:
                return self.stop()
            stop_response = self.stop()
            return ControlCommandResponse(
                False,
                f"Unknown action '{request.action}'. Executed stop instead: {stop_response.message}",
            )
        except Exception as exc:
            self.stop()
            return ControlCommandResponse(
                False, f"Action '{request.action}' failed and stop was called: {exc}"
            )

    def move(
        self, target_pose: PoseCommand, action_name: str
    ) -> ControlCommandResponse:
        target_orientation = QuaternionUtils.xyzw_to_wxyz(target_pose.orientation_xyzw)
        fk_position, _ = self.compute_current_ee_pose(position_only=True)
        horizontal_position = np.array(target_pose.position, dtype=np.float64)
        horizontal_position[2] = float(fk_position[2])
        vertical_position = np.array(target_pose.position, dtype=np.float64)
        self.motion_phases = [
            MotionPhase(
                horizontal_position,
                target_orientation,
                dwell_after=AUTO_MOVE_DWELL_SEC,
            ),
            MotionPhase(vertical_position, target_orientation),
        ]
        self.phase_dwell_until = None
        return ControlCommandResponse(
            True,
            f"Queued {action_name}: horizontal XY/orientation motion, 0.5s wait, then vertical Z descent.",
        )

    def step_motion(self) -> None:
        if self.phase_dwell_until is not None:
            if time.monotonic() < self.phase_dwell_until:
                return
            self.phase_dwell_until = None

        if not self.motion_phases:
            return

        target = self.motion_phases[0]
        ik_action, success = self.compute_ik_action(
            target.position, target.orientation_wxyz
        )
        if not success:
            carb.log_warn(
                f"IK failed for queued move target {target.position.tolist()}"
            )
            self.motion_phases = []
            self.phase_dwell_until = None
            return

        self.apply_arm_action_preserving_gripper(ik_action)
        fk_position, fk_rotation = self.compute_current_ee_pose(position_only=False)
        position_reached = (
            np.linalg.norm(fk_position - target.position)
            <= AUTO_MOVE_POSITION_TOLERANCE
        )
        if target.orientation_wxyz is None:
            orientation_reached = True
        else:
            current_orientation = rot_matrices_to_quats(fk_rotation)
            orientation_reached = (
                QuaternionUtils.angular_distance_wxyz(
                    current_orientation, target.orientation_wxyz
                )
                <= AUTO_MOVE_ORIENTATION_TOLERANCE
            )
        if position_reached and orientation_reached:
            finished_phase = self.motion_phases.pop(0)
            if finished_phase.apply_home_on_arrival:
                self.apply_home_arm_direct()
            if finished_phase.dwell_after > 0.0:
                self.phase_dwell_until = time.monotonic() + finished_phase.dwell_after

    def apply_arm_action_preserving_gripper(self, action: ArticulationAction) -> None:
        if action.joint_positions is None:
            return

        if action.joint_indices is None:
            joint_positions = np.array(action.joint_positions, dtype=np.float64)
            joint_velocities = (
                None
                if action.joint_velocities is None
                else np.array(action.joint_velocities, dtype=np.float64)
            )
            joint_efforts = (
                None
                if action.joint_efforts is None
                else np.array(action.joint_efforts, dtype=np.float64)
            )
            for target_offset, gripper_index in enumerate(self.gripper_joint_indices):
                if gripper_index < joint_positions.shape[0]:
                    joint_positions[gripper_index] = self.gripper_target_positions[
                        target_offset
                    ]
                if joint_velocities is not None and gripper_index < joint_velocities.shape[0]:
                    joint_velocities[gripper_index] = 0.0
                if joint_efforts is not None and gripper_index < joint_efforts.shape[0]:
                    joint_efforts[gripper_index] = 0.0
            self.controller.apply_action(
                ArticulationAction(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                    joint_efforts=joint_efforts,
                    joint_indices=action.joint_indices,
                )
            )
            return

        joint_indices = np.array(action.joint_indices)
        keep_mask = ~np.isin(joint_indices, self.gripper_joint_indices)
        if not np.any(keep_mask):
            return
        arm_joint_positions = np.array(action.joint_positions)[keep_mask]
        arm_joint_indices = joint_indices[keep_mask]
        merged_joint_positions = np.concatenate(
            [arm_joint_positions, self.gripper_target_positions]
        )
        merged_joint_indices = np.concatenate(
            [arm_joint_indices, self.gripper_joint_indices]
        )
        merged_joint_velocities = None
        if action.joint_velocities is not None:
            merged_joint_velocities = np.concatenate(
                [
                    np.array(action.joint_velocities)[keep_mask],
                    np.zeros(len(self.gripper_joint_indices), dtype=np.float64),
                ]
            )
        merged_joint_efforts = None
        if action.joint_efforts is not None:
            merged_joint_efforts = np.concatenate(
                [
                    np.array(action.joint_efforts)[keep_mask],
                    np.zeros(len(self.gripper_joint_indices), dtype=np.float64),
                ]
            )
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=merged_joint_positions,
                joint_velocities=merged_joint_velocities,
                joint_efforts=merged_joint_efforts,
                joint_indices=merged_joint_indices,
            )
        )

    def grip(self) -> ControlCommandResponse:
        self.apply_gripper_target(GRIPPER_CLOSED_POSITIONS)
        return ControlCommandResponse(True, "Gripper closing.")

    def release(self) -> ControlCommandResponse:
        self.apply_gripper_target(GRIPPER_OPEN_POSITIONS)
        return ControlCommandResponse(True, "Gripper opening.")

    def apply_gripper_target(self, target_positions: list[float]) -> None:
        self.gripper_target_positions = np.array(target_positions, dtype=np.float64)
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=self.gripper_target_positions,
                joint_indices=self.gripper_joint_indices,
            )
        )

    def stop(self) -> ControlCommandResponse:
        self.phase_dwell_until = None
        current_position, current_rotation = self.compute_current_ee_pose(
            position_only=False
        )
        current_orientation = rot_matrices_to_quats(current_rotation)
        vertical_position = np.array(current_position, dtype=np.float64)
        vertical_position[2] = float(self.home_ee_position[2])
        self.motion_phases = [
            MotionPhase(
                vertical_position,
                current_orientation,
                dwell_after=AUTO_MOVE_DWELL_SEC,
            ),
            MotionPhase(
                np.array(self.home_ee_position, dtype=np.float64),
                self.home_ee_orientation,
                apply_home_on_arrival=True,
            ),
        ]
        return ControlCommandResponse(
            True,
            "Queued stop: vertical rise to home Z, 0.5s wait, then return to home pose; gripper unchanged.",
        )

    def compute_home_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self.sync_kinematics_base_pose()
        kinematics = self.kinematics_solver.get_kinematics_solver()
        frame_name = self.kinematics_solver.get_end_effector_frame()
        joint_count = len(kinematics.get_joint_names())
        configured_home = np.array(HOME_JOINT_POSITIONS, dtype=np.float64)
        home_joint_positions = configured_home[:joint_count]
        return kinematics.compute_forward_kinematics(frame_name, home_joint_positions)

    def get_home_arm_action(self) -> ArticulationAction:
        current_positions = self.robot.get_joint_positions()
        if current_positions is None:
            current_positions = np.zeros(len(self.robot.dof_names), dtype=np.float64)
        home_positions = np.array(current_positions, dtype=np.float64)
        configured_home = np.array(HOME_JOINT_POSITIONS, dtype=np.float64)
        arm_joint_count = max(
            0, len(self.robot.dof_names) - len(self.gripper_joint_indices)
        )
        target_count = min(
            configured_home.shape[0], home_positions.shape[0], arm_joint_count
        )
        home_positions[:target_count] = configured_home[:target_count]
        arm_joint_indices = np.arange(target_count, dtype=np.int32)
        return ArticulationAction(
            joint_positions=home_positions[:target_count],
            joint_indices=arm_joint_indices,
        )

    def apply_home_arm_direct(self) -> None:
        self.apply_arm_action_preserving_gripper(self.get_home_arm_action())

    def apply_initial_gripper_open_direct(self) -> None:
        self.apply_gripper_target(GRIPPER_OPEN_POSITIONS)

    def ensure_home_pose(self) -> None:
        if self.home_applied:
            return
        self.motion_phases = []
        self.phase_dwell_until = None
        self.apply_home_arm_direct()
        self.apply_initial_gripper_open_direct()
        self.home_applied = True


def main() -> None:
    control_mode = CONTROL_MODE.strip().lower()
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    viewports.set_camera_view(
        eye=np.array([1.7, 1.2, 1.0]), target=np.array([0.55, 0.0, 0.35])
    )

    scene = CustomScene(RANDOM_SEED)
    scene.build(assets_root_path)
    if set(scene.frankas) != set(ROBOT_CONFIGS):
        carb.log_error("Failed to create both Franka robots")
        simulation_app.close()
        sys.exit(1)
    scene.world.reset()
    scene.initialize_sensors()
    scene.world.play()

    ros_tcp_client: RosTcpClientBase
    controllers: list[BaseController]
    auto_controllers: dict[str, AutoController] = {}
    ros_kwargs = {
        "host": ROS_TCP_HOST,
        "port": ROS_TCP_PORT,
        "joint_topics": JOINT_STATE_TOPICS,
        "robot_pose_topics": ROBOT_POSE_TOPICS,
        "eef_pose_topics": EEF_POSE_TOPICS,
        "marker_topic": MARKER_TOPIC,
        "image_topic": RGB_CAMERA_TOPIC,
        "camera_pose_topic": CAMERA_POSE_TOPIC,
        "frame_id": MARKER_FRAME_ID,
        "publish_hz": MARKER_PUBLISH_HZ,
        "image_publish_hz": RGB_CAMERA_PUBLISH_HZ,
        "joint_state_publish_hz": JOINT_STATE_PUBLISH_HZ,
        "control_service_topics": CONTROL_SERVICE_TOPICS,
        "control_service_type": CONTROL_SERVICE_TYPE,
        "objects": scene.spawned_objects,
    }
    if control_mode == "manual":
        ros_tcp_client = ManualRosClient(**ros_kwargs)
        controllers = [ManualController(scene.frankas["left"])]
    elif control_mode == "auto":
        ros_tcp_client = AutoRosClient(**ros_kwargs)
        controllers = [
            AutoController(scene.frankas[robot_id], CONTROL_SERVICE_TOPICS[robot_id])
            for robot_id in ("left", "right")
        ]
        auto_controllers = {
            robot_id: controller
            for robot_id, controller in zip(("left", "right"), controllers)
            if isinstance(controller, AutoController)
        }
        for controller in controllers:
            if isinstance(controller, AutoController):
                controller.ensure_home_pose()
    else:
        raise ValueError(f"Unsupported CONTROL_MODE: {CONTROL_MODE}")

    reset_needed = False
    objects_spawned = False
    object_spawn_time = time.monotonic() + OBJECT_SPAWN_DELAY_SEC
    step_count = 0

    try:
        while simulation_app.is_running():
            scene.world.step(render=not HEADLESS)
            if not objects_spawned and time.monotonic() >= object_spawn_time:
                scene.spawn_objects()
                objects_spawned = True
            for controller in controllers:
                controller.step(ros_tcp_client)
            ros_tcp_client.publish_joint_state_if_due(scene.frankas)
            eef_poses = {
                robot_id: controller.get_end_effector_pose()
                for robot_id, controller in auto_controllers.items()
            }
            ros_tcp_client.publish_pose_if_due(
                scene.frankas, eef_poses, scene.top_camera
            )
            ros_tcp_client.publish_camera_if_due(scene.top_camera)
            ros_tcp_client.publish_if_due()

            if scene.world.is_stopped() and not reset_needed:
                reset_needed = True
            if scene.world.is_playing() and reset_needed:
                scene.world.reset()
                reset_needed = False

            step_count += 1
            if TEST_STEPS > 0 and step_count >= TEST_STEPS:
                break
    finally:
        ros_tcp_client.shutdown()
        scene.world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
