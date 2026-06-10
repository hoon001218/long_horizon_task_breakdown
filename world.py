"""Isaac Sim 5.0 standalone scene synced through ROS-TCP-Endpoint.

Run this file with Isaac Sim's bundled Python, for example:

    ./isaac-sim/python.sh world.py

The script creates a fixed Franka robot, a cuboid table, and randomized small
objects on the table. Isaac Sim connects as a TCP client to an already-running
ROS-TCP-Endpoint server and:

* Manual mode subscribes to ``sensor_msgs/msg/JointState`` on ``/joint_states``;
* Auto mode serves ``custom_msgs/srv/ControlCommand`` and publishes ``/joint_states``;
* both modes publish ``visualization_msgs/msg/MarkerArray`` with object poses.
"""

from isaacsim import SimulationApp

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass

# Object count controls.
NUM_CUBES = 2
NUM_CAPSULES = 2
NUM_SPHERES = 2
OBJECT_SIZE_SCALE = 0.5

MARKER_TOPIC = "/world/object_markers"
MARKER_FRAME_ID = "world"
MARKER_PUBLISH_HZ = 15.0
ROS_TCP_HOST = "127.0.0.1"
ROS_TCP_PORT = 10000
JOINT_STATE_TOPIC = "/joint_states"
JOINT_STATE_PUBLISH_HZ = 30.0
CONTROL_SERVICE_TOPIC = "/control_command"
CONTROL_SERVICE_TYPE = "custom_msgs/ControlCommand"
CONTROL_SERVICE_TIMEOUT = 5.0

CONTROL_MODE = "auto"  # "manual" or "auto"
HEADLESS = False
RANDOM_SEED = None
TEST_STEPS = 0

HOME_JOINT_POSITIONS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
GRIPPER_OPEN_POSITIONS = [0.04, 0.04]
GRIPPER_CLOSED_POSITIONS = [0.0, 0.0]
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
AUTO_MOVE_POSITION_TOLERANCE = 0.012
AUTO_MOVE_ORIENTATION_TOLERANCE = 0.1
AUTO_MOVE_DWELL_SEC = 0.5

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
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.kinematics_solver import (
    KinematicsSolver as FrankaKinematicsSolver,
)
from isaacsim.storage.native import get_assets_root_path

FRANKA_PRIM_PATH = "/World/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

TABLE_DIMS = np.array([0.9, 0.7, 0.08])
TABLE_CENTER = np.array([0.6, 0.0, 0.38])
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_DIMS[2] / 2.0

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


def normalize_ros_topic(raw_topic_name: str) -> str:
    if not raw_topic_name:
        raise ValueError("ROS topic name must not be empty.")
    return raw_topic_name if raw_topic_name.startswith("/") else f"/{raw_topic_name}"


def random_yaw_quaternion(rng: np.random.Generator) -> np.ndarray:
    yaw = float(rng.uniform(-np.pi, np.pi))
    half_yaw = yaw * 0.5
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def random_horizontal_capsule_quaternion(rng: np.random.Generator) -> np.ndarray:
    yaw = random_yaw_quaternion(rng)
    pitch_90 = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    return quaternion_multiply(yaw, pitch_90)


def marker_orientation_from_wxyz(
    orientation: np.ndarray,
) -> tuple[float, float, float, float]:
    return (
        float(orientation[1]),
        float(orientation[2]),
        float(orientation[3]),
        float(orientation[0]),
    )


def table_limited_workspace(
    footprint_radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    table_x_min = TABLE_CENTER[0] - TABLE_DIMS[0] / 2.0 + footprint_radius
    table_x_max = TABLE_CENTER[0] + TABLE_DIMS[0] / 2.0 - footprint_radius
    table_y_min = TABLE_CENTER[1] - TABLE_DIMS[1] / 2.0 + footprint_radius
    table_y_max = TABLE_CENTER[1] + TABLE_DIMS[1] / 2.0 - footprint_radius

    x_range = (
        max(ROBOT_WORKSPACE_X_RANGE[0], table_x_min),
        min(ROBOT_WORKSPACE_X_RANGE[1], table_x_max),
    )
    y_range = (
        max(ROBOT_WORKSPACE_Y_RANGE[0], table_y_min),
        min(ROBOT_WORKSPACE_Y_RANGE[1], table_y_max),
    )
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise RuntimeError(
            "Object footprint is too large for the configured robot workspace and table bounds."
        )
    return x_range, y_range


def sample_non_overlapping_xy(
    rng: np.random.Generator,
    footprint_radius: float,
    occupied: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    x_range, y_range = table_limited_workspace(footprint_radius)
    for _ in range(MAX_SPAWN_ATTEMPTS):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
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


def create_physics_material(
    name: str, static_friction: float, dynamic_friction: float
) -> PhysicsMaterial:
    return PhysicsMaterial(
        prim_path=f"/World/PhysicsMaterials/{name}",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )


def add_franka(world: World, assets_root_path: str) -> Robot:
    robot = add_reference_to_stage(
        usd_path=assets_root_path + FRANKA_USD_PATH,
        prim_path=FRANKA_PRIM_PATH,
    )
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    # The Franka USD is a fixed-base articulation; keep it at the world origin facing +X.
    return world.scene.add(
        Robot(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            position=np.array([0.0, 0.0, 0.0]),
        )
    )


def create_spawn_specs(rng: np.random.Generator) -> list[SpawnSpec]:
    specs: list[SpawnSpec] = []

    for _ in range(NUM_CUBES):
        size = float(rng.uniform(0.045, 0.065) * OBJECT_SIZE_SCALE)
        specs.append(
            SpawnSpec(
                shape="cube",
                size=size,
                radius=0.0,
                height=0.0,
                footprint_radius=np.sqrt(2.0) * size / 2.0,
                color=np.array([0.1, 0.45, 0.95]),
                orientation=random_yaw_quaternion(rng),
                z=TABLE_TOP_Z + size / 2.0 + 0.003,
            )
        )

    for _ in range(NUM_CAPSULES):
        radius = float(rng.uniform(0.022, 0.032) * OBJECT_SIZE_SCALE)
        height = float(rng.uniform(0.09, 0.13) * OBJECT_SIZE_SCALE)
        specs.append(
            SpawnSpec(
                shape="capsule",
                size=0.0,
                radius=radius,
                height=height,
                footprint_radius=height / 2.0 + radius,
                color=np.array([0.95, 0.62, 0.12]),
                orientation=random_horizontal_capsule_quaternion(rng),
                z=TABLE_TOP_Z + radius + 0.004,
            )
        )

    for _ in range(NUM_SPHERES):
        radius = float(rng.uniform(0.028, 0.04) * OBJECT_SIZE_SCALE)
        specs.append(
            SpawnSpec(
                shape="sphere",
                size=0.0,
                radius=radius,
                height=0.0,
                footprint_radius=radius,
                color=np.array([0.12, 0.75, 0.35]),
                orientation=random_yaw_quaternion(rng),
                z=TABLE_TOP_Z + radius + 0.004,
            )
        )

    rng.shuffle(specs)
    return specs


def add_table_and_objects(world: World, seed: int | None) -> list[SpawnedObject]:
    rng = np.random.default_rng(seed)
    table_material = None
    object_material = None
    if not USE_DEFAULT_FRICTION:
        table_material = create_physics_material(
            "table_high_friction", TABLE_STATIC_FRICTION, TABLE_DYNAMIC_FRICTION
        )
        object_material = create_physics_material(
            "object_high_friction", OBJECT_STATIC_FRICTION, OBJECT_DYNAMIC_FRICTION
        )

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=TABLE_CENTER,
            scale=TABLE_DIMS,
            size=1.0,
            color=np.array([0.55, 0.42, 0.30]),
            **({} if table_material is None else {"physics_material": table_material}),
        )
    )

    spawned_objects: list[SpawnedObject] = []
    occupied: list[tuple[np.ndarray, float]] = []
    shape_counts = {"cube": 0, "capsule": 0, "sphere": 0}

    for spec in create_spawn_specs(rng):
        shape_counts[spec.shape] += 1
        object_index = shape_counts[spec.shape]
        xy = sample_non_overlapping_xy(rng, spec.footprint_radius, occupied)
        position = np.array([xy[0], xy[1], spec.z])

        if spec.shape == "cube":
            prim = world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Objects/Cube_{object_index}",
                    name=f"cube_{object_index}",
                    position=position,
                    orientation=spec.orientation,
                    scale=np.array([spec.size, spec.size, spec.size]),
                    size=1.0,
                    color=spec.color,
                    **(
                        {}
                        if object_material is None
                        else {"physics_material": object_material}
                    ),
                    mass=0.05,
                )
            )
            marker_scale = np.array([spec.size, spec.size, spec.size])
        elif spec.shape == "capsule":
            prim = world.scene.add(
                DynamicCapsule(
                    prim_path=f"/World/Objects/Capsule_{object_index}",
                    name=f"capsule_{object_index}",
                    position=position,
                    orientation=spec.orientation,
                    radius=spec.radius,
                    height=spec.height,
                    color=spec.color,
                    **(
                        {}
                        if object_material is None
                        else {"physics_material": object_material}
                    ),
                    mass=0.04,
                )
            )
            marker_scale = np.array([2.0 * spec.radius, 2.0 * spec.radius, spec.height])
        else:
            prim = world.scene.add(
                DynamicSphere(
                    prim_path=f"/World/Objects/Sphere_{object_index}",
                    name=f"sphere_{object_index}",
                    position=position,
                    orientation=spec.orientation,
                    radius=spec.radius,
                    color=spec.color,
                    **(
                        {}
                        if object_material is None
                        else {"physics_material": object_material}
                    ),
                    mass=0.04,
                )
            )
            marker_scale = np.array(
                [2.0 * spec.radius, 2.0 * spec.radius, 2.0 * spec.radius]
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

    return spawned_objects


class CdrWriter:
    """Small ROS2 CDR writer for the message subset used by this scene."""

    def __init__(self) -> None:
        self.buffer = bytearray(b"\x00\x01\x00\x00")

    def align(self, alignment: int) -> None:
        cdr_offset = len(self.buffer) - 4
        self.buffer.extend(b"\x00" * ((-cdr_offset) % alignment))

    def write_bool(self, value: bool) -> None:
        self.buffer.extend(struct.pack("<?", value))

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


def serialize_control_command_response(response: ControlCommandResponse) -> bytes:
    writer = CdrWriter()
    writer.write_bool(response.success)
    writer.write_string(response.message)
    return writer.to_bytes()


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


def quaternion_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-6:
        return None
    x, y, z, w = quaternion / norm
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_angular_distance_wxyz(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left / max(float(np.linalg.norm(left)), 1.0e-9)
    right_norm = right / max(float(np.linalg.norm(right)), 1.0e-9)
    dot = float(np.clip(abs(np.dot(left_norm, right_norm)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def write_marker(
    writer: CdrWriter, marker_id: int, frame_id: str, spawned_object: SpawnedObject
) -> None:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1_000_000_000)
    position, orientation = spawned_object.prim.get_world_pose()
    qx, qy, qz, qw = marker_orientation_from_wxyz(orientation)
    marker_type = 1 if spawned_object.shape == "cube" else 2

    writer.write_header(sec, nanosec, frame_id)
    writer.write_string(spawned_object.name)
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
    writer.write_vector3(spawned_object.marker_scale)
    writer.write_color(spawned_object.color, 0.9)
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


def serialize_marker_array(objects: list[SpawnedObject], frame_id: str) -> bytes:
    writer = CdrWriter()
    writer.write_uint32(len(objects))
    for marker_id, spawned_object in enumerate(objects):
        write_marker(writer, marker_id, frame_id, spawned_object)
    return writer.to_bytes()


class RosTcpEndpointClient:
    def __init__(
        self,
        host: str,
        port: int,
        control_mode: str,
        joint_topic: str,
        marker_topic: str,
        frame_id: str,
        publish_hz: float,
        joint_state_publish_hz: float,
        control_service_topic: str,
        control_service_type: str,
        objects: list[SpawnedObject],
    ) -> None:
        self.host = host
        self.port = port
        self.control_mode = control_mode.lower()
        self.joint_topic = normalize_ros_topic(joint_topic)
        self.marker_topic = normalize_ros_topic(marker_topic)
        self.control_service_topic = normalize_ros_topic(control_service_topic)
        self.control_service_type = control_service_type
        self.objects = objects
        self.frame_id = frame_id
        self.publish_period = 1.0 / max(publish_hz, 0.1)
        self.joint_state_publish_period = 1.0 / max(joint_state_publish_hz, 0.1)
        self.last_publish_time = 0.0
        self.last_joint_state_publish_time = 0.0
        self.socket: socket.socket | None = None
        self.send_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.service_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None
        self.latest_joint_state: JointStateCommand | None = None
        self.pending_service_request_id: int | None = None
        self.pending_service_requests: list[PendingServiceRequest] = []
        self.logged_first_publish = False
        self.logged_first_joint_state = False
        self.logged_first_joint_state_publish = False
        self.logged_disconnect = False
        self.connect()

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
        return RosTcpEndpointClient.pack_packet(command, payload)

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

    def connect(self) -> None:
        self.socket = socket.create_connection((self.host, self.port), timeout=5.0)
        self.socket.settimeout(None)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.reader_thread.start()

        if self.control_mode == "manual":
            self.send_syscommand(
                "__subscribe",
                {"topic": self.joint_topic, "message_name": "sensor_msgs/JointState"},
            )
        elif self.control_mode == "auto":
            self.send_syscommand(
                "__publish",
                {
                    "topic": self.joint_topic,
                    "message_name": "sensor_msgs/JointState",
                    "queue_size": 10,
                    "latch": False,
                },
            )
            self.send_syscommand(
                "__unity_service",
                {
                    "topic": self.control_service_topic,
                    "message_name": self.control_service_type,
                },
            )
        else:
            raise ValueError(f"Unsupported CONTROL_MODE: {self.control_mode}")

        self.send_syscommand(
            "__publish",
            {
                "topic": self.marker_topic,
                "message_name": "visualization_msgs/MarkerArray",
                "queue_size": 10,
                "latch": False,
            },
        )
        carb.log_info(
            f"Connected to ROS-TCP-Endpoint at {self.host}:{self.port}; "
            f"mode={self.control_mode}, joint_topic={self.joint_topic}, marker_topic={self.marker_topic}"
        )

    def send_syscommand(self, command: str, params: dict) -> None:
        self.send_raw(self.pack_syscommand(command, params))

    def send_packet(self, destination: str, payload: bytes) -> None:
        self.send_raw(self.pack_packet(destination, payload))

    def send_raw(self, packet: bytes) -> None:
        if self.socket is None:
            return
        with self.send_lock:
            self.socket.sendall(packet)

    def reader_loop(self) -> None:
        assert self.socket is not None
        try:
            while not self.stop_event.is_set():
                destination_length = struct.unpack("<I", self.recvall(self.socket, 4))[
                    0
                ]
                destination = (
                    self.recvall(self.socket, destination_length)
                    .decode("utf-8")
                    .rstrip("\x00")
                )
                payload_length = struct.unpack("<I", self.recvall(self.socket, 4))[0]
                payload = self.recvall(self.socket, payload_length)

                if self.pending_service_request_id is not None:
                    self.handle_service_payload(destination, payload)
                elif destination == self.joint_topic and self.control_mode == "manual":
                    command = deserialize_joint_state(payload)
                    with self.state_lock:
                        self.latest_joint_state = command
                    if not self.logged_first_joint_state:
                        carb.log_info(
                            f"Received first JointState from ROS-TCP-Endpoint on {self.joint_topic}"
                        )
                        self.logged_first_joint_state = True
                elif destination.startswith("__"):
                    self.log_syscommand(destination, payload)
        except Exception as exc:
            if not self.stop_event.is_set() and not self.logged_disconnect:
                carb.log_error(f"ROS-TCP-Endpoint reader stopped: {exc}")
                self.logged_disconnect = True

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
        now = time.monotonic()
        if now - self.last_publish_time < self.publish_period:
            return
        self.last_publish_time = now
        self.publish()

    def publish(self) -> None:
        payload = serialize_marker_array(self.objects, self.frame_id)
        self.send_packet(self.marker_topic, payload)
        if not self.logged_first_publish:
            carb.log_info(
                f"Published first MarkerArray with {len(self.objects)} objects through ROS-TCP-Endpoint"
            )
            self.logged_first_publish = True

    def publish_joint_state_if_due(self, robot: Robot) -> None:
        if self.control_mode != "auto":
            return
        now = time.monotonic()
        if now - self.last_joint_state_publish_time < self.joint_state_publish_period:
            return
        self.last_joint_state_publish_time = now

        positions = robot.get_joint_positions()
        velocities = robot.get_joint_velocities()
        if positions is None:
            return
        if velocities is None:
            velocities = []
        payload = serialize_joint_state(
            list(robot.dof_names),
            [float(value) for value in positions],
            [float(value) for value in velocities],
            [],
            self.frame_id,
        )
        self.send_packet(self.joint_topic, payload)
        if not self.logged_first_joint_state_publish:
            carb.log_info(
                f"Published first JointState through ROS-TCP-Endpoint on {self.joint_topic}"
            )
            self.logged_first_joint_state_publish = True

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
        if destination != self.control_service_topic:
            response = ControlCommandResponse(
                False, f"Unexpected service payload destination: {destination}"
            )
            self.send_service_response(srv_id, response)
            return

        try:
            request = deserialize_control_command_request(payload)
        except Exception as exc:
            self.send_service_response(
                srv_id, ControlCommandResponse(False, f"Failed to parse request: {exc}")
            )
            return

        pending = PendingServiceRequest(srv_id, request, threading.Event())
        with self.service_lock:
            self.pending_service_requests.append(pending)

        if not pending.done_event.wait(CONTROL_SERVICE_TIMEOUT):
            pending.response = ControlCommandResponse(
                False, "ControlCommand handling timed out."
            )
        self.send_service_response(
            srv_id,
            pending.response or ControlCommandResponse(False, "No response generated."),
        )

    def pop_service_requests(self) -> list[PendingServiceRequest]:
        with self.service_lock:
            requests = self.pending_service_requests
            self.pending_service_requests = []
        return requests

    def send_service_response(
        self, srv_id: int, response: ControlCommandResponse
    ) -> None:
        packet = b"".join(
            [
                self.pack_syscommand("__response", {"srv_id": srv_id}),
                self.pack_packet(
                    self.control_service_topic,
                    serialize_control_command_response(response),
                ),
            ]
        )
        self.send_raw(packet)

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.socket is not None:
            try:
                self.send_packet("", b"")
            except Exception:
                pass
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.socket.close()
            self.socket = None


class JointStateFollower:
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.unknown_joint_names: set[str] = set()

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


class AutoController:
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.kinematics_solver = FrankaKinematicsSolver(robot)
        self.gripper_joint_indices = np.array(
            [self.robot.get_dof_index(name) for name in GRIPPER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.motion_phases: list[MotionPhase] = []
        self.phase_dwell_until: float | None = None
        self.home_ee_position, home_ee_rotation = self.compute_home_ee_pose()
        self.home_ee_orientation = rot_matrices_to_quats(home_ee_rotation)
        self.home_applied = False

    def process_service_requests(self, ros_tcp_client: RosTcpEndpointClient) -> None:
        for pending in ros_tcp_client.pop_service_requests():
            pending.response = self.dispatch(pending.request)
            pending.done_event.set()
        self.step()

    def dispatch(self, request: ControlCommandRequest) -> ControlCommandResponse:
        action = request.action.strip().lower()
        try:
            if action == "move":
                return self.move(request.target_pose)
            if action == "grip":
                return self.grip()
            if action == "release":
                return self.release()
            if action == "stop":
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

    def move(self, target_pose: PoseCommand) -> ControlCommandResponse:
        target_orientation = quaternion_xyzw_to_wxyz(target_pose.orientation_xyzw)
        fk_position, _ = self.kinematics_solver.compute_end_effector_pose(
            position_only=True
        )
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
            "Queued move: horizontal XY/orientation motion, 0.5s wait, then vertical Z descent.",
        )

    def step(self) -> None:
        if self.phase_dwell_until is not None:
            if time.monotonic() < self.phase_dwell_until:
                return
            self.phase_dwell_until = None

        if not self.motion_phases:
            return

        target = self.motion_phases[0]
        ik_action, success = self.kinematics_solver.compute_inverse_kinematics(
            target.position,
            target.orientation_wxyz,
            position_tolerance=AUTO_MOVE_POSITION_TOLERANCE,
            orientation_tolerance=(
                AUTO_MOVE_ORIENTATION_TOLERANCE
                if target.orientation_wxyz is not None
                else None
            ),
        )
        if not success:
            carb.log_warn(
                f"IK failed for queued move target {target.position.tolist()}"
            )
            self.motion_phases = []
            self.phase_dwell_until = None
            return

        self.apply_arm_action_preserving_gripper(ik_action)
        fk_position, fk_rotation = self.kinematics_solver.compute_end_effector_pose(
            position_only=False
        )
        position_reached = (
            np.linalg.norm(fk_position - target.position)
            <= AUTO_MOVE_POSITION_TOLERANCE
        )
        if target.orientation_wxyz is None:
            orientation_reached = True
        else:
            current_orientation = rot_matrices_to_quats(fk_rotation)
            orientation_reached = (
                quaternion_angular_distance_wxyz(
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

        current_positions = self.robot.get_joint_positions()
        if current_positions is None:
            self.controller.apply_action(action)
            return

        if action.joint_indices is None:
            joint_positions = np.array(action.joint_positions, dtype=np.float64)
            for gripper_index in self.gripper_joint_indices:
                if gripper_index < joint_positions.shape[0]:
                    joint_positions[gripper_index] = current_positions[gripper_index]
            self.controller.apply_action(
                ArticulationAction(
                    joint_positions=joint_positions,
                    joint_velocities=action.joint_velocities,
                    joint_efforts=action.joint_efforts,
                    joint_indices=action.joint_indices,
                )
            )
            return

        joint_indices = np.array(action.joint_indices)
        keep_mask = ~np.isin(joint_indices, self.gripper_joint_indices)
        if not np.any(keep_mask):
            return
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=np.array(action.joint_positions)[keep_mask],
                joint_velocities=(
                    None
                    if action.joint_velocities is None
                    else np.array(action.joint_velocities)[keep_mask]
                ),
                joint_efforts=(
                    None
                    if action.joint_efforts is None
                    else np.array(action.joint_efforts)[keep_mask]
                ),
                joint_indices=joint_indices[keep_mask],
            )
        )

    def grip(self) -> ControlCommandResponse:
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=np.array(GRIPPER_CLOSED_POSITIONS, dtype=np.float64),
                joint_indices=self.gripper_joint_indices,
            )
        )
        return ControlCommandResponse(True, "Gripper closing.")

    def release(self) -> ControlCommandResponse:
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=np.array(GRIPPER_OPEN_POSITIONS, dtype=np.float64),
                joint_indices=self.gripper_joint_indices,
            )
        )
        return ControlCommandResponse(True, "Gripper opening.")

    def stop(self) -> ControlCommandResponse:
        self.phase_dwell_until = None
        current_position, current_rotation = (
            self.kinematics_solver.compute_end_effector_pose(position_only=False)
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
        self.controller.apply_action(self.get_home_arm_action())

    def ensure_home_pose(self) -> None:
        if self.home_applied:
            return
        self.motion_phases = []
        self.phase_dwell_until = None
        self.apply_home_arm_direct()
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

    world = World(stage_units_in_meters=1.0)
    if USE_DEFAULT_FRICTION:
        world.scene.add_default_ground_plane()
    else:
        world.scene.add_default_ground_plane(
            static_friction=GROUND_STATIC_FRICTION,
            dynamic_friction=GROUND_DYNAMIC_FRICTION,
            restitution=0.0,
        )
    franka = add_franka(world, assets_root_path)
    spawned_objects = add_table_and_objects(world, RANDOM_SEED)

    ros_tcp_client = RosTcpEndpointClient(
        ROS_TCP_HOST,
        ROS_TCP_PORT,
        control_mode,
        JOINT_STATE_TOPIC,
        MARKER_TOPIC,
        MARKER_FRAME_ID,
        MARKER_PUBLISH_HZ,
        JOINT_STATE_PUBLISH_HZ,
        CONTROL_SERVICE_TOPIC,
        CONTROL_SERVICE_TYPE,
        spawned_objects,
    )

    world.reset()
    world.play()
    joint_state_follower = (
        JointStateFollower(franka) if control_mode == "manual" else None
    )
    auto_controller = AutoController(franka) if control_mode == "auto" else None
    if auto_controller is not None:
        auto_controller.ensure_home_pose()

    reset_needed = False
    step_count = 0

    try:
        while simulation_app.is_running():
            world.step(render=not HEADLESS)
            if joint_state_follower is not None:
                joint_state_follower.apply(ros_tcp_client.consume_latest_joint_state())
            if auto_controller is not None:
                auto_controller.process_service_requests(ros_tcp_client)
                ros_tcp_client.publish_joint_state_if_due(franka)
            ros_tcp_client.publish_if_due()

            if world.is_stopped() and not reset_needed:
                reset_needed = True
            if world.is_playing() and reset_needed:
                world.reset()
                reset_needed = False

            step_count += 1
            if TEST_STEPS > 0 and step_count >= TEST_STEPS:
                break
    finally:
        ros_tcp_client.shutdown()
        world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
