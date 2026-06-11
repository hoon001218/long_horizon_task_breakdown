# ROS2 Interface Docs for `world.py`

This document describes the ROS2 topics and services exposed by `world.py` through ROS-TCP-Endpoint. It is intended for building ROS2 clients that subscribe to simulator state, post-process perception outputs, or send Franka control commands.

## Common Conventions

- Global frame: `world`
- Position units: meters
- Orientation convention in ROS messages: quaternion `x, y, z, w`
- Pose topics use `geometry_msgs/msg/PoseStamped`
- Object/table/goal visualization uses `visualization_msgs/msg/MarkerArray`
- Default endpoint assumption: ROS-TCP-Endpoint server is available on `127.0.0.1:10000`

## Published Topics

| Topic | Type | Rate | Description |
|---|---|---:|---|
| `/world/object_markers` | `visualization_msgs/msg/MarkerArray` | 15 Hz | Table, goal regions, and movable object poses/scales/colors |
| `/world/top_camera/image_raw` | `sensor_msgs/msg/Image` | 8 Hz | RGB top-view camera image |
| `/world/top_camera/pose` | `geometry_msgs/msg/PoseStamped` | 15 Hz | Top camera world pose |
| `/franka_left/joint_states` | `sensor_msgs/msg/JointState` | 30 Hz | Left Franka joint state |
| `/franka_right/joint_states` | `sensor_msgs/msg/JointState` | 30 Hz | Right Franka joint state |
| `/franka_left/pose` | `geometry_msgs/msg/PoseStamped` | 15 Hz | Left Franka base/world pose |
| `/franka_right/pose` | `geometry_msgs/msg/PoseStamped` | 15 Hz | Right Franka base/world pose |
| `/franka_left/end_effector_pose` | `geometry_msgs/msg/PoseStamped` | 15 Hz | Left Franka end-effector world pose |
| `/franka_right/end_effector_pose` | `geometry_msgs/msg/PoseStamped` | 15 Hz | Right Franka end-effector world pose |

## MarkerArray Details

Topic: `/world/object_markers`

Fixed markers:

| Namespace | ID | Type | Meaning |
|---|---:|---:|---|
| `table` | 0 | `Marker.CUBE` | Main table body |
| `red_goal` | 1 | `Marker.CUBE` | Red placement goal region for the left Franka |
| `blue_goal` | 2 | `Marker.CUBE` | Blue placement goal region for the right Franka |

Movable object markers start at ID `3`.

Object namespaces follow:

- `cube_1`, `cube_2`, ...
- `capsule_1`, `capsule_2`, ...
- `sphere_1`, `sphere_2`, ...

Marker type mapping:

- Cubes use `Marker.CUBE`
- Spheres use `Marker.SPHERE`
- Capsules are approximated as elongated `Marker.SPHERE` markers using non-uniform scale

Object colors:

- Movable objects are only red or blue
- Red RGB approx: `(0.95, 0.05, 0.05)`
- Blue RGB approx: `(0.05, 0.25, 0.95)`

## Image Topic Details

Topic: `/world/top_camera/image_raw`

Type: `sensor_msgs/msg/Image`

Fields:

- `header.frame_id`: `world`
- `encoding`: `rgb8`
- `height`: `480`
- `width`: `640`
- `step`: `1920`
- `data`: row-major RGB bytes

No `CameraInfo` topic is currently published.

## PoseStamped Topics

All PoseStamped topics use:

- `header.frame_id`: `world`
- `header.stamp`: simulator wall-clock serialization time

Robot base poses:

- `/franka_left/pose`
- `/franka_right/pose`

End-effector poses:

- `/franka_left/end_effector_pose`
- `/franka_right/end_effector_pose`

Camera pose:

- `/world/top_camera/pose`

## Joint State Topics

Auto mode publishes:

- `/franka_left/joint_states`
- `/franka_right/joint_states`

Type: `sensor_msgs/msg/JointState`

Fields:

- `header.frame_id`: `world`
- `name`: Isaac Franka DOF names
- `position`: current joint positions
- `velocity`: current joint velocities when available
- `effort`: empty

## Control Services

The simulator exposes one control service per Franka.

| Service | Type | Robot |
|---|---|---|
| `/franka_left/control_command` | `custom_msgs/srv/ControlCommand` | Left Franka |
| `/franka_right/control_command` | `custom_msgs/srv/ControlCommand` | Right Franka |

Service definition:

```srv
string action
geometry_msgs/Pose target_pose
---
bool success
string message
```

Valid `action` strings are exact and case-sensitive:

| Action | Uses `target_pose` | Meaning |
|---|---:|---|
| `Moving` | Yes | Move end-effector to the selected object pose |
| `Centering` | Yes | Move end-effector to the table center pose |
| `Placing` | Yes | Move end-effector to the robot-specific goal region |
| `Grip` | No | Close gripper |
| `Realease` | No | Open gripper. Note the spelling is intentionally currently `Realease` in code |
| `Homing` | No | Return arm to home pose without changing gripper state |

Movement behavior for `Moving`, `Centering`, and `Placing`:

- The target position is interpreted in the `world` frame
- The target orientation is interpreted as a world-frame quaternion
- The controller first moves horizontally at the current EEF Z height
- It waits about `0.5 s`
- Then it descends vertically to the requested target Z
- The gripper state is preserved during movement

Recommended vertical end-effector orientation used by `control_command_gui.py`:

```python
orientation.x = 1.0
orientation.y = 0.0
orientation.z = 0.0
orientation.w = 0.0
```

## Built-In Target Positions

These are the target points used by `control_command_gui.py`.

Table center:

```text
x = 0.600
y = 0.000
z = 0.460
```

Left robot placement target, red goal:

```text
x = 0.272
y = 0.228
z = 0.466
```

Right robot placement target, blue goal:

```text
x = 0.928
y = -0.228
z = 0.466
```

## Example ROS2 Python Client Snippets

Subscribe to object markers:

```python
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

class MarkerClient(Node):
    def __init__(self):
        super().__init__("marker_client")
        self.create_subscription(
            MarkerArray,
            "/world/object_markers",
            self.on_markers,
            10,
        )

    def on_markers(self, msg):
        for marker in msg.markers:
            self.get_logger().info(
                f"{marker.ns}[{marker.id}] "
                f"pos=({marker.pose.position.x:.3f}, "
                f"{marker.pose.position.y:.3f}, "
                f"{marker.pose.position.z:.3f})"
            )
```

Call a Franka control service:

```python
from custom_msgs.srv import ControlCommand
from geometry_msgs.msg import Pose

client = node.create_client(ControlCommand, "/franka_left/control_command")
client.wait_for_service()

req = ControlCommand.Request()
req.action = "Moving"
req.target_pose = Pose()
req.target_pose.position.x = 0.6
req.target_pose.position.y = 0.0
req.target_pose.position.z = 0.46
req.target_pose.orientation.x = 1.0
req.target_pose.orientation.y = 0.0
req.target_pose.orientation.z = 0.0
req.target_pose.orientation.w = 0.0

future = client.call_async(req)
```

Subscribe to an EEF pose:

```python
from geometry_msgs.msg import PoseStamped

node.create_subscription(
    PoseStamped,
    "/franka_right/end_effector_pose",
    lambda msg: print(msg.pose.position),
    10,
)
```

## Notes for RViz2

- Use fixed frame `world`
- Pose topics are `PoseStamped`, so RViz2 can resolve the frame directly
- MarkerArray fixed markers include the table and goal regions; filter by `marker.ns` if only movable objects are needed
- For object-only processing, ignore namespaces: `table`, `red_goal`, `blue_goal`
