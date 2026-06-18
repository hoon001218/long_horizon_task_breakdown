# ROS2 LLM Control Agent

This adds a closed-loop Python controller for the Isaac Sim world in `world.py`.
It replaces manual button presses from `control_command_gui.py` with agents that plan from ROS2 Snapshots, execute only supported primitive actions, observe results, and retry or replan.

## Contract Used

The implementation follows the live code contract:

- Marker topic: `/world/object_markers`
- Camera topic: `/world/top_camera/image_raw`
- Robot pose topics: `/franka_left/pose`, `/franka_right/pose`
- End-effector pose topics: `/franka_left/end_effector_pose`, `/franka_right/end_effector_pose`
- Joint state topics: `/franka_left/joint_states`, `/franka_right/joint_states`
- Services: `/franka_left/control_command`, `/franka_right/control_command`
- Service type: `custom_msgs/srv/ControlCommand`
- Primitive actions: `Moving`, `Centering`, `Placing`, `Grip`, `Release`, `Homing`

The executor sends the same service fields as the GUI: `request.action` and `request.target_pose`, with the fixed vertical end-effector orientation `(x=1, y=0, z=0, w=0)`.
For `Placing`, the executor uses the GUI/world service target positions (`red_goal`: `0.272, 0.228, 0.466`; `blue_goal`: `0.928, -0.228, 0.466`) rather than the lower visual marker z value.

## Files

- `main.py`: CLI entrypoint and closed-loop controller.
- `ros.py`: ROS2 subscriptions, services, and Snapshot manager.
- `executor.py`: primitive action validation and service execution.
- `agents.py`: MainAgent, TaskAgent, ActionAgent, CriticAgent, and LLM/heuristic clients.
- `memory.py`: JSONL logging and task memory.
- `prompts/*.md`: agent operating prompts.
- `logs/`: run logs are written as JSONL.

## Setup

Build and source the ROS2 workspace that provides `custom_msgs` and ROS-TCP-Endpoint, then start the world:

```bash
source install/setup.bash
./isaac-sim/python.sh world.py
```

In another sourced shell, configure `.env`:

```bash
cp .env .env.local
```

Edit `.env` if you want an LLM-backed run. With no API key, `LLM_MODE=auto` falls back to a conservative heuristic planner for smoke testing.

## Run

```bash
source install/setup.bash
/usr/bin/python3 main.py "Move the red object to the red goal."
```

More examples:

```bash
/usr/bin/python3 main.py "Move all objects to the red goal."
/usr/bin/python3 main.py "Pick the object near the left robot and place it on the blue goal."
/usr/bin/python3 main.py --planner heuristic "Move the red object to the red goal."
```

## Notes

- ROS2 Snapshot is always the source of truth.
- Agents are expected to output JSON only.
- The runtime filters invalid IDs and unsupported actions before execution.
- The executor always waits before `Grip` (`--pre-grip-delay`, default `2.0` seconds) so the physical approach motion can settle before the gripper closes.
- Logs include LLM requests/responses, action plans, service results, critic decisions, and task status updates.
