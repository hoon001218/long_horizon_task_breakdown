# DB Agent Context

Role separation:

- The Python system prompt defines your response format and source-of-truth role.
- This CLAUDE.md file defines robot-domain observation policy and durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Use ROS2 MarkerArray, image, pose, EEF pose, and JointState snapshots as source data.

- Treat `world_summary.control_service_contract` as the ordinary execution contract because it describes the `world.py` ControlCommand service baseline.
- For ordinary placement, prefer the object-specific `world_summary.objects_by_id[object_id].recommended_goal_drop_pose` over raw goal marker poses or derived recovery poses. If no object-specific recommendation exists, use `world_summary.control_service_contract.goal_targets[color]`.
- Goal marker poses and `eef_drop_pose`/`minimum_safe_release_z` are diagnostic and recovery references. Use them when explaining why a failed placement may need a different pose, not as the default plan.
- `table_center_handover` is a stable table support/drop location. Its z should come from the fixed table-center control target, not from current object z values that may include lifted/floating objects.
- `world.py` enforces movement shape at the action level: vertical rise to home-level Z, high-Z horizontal travel, then vertical descent. `Moving` descends slightly below the object marker for grasping; `Centering` and `Placing` descend slightly above their semantic target and apply a small final XY variation.
- `world.py` uses each robot's home EEF orientation during movement commands to avoid left-arm tip flips from mismatched requested orientations.
- Use gripper joint positions from JointState to summarize gripper state. Larger finger joint values mean more open; smaller values mean more closed.
- Franka finger joints are about `0.04` when fully open and about `0.0` when fully closed with no object. For runtime `Grip` assessment, near empty fully-closed means failed empty grasp; any other finger position still needs object-EEF proximity and later object-following evidence.
- When judging grasp success, compare object pose before/after with EEF pose and gripper state. If the EEF moved but the object did not follow after `Grip` and a subsequent motion, the grasp likely failed or slipped.
- For grasp planning, the working default is `Moving(object marker pose)` immediately before `Grip`, with vertical end-effector orientation enforced by the runtime.
