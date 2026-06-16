# DB Agent Context

Use ROS2 MarkerArray, image, pose, EEF pose, and JointState snapshots as source data.

- Treat goal marker poses as physical surface markers. They describe the visible/physical goal region, not the exact end-effector target for placement.
- When asked for a placement pose, reason about a pose above the goal surface using goal thickness/rim, object size, held-object geometry, and EEF clearance. Avoid both collision with the goal marker/rim and dropping from unnecessarily high above it.
- Prefer `world_summary.goals[color].eef_drop_pose` or a pose at/above `minimum_safe_release_z` for goal Release. Never recommend `physical_marker_pose` as the Release predecessor.
- Use gripper joint positions from JointState to summarize gripper state. Larger finger joint values mean more open; smaller values mean more closed.
- When judging grasp success, compare object pose before/after with EEF pose and gripper state. If the EEF moved but the object did not follow after `Grip` and a subsequent motion, the grasp likely failed or slipped.
- Remember the Isaac `Moving` service behavior: horizontal XY motion happens at the current EEF Z before vertical Z motion. When asked about route safety, consider intermediate low sweeps, not only start/end poses.
- For crowded scenes, summarize whether an object transfer should use a lifted/clearance waypoint above the table before horizontal motion.
- For grasp planning, distinguish approach poses from grasp poses. A high pose above an object is useful for approach, but the EEF must descend to the object's current pose/grasp height before `Grip`.
