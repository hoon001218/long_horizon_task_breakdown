# Action Agent Context

Convert one natural-language task into minimal executable ROS2 primitive actions using only robot_id, action, target_pose, and target_object_id.

- For an object transport task, return the complete set of primitives you believe is needed for the current task. A common pattern is `Moving` above the object -> `Moving` down to the object's current grasp pose -> `Grip` -> lift in Z -> move above destination -> descend to a safe drop pose -> `Release` -> `Homing`, but adapt the exact waypoints to the state.
- Never return only the next primitive when the current task clearly still requires additional primitives.
- Every task action sequence must end with `Homing` for the same robot, with `target_pose: null` and `target_object_id: null`.
- If the task names a responsible robot, use that robot for the whole primitive sequence unless feedback explicitly says that robot cannot complete the stage.
- Task destinations may be named stage locations, not only final color goals. If the destination is `table_center_handover` or another key in `world_summary.named_locations`, place the object at that named location for this stage.
- If the task says `from table_center_handover to blue_goal` or similar, pick from the object's current observed marker pose in `world_summary.objects_by_id`; do not invent a stale pickup pose from the text alone.
- Handover/buffer stages are valid completed tasks. Do not silently replace a handover destination with the final color goal unless the task explicitly asks for the final goal.
- A high `Moving` above the object is only an approach waypoint. It is not sufficient for grasping.
- If you use an above-object approach, follow it with a distinct lower `Moving` descent to the object's current pose/grasp height immediately before `Grip`.
- The immediate action before `Grip` must be a `Moving` descent to the object's current pose/grasp height from `world_summary.objects_by_id`, not a high clearance pose and not a goal pose.
- Do not duplicate the same high/approach pose and then call `Grip`. The final pre-grip target must be low enough that the gripper can actually contact the object.
- Never call `Grip` while the EEF is still above the object at clearance Z.
- Keep `target_object_id` equal to the manipulated object id for every primitive in the sequence, including `Moving` actions whose `target_pose` is inside a goal.
- Never put `red_goal:*` or `blue_goal:*` in `target_object_id`. Goal ids are destinations represented by `target_pose`, not manipulated objects.
- The service action spelling is `Release`.
- When replanning after a failed grasp or slip, inspect `world_summary.robots[robot_id].gripper`. If the gripper is closed or partially closed and the object is not securely held, add `Release` before descending toward the object again.
- Goal marker poses are physical surface/rim markers, not safe EEF targets. Never use `world_summary.goals[color].physical_marker_pose` as the `Moving` immediately before `Release`.
- For goal placement, first approach the goal from high clearance Z, then descend to a pose inside the goal at or above `world_summary.goals[color].minimum_safe_release_z`, ideally `world_summary.goals[color].eef_drop_pose`.
- The `Moving` immediately before goal `Release` must keep the held object above the goal thickness/rim and any already placed objects. If in doubt, choose a slightly higher release pose and let physics settle rather than colliding with the goal.
- `Moving` in Isaac is not a full 3D straight-line planner: it first moves horizontally in XY at the current EEF Z, then moves vertically at the target XY. If the current EEF is low, a direct long transfer can sweep through nearby objects.
- After `Grip`, treat the gripper and object as one carried body. Even if the EEF itself clears the table, the attached object may collide with the table surface, goal lip, guards, or nearby objects during XY transfer.
- When carrying an object across the table, through crowded space, between robots, or near another object/goal boundary, include explicit clearance waypoints: lift at the current/object XY, move above the destination at a safe height, then descend/place. Use `world_summary.motion_model.suggested_transfer_clearance_z` as guidance, not as a mandatory constant.
- Omit a post-grip lift only when the transfer is extremely local and the current EEF Z is already clearly high enough to keep the carried object above obstacles. In ordinary transport and all goal placement, use Z clearance.
- If feedback says world.py rejected or failed an action, treat the old sequence as non-executable. Replan with reachable poses, explicit Z lift, and safe goal release height.
- `left` is the bottom robot in the top-view image. `right` is the top robot.
