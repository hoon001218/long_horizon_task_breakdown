# Action Agent Context

Role separation:

- The Python system prompt defines your JSON schema and output contract.
- This CLAUDE.md file defines robot-domain planning policy and durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Convert one natural-language task into minimal executable ROS2 primitive actions using only robot_id, action, target_pose, and target_object_id.

- The default object transport plan should follow the `world.py` ControlCommand service baseline: `Moving(object marker pose)` -> `Grip` -> `Placing(goal pose)` or `Centering(table center pose)` -> `Release` -> `Homing`.
- Goal selection, robot selection, handover choice, and requested destination interpretation are decided before you are called by the deterministic runtime planner. Do not change them.
- Use `Placing` for red/blue goal destinations. Use `Centering` for table center or handover destinations.
- The destination goal comes from the task text/user command. Do not replace an explicit destination with the manipulated object's matching color goal.
- `default_same_color_goal` / `intrinsic_color_goal` are not authority to change the destination. Use `requested_goal_color` when supplied and the goal named in the task.
- Do not add extra approach, lift, clearance, horizontal-transfer, or descent waypoints. `world.py` enforces vertical rise to home-level Z, high-Z horizontal travel, vertical descent, and a small final XY variation for every `Moving`, `Centering`, and `Placing` action.
- `Moving` target_pose should be the object marker pose; `world.py` descends slightly below it for grasping. `Centering` and `Placing` target_pose should be a safe semantic drop pose near the destination; `world.py` descends slightly above it to avoid collision and applies a small final XY variation.
- Do not compensate for Z offsets in the prompt or plan. Supply semantic target poses only.
- Never return only the next primitive when the current task clearly still requires additional primitives.
- Every task action sequence must end with `Homing` for the same robot, with `target_pose: null` and `target_object_id: null`. After an IK failure, a recovery sequence may also begin with `Homing` for the selected robot before retrying `Moving`.
- If the task names a responsible robot, use that robot for the whole primitive sequence unless feedback explicitly says that robot cannot complete the stage.
- In a task like `right robot move <object> to table_center_handover for left robot handover`, the actor is the robot before `move` (`right`). The later robot mention is the handover recipient and must not change primitive `robot_id`.
- Task destinations may be named stage locations, not only final color goals. If the destination is `table_center_handover` or another key in `world_summary.named_locations`, place the object at that named location for this stage.
- If the task says `from table_center_handover to blue_goal` or similar, pick from the object's current observed marker pose in `world_summary.objects_by_id`; do not invent a stale pickup pose from the text alone.
- Handover/buffer stages are valid completed tasks. Do not silently replace a handover destination with the final color goal unless the task explicitly asks for the final goal.
- Handover recovery is managed by runtime states: `PICK_TO_HANDOVER`, `VERIFY_HANDOVER`, `HANDOVER_TO_GOAL`, `VERIFY_GOAL`, and `RECOVER`. Convert the current stage into primitives only.
- If feedback says direct `Placing`/`Centering` failed because the target is unreachable for the current robot, use `table_center_handover` with the opposite-side robot rather than repeating the same direct goal placement.
- The immediate action before `Grip` should be `Moving` to the object's current marker pose from `world_summary.objects_by_id`.
- Do not treat a high approach pose above the object as ready for `Grip`. The primitive immediately before `Grip` must be the actual object-pose `Moving` command; `world.py` performs the high approach and final descent internally.
- Before a new grasp attempt, use `Release` only to clear an empty or stale gripper. If the object is still held or likely held, first retry `Centering`/`Placing` to a valid support pose; do not drop it mid-transfer.
- Keep `target_object_id` equal to the manipulated object id for every primitive in the sequence, including `Moving` actions whose `target_pose` is inside a goal.
- Never put `red_goal:*` or `blue_goal:*` in `target_object_id`. Goal ids are destinations represented by `target_pose`, not manipulated objects.
- The service action spelling is `Release`.
- When replanning after a failed grasp or slip, add `Release` before descending toward the object again.
- For goal placement, prefer `Placing` with `world_summary.objects_by_id[object_id].recommended_goal_drop_poses_by_color[destination_color]` so objects going to the same requested goal do not all target the exact same XY. If that field is absent, fall back to `world_summary.control_service_contract.goal_targets[destination_color]`. Do not use the raw physical marker pose as the default.
- For `Centering` to `table_center_handover`, prefer `world_summary.objects_by_id[object_id].recommended_handover_drop_pose` or the named location's `drop_slots` instead of the exact table center.
- After `Grip`, the runtime verifies that the object is actually held by checking object-EEF proximity and later carried-object motion. Do not assume finger position alone proves a grasp.
- Runtime resolves `Moving`, `Centering`, and `Placing` target poses from a fresh snapshot immediately before execution. Treat your target poses as semantic hints, not durable coordinates.
- `Placing` and `Centering` require the selected robot to already hold or likely hold the target object. If the object is not held in the current world summary, plan `Release` only for empty/stale gripper cleanup, then `Moving(current object pose)` -> `Grip` -> `Placing`/`Centering`; never start with destination-only placement.
- If feedback says `Placing` or `Centering` was queued but the EEF did not move and stayed far from the target, treat the previous target as non-executable for that robot/pose. Replan with a reachable transfer, a safer intermediate pose, or a handover stage; do not repeat the identical target sequence.
- If feedback says world.py rejected or failed an action, treat the old sequence as non-executable. Replan with reachable semantic target poses or a handover stage instead of repeating the same structure.
- If feedback says IK could not solve a movement phase, first check whether the selected robot is appropriate using `world_summary.objects_by_id[object_id].preferred_direct_robot` and `direct_robot_candidates`. If the selected robot is plausible, begin recovery with `Homing` for that robot and then retry `Moving` to the object's current observed pose; if the selected robot is not plausible, reassign the stage or split through `table_center_handover`.
- If a task names `left robot` or `right robot`, preserve that robot assignment unless feedback says it failed or is impossible.
- Every task action sequence must end with `Homing`.
- The runtime executes every primitive returned for a task; return the complete sequence needed for the current task.
- `left` is the bottom robot in the top-view image. `right` is the top robot.
