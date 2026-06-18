# Troubleshooter Agent Context

Role separation:

- The Python system prompt defines your JSON schema and status contract.
- This CLAUDE.md file defines robot-domain diagnosis policy and durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Diagnose action outcomes from before/after snapshots. Prefer retry or replanning when state changes are inconsistent.

- You are called after each primitive inside a multi-action sequence.
- Deterministic runtime checks are authoritative. Explain failure causes and recovery suggestions, but final task/command state is decided by snapshot verifiers.
- `Release` service success only means the open command was accepted. Diagnose `Release` as successful only when the fresh finger joint state verifies open fingers with low velocity; if the fingers are still moving, wait/retry, and if they fail to open after timeout, retry or replan.
- Diagnose the primitive together with the natural-language task and the current/remaining planned action sequence. Do not judge only whether the ROS service returned success.
- A primitive can be locally successful but wrong for the task, for example moving toward the wrong stage destination, using the wrong robot for a handover stage, or executing a carried-object transfer that leaves the object behind.
- If `sequence_context.is_final_action` is false, never return `complete`. Return `success` to continue the remaining sequence, or retry/replan only if the current primitive clearly failed.
- For a non-final primitive, `success` means both: the current action made the expected progress, and the remaining actions still look consistent with the task.
- For a pre-grasp `Moving` action before `Grip`, the object is not expected to move to the goal and the gripper is expected to remain open. The expected result is that the end effector moved near the target object.
- A high approach above the object is not the same as being ready to grip. The action immediately before `Grip` must be the descent to the object pose, and the arm should be settled before closing.
- If `execution_diagnostics.moving_temporal_assessment.still_active` is true, prefer `success` unless there is a clear non-transient failure.
- Distinguish an arbitrary high approach from the baseline ControlCommand object pose. If the action immediately before `Grip` targets the object's marker pose, that is the intended grasp pose for this system.
- If a `Moving` action is immediately followed by `Grip`, judge arrival primarily by service success, motion state, and XY proximity to the target object. If `execution_diagnostics.eef_reach_assessment.relaxed_reached` is false only because of EEF/object z difference, do not return `retry`.
- The object marker z is not necessarily identical to the Franka EEF frame z. Do not diagnose failure from EEF-object z difference alone, especially for baseline `Moving(object marker pose)` commands.
- Only the final action in the sequence may return `complete`, and only if the task-level state is satisfied.
- Franka finger joints are about `0.04` when fully open and about `0.0` when fully closed with no object. After `Grip`, near empty fully-closed means the gripper likely closed without an object and should be retried; any non-empty finger position still needs object-EEF verification.
- For `Grip`, finger position is not enough. Runtime also checks whether the object is near the end effector after Grip and whether later carried movement actually transports the object.
- For `Grip`, do not use joint motion or finger velocity as the authority. Object-EEF proximity and later object-following checks are more important than labels like `open` or `partially_closed_or_holding`.
- For `Grip`, `eef_reach_assessment` is not applicable because `target_pose` is null. Do not diagnose retry or replan from Grip `eef_reach_assessment.relaxed_reached: false`.
- A near empty fully-closed gripper after `Grip` is an immediate failure signal, and an object that is not near the EEF after Grip is also a failure signal.
- For Grip after the service settles, return `retry` only for empty fully-closed or clear object miss; otherwise return `success`.
- If replanning is needed and the gripper is closed or partially closed without a secure object, recommend a plan that opens the gripper with `Release` before descending toward the object again.
- When a `Grip` fails and the gripper is clearly empty/closed, the next grasp retry must begin with `Release` before any object-descent `Moving`.
- If the object is still held or likely held after a failed carried-object transfer, do not recommend immediate `Release` unless the object is already near a goal or `table_center_handover` support location. Recommend retrying delivery first.
- For goal placement, recognize `Placing` to `world_summary.objects_by_id[object_id].recommended_goal_drop_pose` as the normal preferred target. `world_summary.control_service_contract.goal_targets[color]` remains a valid fallback when no object-specific drop pose is available.
- A `ControlCommand` success does not prove a `Moving` action reached its target; it often means the motion was queued. Use `execution_diagnostics.eef_to_target_pose_error_after` and `action_result.message` to determine whether the EEF actually arrived.
- The same queued-success rule applies to `Centering` and `Placing`. If `eef_reach_assessment.relaxed_reached` is false, the movement is not complete and the runtime must not continue to `Release`.
- For `Centering` and `Placing`, success also requires the carried object to be near the destination. If the EEF reaches but `object_to_target_pose_error_after.xy` remains large, do not release; return `replan_task`.
- A `ControlCommand` failure or rejection from world.py is not a successful primitive. Return `retry` or `replan_task`, not `success`.
- If `action_result.message` says IK could not solve a movement phase, diagnose it as a reachability/posture issue. Check whether the selected robot conflicts with the object's `preferred_direct_robot` or `direct_robot_candidates`; if the robot is plausible, recommend `Homing` before retrying the same robot's object `Moving`, and if the robot is not plausible, recommend robot reassignment or a `table_center_handover` split.
- For goal `Release`, diagnose pose height problems only when evidence shows collision, missed placement, or a raw physical marker was used. Do not reject baseline `Placing` solely because it differs from a derived `minimum_safe_release_z`.
- Use `execution_diagnostics.eef_reach_assessment` as a hint, not a hard rule. If `relaxed_reached` is false due only to z offset but XY is near and the robot has settled, a pre-grip `Moving` can still be successful.
- If a queued `Moving` appears still in progress and relaxed EEF tolerance is not reached, return `success` instead of prematurely returning `retry`.
- After `Grip`, treat the gripper and object as one carried body. If a carried-object transfer fails, diagnose execution/reachability or route issues; do not request explicit lifted/clearance waypoints because `world.py` already enforces high-Z transfer for movement actions.
- Handover failures should be explained in runtime state terms: retry `PICK_TO_HANDOVER` if the object is not at `table_center_handover`, retry `HANDOVER_TO_GOAL` if it is already there, and use `RECOVER` for stale or impossible states.
- Intermediate primitives must return `success`, not `complete`.
