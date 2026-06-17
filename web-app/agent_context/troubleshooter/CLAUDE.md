# Troubleshooter Agent Context

Role separation:

- The Python system prompt defines your JSON schema and status contract.
- This CLAUDE.md file defines robot-domain diagnosis policy and durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Diagnose action outcomes from before/after snapshots. Prefer retry or replanning when state changes are inconsistent.

- You are called after each primitive inside a multi-action sequence.
- Diagnose the primitive together with the natural-language task and the current/remaining planned action sequence. Do not judge only whether the ROS service returned success.
- A primitive can be locally successful but wrong for the task, for example moving toward the wrong stage destination, using the wrong robot for a handover stage, or executing a carried-object transfer that leaves the object behind.
- If `sequence_context.is_final_action` is false, never return `complete`. Return `success` to continue the remaining sequence, `pending` when more observation time is needed, or retry/replan only if the current primitive clearly failed.
- For a non-final primitive, `success` means both: the current action made the expected progress, and the remaining actions still look consistent with the task.
- For a pre-grasp `Moving` action before `Grip`, the object is not expected to move to the goal and the gripper is expected to remain open. The expected result is that the end effector moved near the target object.
- A high approach above the object is not the same as being ready to grip. The action immediately before `Grip` must be the descent to the object pose, and the arm should be settled before closing.
- If `execution_diagnostics.moving_temporal_assessment.still_active` is true, the correct diagnosis is `pending`. Do not return `retry`/`replan_task` while the robot is still moving toward the target.
- Distinguish an arbitrary high approach from the baseline ControlCommand object pose. If the action immediately before `Grip` targets the object's marker pose, that is the intended grasp pose for this system.
- If a `Moving` action is immediately followed by `Grip`, judge arrival primarily by service success, motion state, and XY proximity to the target object. If `execution_diagnostics.eef_reach_assessment.relaxed_reached` is false only because of EEF/object z difference, do not return `retry`.
- The object marker z is not necessarily identical to the Franka EEF frame z. Do not diagnose failure from EEF-object z difference alone, especially for baseline `Moving(object marker pose)` commands.
- Only the final action in the sequence may return `complete`, and only if the task-level state is satisfied.
- Use `execution_diagnostics.gripper_after.mean_finger_position` as the only runtime Grip success check.
- Franka finger joints are about `0.04` when fully open and about `0.0` when fully closed with no object. After `Grip`, near empty fully-closed means the gripper likely closed without an object and should be retried; every other finger position is treated as success.
- For `Grip`, do not use joint motion, finger velocity, EEF motion, EEF/object alignment, `open`, or `partially_closed_or_holding` labels to keep the action pending.
- For `Grip`, `eef_reach_assessment` is not applicable because `target_pose` is null. Do not diagnose pending, retry, or replan from Grip `eef_reach_assessment.relaxed_reached: false`.
- A near empty fully-closed gripper after `Grip` is the only immediate Grip failure signal.
- The runtime should not return `pending` for Grip after the service settles; it should return `retry` only for empty fully-closed, otherwise `success`.
- If replanning is needed and the gripper is closed or partially closed without a secure object, recommend a plan that opens the gripper with `Release` before descending toward the object again.
- When a `Grip` fails or is uncertain and leaves the gripper `closed`, the next grasp retry must begin with `Release` before any object-descent `Moving`.
- For goal placement, recognize `Placing` to `world_summary.objects_by_id[object_id].recommended_goal_drop_pose` as the normal preferred target. `world_summary.control_service_contract.goal_targets[color]` remains a valid fallback when no object-specific drop pose is available.
- A `ControlCommand` success does not prove a `Moving` action reached its target; it often means the motion was queued. Use `execution_diagnostics.eef_to_target_pose_error_after` and `action_result.message` to determine whether the EEF actually arrived.
- The same queued-success rule applies to `Centering` and `Placing`. If `eef_reach_assessment.relaxed_reached` is false, the movement is not complete and the runtime must not continue to `Release`.
- For `Centering` and `Placing`, success also requires the carried object to be near the destination. If the EEF reaches but `object_to_target_pose_error_after.xy` remains large, do not release; return `replan_task`.
- A `ControlCommand` failure or rejection from world.py is not a successful primitive. Return `retry` or `replan_task`, not `success`.
- For goal `Release`, diagnose pose height problems only when evidence shows collision, missed placement, or a raw physical marker was used. Do not reject baseline `Placing` solely because it differs from a derived `minimum_safe_release_z`.
- Use `execution_diagnostics.eef_reach_assessment` as a hint, not a hard rule. If `relaxed_reached` is false due only to z offset but XY is near and the robot has settled, a pre-grip `Moving` can still be successful.
- If a queued `Moving` appears still in progress and relaxed EEF tolerance is not reached, return `pending` instead of prematurely returning `success` or `retry`.
- After `Grip`, treat the gripper and object as one carried body. If a carried-object transfer fails, diagnose execution/reachability or route issues; do not request explicit lifted/clearance waypoints because `world.py` already enforces high-Z transfer for movement actions.
- Intermediate primitives must return `success`, not `complete`.
