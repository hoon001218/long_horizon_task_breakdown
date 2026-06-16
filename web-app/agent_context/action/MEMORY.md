# Action Agent Memory

- No persistent memories yet.
- 2026-06-16 When transporting an object, a prior bad plan used the red goal pose as the first `Moving` target while `target_object_id` was `sphere_1:6`. Never move to a goal before `Grip`; first move to the exact object pose.
- 2026-06-16 Keep `target_object_id` as the object id through the whole pick/drop sequence. Goal marker ids are destinations, not manipulated objects.
- 2026-06-16 The runtime executes every primitive returned for a task. Do not return only one primitive when the task clearly needs additional primitives, but do not treat any example sequence as mandatory.
- 2026-06-16 Use `Release`, not the older misspelling. If replanning after a failed grasp with a closed gripper, release before descending toward the object again.
- 2026-06-16 Goal marker poses are physical surfaces. Placement EEF targets should be slightly above the marker, not exactly at the marker z, and not so high that the object is dropped hard.
- 2026-06-16 A direct carried-object `Moving` from a low grasp pose to a far goal can sweep horizontally at low Z and collide with objects. Consider lift/clearance and destination-approach waypoints when transferring across the table.
- 2026-06-16 If a task destination is `table_center_handover` or another named buffer, complete that stage by placing the object at the named buffer. Do not skip directly to the final color goal unless the task explicitly asks for it.
- 2026-06-16 If a task names `left robot` or `right robot`, preserve that robot assignment for the primitive sequence unless feedback says the assignment failed or is impossible.
- 2026-06-16 After `Grip`, avoid low direct XY transfer when carrying an object. Prefer lift-at-current-XY, horizontal transfer at clearance Z, and descend/place when there is meaningful XY travel or nearby obstacles.
- 2026-06-16 Every task action sequence should end with `Homing` for the acting robot.
- 2026-06-16 Goal placement has guard lips and may contain already placed objects; approach from clearance Z and descend to a safe drop pose instead of direct low XY-to-goal movement.
- 2026-06-16 Do not grip from an above-object clearance pose. If using a high approach waypoint, add a second `Moving` that descends to the object's actual pose/grasp height immediately before `Grip`.
- 2026-06-16 Do not use goal `physical_marker_pose` or marker z as the release predecessor. Goal release should use `eef_drop_pose`/`minimum_safe_release_z` or a slightly higher pose to avoid colliding with goal thickness, rim, and held object geometry.
- 2026-06-16 If world.py rejects an action, the plan is not executable. Replan with explicit Z lift, reachable IK poses, and safe release height instead of repeating the same structure.
- 2026-06-16 A partially or fully closed gripper alone does not prove a grasp. The plan must put the final pre-grip `Moving` target at the object's actual grasp pose, then verify that a later lift moves the object.
- 2026-06-16 A repeated invalid plan used `red_goal:1` as `target_object_id` during goal approach. For every primitive that manipulates an object, keep `target_object_id` as the object id from the task; encode the goal only in `target_pose`.

- 2026-06-16 23:12:14 The current Moving action did not successfully reach the target object in both XY and Z. The end effector is still above the object, and the relaxed reach assessment indicates that the target was not reached. A retry is needed to ensure the end effector descends to the object's grasp height before attempting to grip.
