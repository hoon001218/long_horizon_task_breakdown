# Troubleshooter Agent Memory

- No persistent memories yet.

- 2026-06-16 21:04:20 The robot's action to move 'cube_2:4' to 'blue_goal:2' was successful, but the state of the cube's position needs to be verified as it may not have reached the intended goal. A retry is recommended to ensure the task is completed correctly.
- 2026-06-16 A prior diagnosis returned `complete` immediately after the first pre-grasp `Moving`, which stopped the rest of the sequence. Intermediate primitives must return `success`, not `complete`.

- 2026-06-16 21:14:51 The action 'Release' was unsuccessful, and a stop action was executed instead. A replan is needed to attempt the task again.
- 2026-06-16 A queued Moving service response is not proof of arrival. If the observed EEF pose is still moving or not enough time has passed, return `pending`; diagnose retry/replan only when the current primitive clearly failed.
- 2026-06-16 Direct low-Z horizontal sweeps while carrying can collide with intermediate objects. Recommend lifted/clearance waypoints when transfer movement crosses occupied table space.

- 2026-06-16 21:47:53 Correction: after a pre-grasp `Moving` before `Grip`, an open gripper is normal. Do not diagnose failed grasp from an open gripper until a `Grip` or post-grip lift/transfer has occurred.

- 2026-06-16 21:48:09 Correction: repeated pre-grasp `Moving` retries were caused by misinterpreting open gripper as failed grasp. For pre-grasp `Moving`, return success if EEF reached the object.

- 2026-06-16 21:48:28 Correction: do not require gripper closure during a `Moving` action that precedes `Grip`.
- 2026-06-16 Do not diagnose only the current primitive in isolation. Compare the primitive, natural-language task, and remaining planned actions; return retry/replan if the primitive is locally successful but does not advance the task stage or makes the remaining plan unsafe.
- 2026-06-16 After `Grip`, carried-object transfer should be checked for object-following and sufficient Z clearance. Low direct XY transfer can drag the held object into the table, goal lip, guards, or other objects.
- 2026-06-16 Avoid excessive `pending`: if `execution_diagnostics.eef_reach_assessment.relaxed_reached` is true and joint motion is not clearly ongoing, treat the primitive as arrived enough and judge success/replan from task-level evidence.
- 2026-06-16 A `Grip` after only moving above the object is not a successful grasp. The plan must descend to the object's current pose/grasp height immediately before `Grip`; a closed gripper in the air should trigger retry/replan.
- 2026-06-16 A world.py ControlCommand failure/rejection must not be diagnosed as success. Request retry/replan with safer reachable target poses.
- 2026-06-16 Goal Release after descending to physical marker z is unsafe. Diagnose it as a plan error and request a higher release pose at or above `minimum_safe_release_z`.
- 2026-06-16 If a pre-grip `Moving` is followed by `Grip`, check Z alignment tightly. A 5 cm EEF-object Z gap is hovering, not a successful descent.
- 2026-06-16 Do not infer `Grip` success from gripper closure alone. Use `grip_alignment_assessment` and later object-following during lift/transfer.

- 2026-06-16 23:12:14 The current Moving action did not successfully reach the target object in both XY and Z. The end effector is still above the object, and the relaxed reach assessment indicates that the target was not reached. A retry is needed to ensure the end effector descends to the object's grasp height before attempting to grip.
