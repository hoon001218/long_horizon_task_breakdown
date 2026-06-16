# Main Agent Memory

- 2026-06-16 Correction: do not accept a Troubleshooter `complete` report or Release success as task completion unless the latest world state shows the object is actually at the task destination or handover location.
- 2026-06-16 If a task fails because a carried object moved in low XY without enough Z clearance, request ActionAgent replanning with lift/clearance waypoints instead of advancing to the next task.
- 2026-06-16 If world.py rejects a ControlCommand or reports action failure, do not advance tasks. Replan the same task with safer/reachable poses.
- 2026-06-16 For goal placement, do not count Release at physical goal marker height as safe completion; require the object to be observed in the goal region after a safe-height release.
- 2026-06-16 An open gripper after pre-grasp `Moving` is normal and should not be treated as failed grasp.
- 2026-06-16 Do not infer grasp success from gripper closure alone. Require descent to object pose and later object-following during lift/transfer.
- 2026-06-16 Direct single-robot tasks are preferred when a robot can reach both the object and its target goal; handover is only for no-direct-candidate or failed-direct cases.

- 2026-06-16 23:12:14 The current Moving action did not successfully reach the target object in both XY and Z. The end effector is still above the object, and the relaxed reach assessment indicates that the target was not reached. A retry is needed to ensure the end effector descends to the object's grasp height before attempting to grip.
