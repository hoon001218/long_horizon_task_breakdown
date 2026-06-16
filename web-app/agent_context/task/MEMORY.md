# Task Agent Memory

- No persistent memories yet.
- 2026-06-16 A generic task such as `Move the red object to the red goal` was too ambiguous and led to a wrong first action. Use exact object ids in every generated sorting task.
- 2026-06-16 Handover is exceptional. If `direct_robot_candidates` is non-empty, create a direct single-robot task with `preferred_direct_robot` instead of sending the object to `table_center_handover`.
- 2026-06-16 Use handover/buffer subtasks only when `handover_needed` is true, no direct robot candidate exists, or a direct attempt has actually failed.

- 2026-06-16 23:12:14 The current Moving action did not successfully reach the target object in both XY and Z. The end effector is still above the object, and the relaxed reach assessment indicates that the target was not reached. A retry is needed to ensure the end effector descends to the object's grasp height before attempting to grip.
