# Main Agent Context

Role separation:

- The Python system prompt defines your JSON schema and coordination role.
- This CLAUDE.md file defines robot-domain completion and replanning policy plus durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Coordinate all agents. Prefer querying DBAgent for current state before deciding completion or replanning.

- Do not mark a task complete solely because a primitive action or service call succeeded.
- Verify task-level state from the latest `world_summary`: object pose, goal pose, handover location, gripper state, and remaining tasks.
- Treat Troubleshooter reports as evidence, not as absolute truth. If a report says complete but the object is not at the task destination, choose retry or replan.
- If a carried-object transfer appears unsafe, fails to execute, or does not move the object with the EEF, continue through Troubleshooter/Action replanning rather than skipping to the next task.
- If world.py rejects/fails an action, do not advance to the next task. Treat it as evidence that the current action plan is non-executable and request replanning.
- For goal completion, verify that the object is actually in the goal region after Release when a verification step is requested or when recovery is active.
- If all requested color-sort objects are already inside their matching goal regions, finish immediately. Do not ask ActionAgent to pick up and re-place objects that are already sorted.
- The ordinary execution baseline is the `world.py` ControlCommand sequence: Move object, Grip, then `Placing` for color goals or `Centering` for table-center/handover stages, Release, Home.
- An open gripper after pre-grasp `Moving` is normal and is not a failed grasp.
- Direct single-robot tasks are preferred only when feasible. If the pickup-side robot and destination-side robot differ and no direct robot candidate exists, use `table_center_handover`.
- `world.py` owns safe movement shape, Z offsets, and small final XY variation for `Centering`/`Placing`, so coordination should not ask ActionAgent for extra lift/clearance waypoints unless changing the semantic route is required.
- For repeated same-color placements, ActionAgent should use object-specific recommended drop poses from `world_summary.objects_by_id` rather than sending every object to the exact same goal center.
