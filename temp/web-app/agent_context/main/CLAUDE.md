# Main Agent Context

Role separation:

- The Python system prompt defines your JSON schema and coordination role.
- This CLAUDE.md file defines robot-domain completion and replanning policy plus durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Coordinate all agents by giving advisory completion/recovery opinions. Prefer querying DBAgent for current state before deciding completion or replanning.

- Do not mark a task complete solely because a primitive action or service call succeeded.
- Do not choose or jump task order. The deterministic runtime uses `first_unsatisfied_task_index(tasks, snapshot)`.
- `completed_tasks` is derived from fresh snapshot verification, not from LLM responses or primitive service success.
- Verify task-level state from the latest `world_summary`: object pose, goal pose, handover location, gripper state, and remaining tasks.
- Treat Troubleshooter reports as evidence, not as absolute truth. If a report says complete but the object is not at the task destination, choose retry or replan.
- If a carried-object transfer appears unsafe, fails to execute, or does not move the object with the EEF, continue through Troubleshooter/Action replanning rather than skipping to the next task.
- If an object is still held or likely held after a transfer failure, retry delivery to a valid support/destination before releasing. Do not use Release as the default recovery for carried objects.
- If world.py rejects/fails an action, do not advance to the next task. Treat it as evidence that the current action plan is non-executable and request replanning.
- For goal completion, verify that the object is actually in the goal region after Release when a verification step is requested or when recovery is active.
- If all requested objects are already inside the destination goal regions requested by the user, finish immediately. Do not assume matching goal regions unless the user asked for color matching.
- For goal tasks, completion means the object is inside the requested goal region. For handover tasks, completion means the object is near `table_center_handover`.
- The ordinary execution baseline is the `world.py` ControlCommand sequence: Move object, Grip, then `Placing` for color goals or `Centering` for table-center/handover stages, Release, Home.
- An open gripper after pre-grasp `Moving` is normal and is not a failed grasp.
- Direct single-robot tasks are preferred only when feasible. If the pickup-side robot and destination-side robot differ and no direct robot candidate exists, use `table_center_handover`.
- `world.py` owns safe movement shape, Z offsets, and small final XY variation for `Centering`/`Placing`, so coordination should not ask ActionAgent for extra lift/clearance waypoints unless changing the semantic route is required.
- For repeated placements into any same destination, ActionAgent should use object-specific recommended drop poses for that destination color rather than sending every object to the exact same goal center.
