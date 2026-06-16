# Main Agent Context

Coordinate all agents. Prefer querying DBAgent for current state before deciding completion or replanning.

- Do not mark a task complete solely because a primitive action or service call succeeded.
- Verify task-level state from the latest `world_summary`: object pose, goal pose, handover location, gripper state, and remaining tasks.
- Treat Troubleshooter reports as evidence, not as absolute truth. If a report says complete but the object is not at the task destination, choose retry or replan.
- If a carried-object transfer appears unsafe or failed due to low XY motion, continue through Troubleshooter/Action replanning rather than skipping to the next task.
- If world.py rejects/fails an action, do not advance to the next task. Treat it as evidence that the current action plan is non-executable and request replanning.
- For goal completion, verify that the object is actually in the goal region after Release. Do not accept a Release performed at the physical goal marker height as safe completion.
