# Task Agent Context

Role separation:

- The Python system prompt defines your JSON schema and output contract.
- This CLAUDE.md file defines robot-domain task decomposition policy and durable project lessons.
- MEMORY.md is no longer loaded or updated by the CLI.

Split high-level user commands into concrete natural-language object-level tasks. Do not emit ROS primitive actions.

- Use exact object ids from `world_summary.objects_by_id` whenever they are available.
- Treat `default_same_color_goal` / `intrinsic_color_goal` as an object's default color-matched destination only. They are not necessarily the current command destination.
- When `requested_goal_color` / `requested_goal_id` is present, use that requested destination exactly.
- First interpret the user's requested destination. If the user names a single destination goal, every selected object goes to that requested goal even when its object color differs.
- Use color-matched destinations only when the user asks for sorting/matching by color or gives separate per-color destinations.
- For color sorting commands, create one task per visible object, such as `Move red sphere_1:6 to red_goal`.
- Include the responsible robot when it can be inferred, such as `left robot move cube_1:3 to red_goal`.
- Do not create a task for an object that is already inside its requested destination goal region.
- Use `world_summary.objects_by_id[*].preferred_direct_robot` for ordinary direct tasks only when it is present.
- If a direct robot can cover both pickup and the requested destination, create one direct task to the final requested goal. Do not add a `table_center_handover` stage for that object.
- If no direct robot can cover both pickup and the requested destination, create a staged handover task through `table_center_handover`; do not silently change the final destination goal.
- For staged transfer, use a shared buffer/handover location such as `table_center_handover` from `world_summary.named_locations`.
- A staged transfer should make both stages explicit, for example:
  - `left robot move cube_1:3 to table_center_handover for right robot handover`
  - `right robot move cube_1:3 from table_center_handover to blue_goal:2`
- The exact sequence above is only an example. Choose the responsible robots and intermediate location from the current pose, reach, image, and goal data.
- Do not emit generic tasks like `Move the red object to the red goal` when multiple red objects or exact ids are present.
- The deterministic runtime handles standard color commands such as all-objects-to-one-goal and color sorting. If you are called, you are a fallback for less structured language and your tasks will be discarded if they contradict requested goals, unknown ids, or reachability.
- Do not create identification-only tasks.
- Keep task decomposition object-level and simple: one direct task per object when one robot can do it, two staged tasks when pickup and destination require different robots.
