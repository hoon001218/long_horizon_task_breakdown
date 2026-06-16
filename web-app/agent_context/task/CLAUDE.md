# Task Agent Context

Split high-level user commands into concrete natural-language object-level tasks. Do not emit ROS primitive actions.

- Use exact object ids from `world_summary.objects_by_id` whenever they are available.
- For color sorting commands, create one task per visible object, such as `Move red sphere_1:6 to red_goal`.
- Include the responsible robot when it can be inferred, such as `left robot move cube_1:3 to red_goal`.
- Use `world_summary.objects_by_id[*].nearest_robot`, `destination_nearest_robot`, `route_hint`, `goals[*].nearest_robot`, and `named_locations` to decide whether an object can be handled directly or needs staged transfer.
- Prefer a direct single-robot task whenever `world_summary.objects_by_id[*].direct_robot_candidates` is non-empty.
- If `preferred_direct_robot` is present, use that robot for a direct task to the final color goal. Do not send the object to `table_center_handover`.
- A difference between `nearest_robot` and `destination_nearest_robot` is only weak evidence. It is not enough by itself to create a handover.
- Split the object-level goal into route-stage tasks only when `handover_needed` is true, `direct_robot_candidates` is empty, or a previous direct attempt failed.
- For staged transfer, use a shared buffer/handover location such as `table_center_handover` or another explicitly named safe intermediate location from `world_summary.named_locations`.
- A staged transfer should make both stages explicit, for example:
  - `left robot move cube_1:3 to table_center_handover for right robot handover`
  - `right robot move cube_1:3 from table_center_handover to blue_goal:2`
- The exact sequence above is only an example. Choose the responsible robots and intermediate location from the current pose, reach, image, and goal data.
- Do not emit generic tasks like `Move the red object to the red goal` when multiple red objects or exact ids are present.
- Do not create identification-only tasks.
