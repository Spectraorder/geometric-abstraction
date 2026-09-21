You are the policy learner for a new SO-100 three-block manipulation experiment.
No prior policy or skill library is supplied. Use only this task contract, attached sensor images,
and previous attempts from THIS experiment when supplied below. Never use tools or inspect files.
Return JSON with hypothesis and code; code defines run(api). Only numpy, math, and scipy imports.
No file/network/process/introspection. Do not access private attributes. All diagnostics via api.note.
TASK: blue block stays on table; pick red and place on blue; then pick green and place on red.
All are free rigid bodies. Objects must be grasped, physically transported, released, and robot withdrawn.
After placing red, leave it untouched for at least 3 seconds before picking green. After green, withdraw
and wait at least 6 seconds. A separate evaluator observes support, alignment, release, and drift.
The policy cannot query evaluator or read any object ground truth. You receive only success/failure AFTER a trial.

SENSOR API:
api.camera() -> rgb uint8[240,320,3], depth_m float32[240,320], valid_depth bool,
intrinsics 3x3, camera_to_world 4x4, time. Ideal side-mounted calibrated RGB-D.
Depth is optical-axis z in meters, NOT range. For pixel (u,v):
world=R @ (inv(K) @ [u,v,1] * depth[v,u]) + t. Camera axes right,down,forward.
Use RGB segmentation and point clouds to estimate object positions/heights; NO preknown object coordinates or sizes.
The table and objects are visible in the attached initial image. Object colors are blue, red, green.
api.proprioception() -> joint_angles length5, tool_position world XYZ meters (robot forward kinematics), time.

ACTUATION API (robot SDK, no object perception or hidden grasp heuristics):
api.move_tool([x,y,z], seconds=1.5): position with soft preference for downward gripper,
box-constrained robot-model IK; may tilt near reach limits. It moves joints along a smooth interpolation.
Returns no object state or grasp result. Site is an approximate gripper center, not a guaranteed object center.
api.move_joints(five_angles, seconds=1.5) optional raw control; joint limits radians
[-1.92,1.92],[-3.32,.174],[-.174,3.14],[-1.66,1.66],[-2.79,2.79].
api.gripper(angle, seconds=.6): jaw angle in [-.174,1.75], larger is more open; .8 is open.
api.wait(seconds): advances physics, max10s per call. api.note(text): logs your perception/decision (no state oracle).
All motion durations 0.1..10s. All locations MUST come from current camera measurements, not constants keyed
by seed or object names. Known camera calibration and measured robot joint/tool pose are allowed.
No api.get_observation(), no api.check_structure(), no privileged object state, no forced attachment.
Pixel and pointcloud operations may use numpy and scipy.ndimage. Do not print; use api.note for diagnostics.
There is a 120s wall-time, 600 RPC call limit per trial. Robot base is fixed; workspace actions should be conservative.
Develop a physically plausible policy and adjust it using subsequent rendered outcomes and your logged estimates.

Design the next policy. Return complete code, not a patch.