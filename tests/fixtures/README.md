`occluded_red_depth.npz` is a 14×15 crop of side-camera observation 54 from
`outputs/visual_retry_02/fast_compensated/training/seed_105`. Intrinsics were
translated into crop coordinates. It contains only RGB-D, camera calibration,
and capture time; no object state. The partially hidden face previously caused
a median over an empty normal-consistency subset and a NaN cube centroid.

`inventory_six_rgbd.npz` is side camera observation 1 from sampled inventory
seed 604, six objects, recorded in `outputs/inventory_transfer_01/adapted_six`.
It includes all five shapes and four red instances. Only RGB-D, camera
calibration and time are stored; no simulator object metadata is present.

`column_blue_cylinder_rgbd.npz` is side camera observation 1 from the archived
three-level random column, seed 826, in `outputs/columns_01/random_reserved`.
The original policy classified a small visible blue cylinder as unknown. This
fixture tests recovery from curved surface evidence. It contains only RGB-D,
camera calibration and capture time. This earlier evaluation example is now
explicitly development data; new policy evaluations use fresh reserved seeds.

`column_cylinder_slip_rgbd.npz` is side camera observation 64 from corrected
pixel-center sensor development trial `calibrated_v11/mixed_triple/seed_902`.
It stores the original RGB-D/calibration plus a prior cylinder center/normal
propagated from camera 61 using logged robot proprioception. Radius and height
were fitted from camera 59; the table plane was fitted from its depth. These
extra arrays are sensor-derived estimates, not simulator object state. Instance
isolation hides the top cap; visible side normals independently reveal about
13 degrees of slip relative to the rigid-grasp prediction.

`column_wrist_cube_rgb.npz` is wrist observation 14 from the development trial
`colors_v15/cube_pair_d0_colors/seed_907`. It contains RGB, camera calibration and
capture time only; the wrist sensor has no depth channel. The accompanying
`column_wrist_cube_prior.json` contains the policy's side-camera geometry estimate
at that time, including learned dimensions and explicitly unconstrained axes.
It contains no simulator truth. The physical cube had slipped along an axis not
observed by the side fit; its complete wrist silhouette resolves that translation.
The regression test's expected center is an offline replay oracle and is never
passed into the fitting function.

`column_cube_edge_907_rgbd.npz` and `column_cube_edge_908_rgbd.npz` are side
observations 35 and 28 from development `v18_colors/cube_pair_d0_colors`, seeds
907 and 908. They contain RGB-D, calibration and time, plus `learned_height_m`
and `learned_table_plane` fitted from observation 1. No simulator state is
stored. The green cube masks contain 65 and 72 pixels; interior face RMS errors
are 2.50 and 2.26 micrometers. Exactly one silhouette sample extends 4.73 and
2.65 mm beyond a fitted face. Interpolated 1st/99th percentiles retain part of
that sample, producing a false extent or face-plane conflict. Inward-rounded
empirical quantiles reject the isolated edge sample while still rejecting a
model whose learned side is 5 mm too short.

`column_cube_double_edge_rgbd.npz` is side observation 44 from development
`v19_colors/cube_triple_d0_colors/seed_907`. Its green cube has 61 colored pixels
and two measured faces with 1.31 micrometer RMS. Two adjacent silhouette depth
samples lie 1.91 and 5.85 mm in front of the measured top plane. Their rejection
uses the independently measured outward half-spaces, before applying learned
finite dimensions; legitimate narrow third-face pixels remain available.
The stored height and table plane are RGB-D estimates from observation 1.

`column_thin_width_903_{initial,later}.npz` and
`column_thin_width_904_{initial,later}.npz` preserve isolated side RGB-D from
`v21_random/random_l3` development seeds 903 and 904: initial observation 1 and
later observations 72 and 65. Initial masks use visual instance tracking; later
masks reproduce the policy's recorded isolated surface points. Stored table
planes and dimensions are the policy's initial visual estimates. The optional
904 prior is its robot-propagated visual pose, with no simulator object state.
The 903 model's 13.5809 mm width is contradicted by later finite evidence by
0.0429 mm after tolerance; the 904 model's 29.4283 mm width by 0.1094 mm. All 73
and 86 later pixels pass the independent observed-plane half-space filter,
showing genuine silhouette sampling uncertainty rather than edge outliers.
The tests permit minimal horizontal size refinement only inside calibrated
initial pixel bounds, and keep fixed-size contradictions and height strict.
