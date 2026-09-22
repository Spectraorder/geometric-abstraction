`occluded_red_depth.npz` is a 14×15 crop of side-camera observation 54 from
`outputs/visual_retry_02/fast_compensated/training/seed_105`. Intrinsics were
translated into crop coordinates. It contains only RGB-D, camera calibration,
and capture time; no object state. The partially hidden face previously caused
a median over an empty normal-consistency subset and a NaN cube centroid.
