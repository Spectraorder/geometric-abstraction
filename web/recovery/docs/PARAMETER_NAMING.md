# Parameter names and lineage

Readable experiment identity:

`task__guard__reobs__wait_ms__fault__roi__frames__seed`

- `guard`: none, rgb, or rgbd-height.
- `reobs`: extra initial observations, not grasp attempts; candidates use 2 × 200 ms.
- Height-filtered guard: 8 valid pixels, 4–150 mm above the estimated table.
- `roi`: broad-rgb is the background-contaminated pilot; tabletop is target-only proxy occlusion.
- `frames`: masked side-camera calls, not video frames; historical values are 2 and 999.
- Miss: first closure replaced by 0.8 rad opening. Lift loss: open for 450 ms after 35 mm lift.
- Seeds 4001 and 4002 are development examples.

Configuration SHA-256, full policy SHA-256, parent policy SHA-256 and injector SHA-256 disambiguate artifacts. Repeated configurations need separate attempt IDs. Raw directory names and evidence filenames are kept as legacy identifiers.

Lineage: frozen baseline → color visibility guard with bounded reobservation → corrected test exposing background confusion → table-height-filtered guard. Correcting the injector is a benchmark change, not policy learning. Miss and lift-loss trials are separate validation branches using an existing retry skill.

The new dual-system protocol uses time-based occlusion instead of camera-call masks; do not pool it with these 15 historical trials.
