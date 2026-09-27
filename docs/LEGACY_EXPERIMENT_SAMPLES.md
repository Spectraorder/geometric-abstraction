# Legacy experiment artifact retention

The user requested cleanup of older experiments, retaining one complete sample per task.
Cleanup reclaimed **186.14 GiB** from 5,587 completed trials in eleven legacy roots.
Task families distinguish support-graph recipes and column shape sequences; color
variants share a family. One successful sample was preferred when available, with
stable path order breaking ties. These samples are for replay, not a success-rate estimate.

Only `rollout.npz` and `camera_*.npz` / `camera_*.png` were deleted from non-sample
trials. All JSON results, configuration, source, trace/API logs, initial/outcome
images, and published videos remain. Raw sensor replay and physical playback are
no longer available for those pruned trials. Aggregate outcomes were not rewritten.

Current geometry learning experiments, schema/validator correction lineage,
the earlier replicated cube comparison, and active mixed-shape replications are
excluded. All 49 retained sample hashes and all pruned-trial result hashes were
verified after cleanup. The detailed receipt is
`outputs/maintenance/legacy_cleanup_20260926.json`.

| Task family | Retained complete sample |
|---|---|
| inventory_prism:upright_thin_prism | [outputs/columns_01/regression_inventory](../outputs/columns_01/regression_inventory) |
| inventory_structure:capped_bridge | [outputs/beam_bridge_01/upright_regression/capped_bridge_d0/seed_1401](../outputs/beam_bridge_01/upright_regression/capped_bridge_d0/seed_1401) |
| inventory_structure:cube_bridge | [outputs/beam_bridge_01/upright_regression/cube_bridge_d0/seed_1401](../outputs/beam_bridge_01/upright_regression/cube_bridge_d0/seed_1401) |
| inventory_structure:staircase | [outputs/beam_bridge_01/upright_regression/staircase_d0/seed_1401](../outputs/beam_bridge_01/upright_regression/staircase_d0/seed_1401) |
| inventory_structure:thin_beam_bridge | [outputs/beam_bridge_01/heldout/thin_beam_bridge_d0_colors/seed_1501](../outputs/beam_bridge_01/heldout/thin_beam_bridge_d0_colors/seed_1501) |
| inventory_tower:cube,cube | [outputs/columns_01/dev_v1_suite/cube_pair/seed_801](../outputs/columns_01/dev_v1_suite/cube_pair/seed_801) |
| inventory_tower:cube,cube,cube | [outputs/columns_01/clutter_reserved/cube_triple/seed_821](../outputs/columns_01/clutter_reserved/cube_triple/seed_821) |
| inventory_tower:cube,cube,cube,cube | [outputs/columns_01/dev_v4_suite/cube_quad/seed_801](../outputs/columns_01/dev_v4_suite/cube_quad/seed_801) |
| inventory_tower:cube,cube,cylinder | [outputs/columns_01/random_dev_v5/random/seed_805](../outputs/columns_01/random_dev_v5/random/seed_805) |
| inventory_tower:cube,cube,thin_prism | [outputs/columns_01/clutter_reserved/slender_top/seed_821](../outputs/columns_01/clutter_reserved/slender_top/seed_821) |
| inventory_tower:cube,cylinder | [outputs/columns_02/random2_v11/random_l2_d0/seed_905](../outputs/columns_02/random2_v11/random_l2_d0/seed_905) |
| inventory_tower:cube,cylinder,cylinder,cylinder | [outputs/columns_02/all_geometry_random4/random_l4/seed_905](../outputs/columns_02/all_geometry_random4/random_l4/seed_905) |
| inventory_tower:cube,cylinder,thin_prism | [outputs/columns_02/v20_random/random_l3/seed_905](../outputs/columns_02/v20_random/random_l3/seed_905) |
| inventory_tower:cube,prism | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1007](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1007) |
| inventory_tower:cube,prism,prism,thin_prism | [outputs/columns_02/all_geometry_random4/random_l4/seed_903](../outputs/columns_02/all_geometry_random4/random_l4/seed_903) |
| inventory_tower:cube,prism,thin_prism | [outputs/columns_02/all_geometry_v16/random_l3/seed_903](../outputs/columns_02/all_geometry_v16/random_l3/seed_903) |
| inventory_tower:cube,thin_prism | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1006](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1006) |
| inventory_tower:cylinder,cube | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1009](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1009) |
| inventory_tower:cylinder,cube,cube | [outputs/columns_01/dev_v1_suite/cylinder_base/seed_801](../outputs/columns_01/dev_v1_suite/cylinder_base/seed_801) |
| inventory_tower:cylinder,cube,prism | [outputs/columns_01/random_reserved/random/seed_828](../outputs/columns_01/random_reserved/random/seed_828) |
| inventory_tower:cylinder,cube,prism,prism | [outputs/columns_02/all_geometry_random4/random_l4/seed_904](../outputs/columns_02/all_geometry_random4/random_l4/seed_904) |
| inventory_tower:cylinder,cube,thin_prism | [outputs/columns_02/all_geometry_v16/random_l3/seed_904](../outputs/columns_02/all_geometry_v16/random_l3/seed_904) |
| inventory_tower:cylinder,cylinder | [outputs/columns_02/random2_v11/random_l2_d0/seed_902](../outputs/columns_02/random2_v11/random_l2_d0/seed_902) |
| inventory_tower:cylinder,cylinder,prism,thin_prism | [outputs/columns_02/all_geometry_random4/random_l4/seed_902](../outputs/columns_02/all_geometry_random4/random_l4/seed_902) |
| inventory_tower:cylinder,cylinder,thin_prism | [outputs/columns_02/v22_random/random_l3/seed_902](../outputs/columns_02/v22_random/random_l3/seed_902) |
| inventory_tower:cylinder,prism,cylinder | [outputs/columns_01/random_reserved/random/seed_830](../outputs/columns_01/random_reserved/random/seed_830) |
| inventory_tower:cylinder,prism,prism | [outputs/columns_01/random_dev_v5/random/seed_804](../outputs/columns_01/random_dev_v5/random/seed_804) |
| inventory_tower:cylinder,prism,prism,cylinder | [outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1017](../outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1017) |
| inventory_tower:cylinder,prism,thin_prism | [outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1017](../outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1017) |
| inventory_tower:cylinder,thin_prism | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1010](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1010) |
| inventory_tower:prism,cube | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1013](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1013) |
| inventory_tower:prism,cube,cube | [outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1020](../outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1020) |
| inventory_tower:prism,cube,cube,cylinder | [outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1020](../outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1020) |
| inventory_tower:prism,cube,cylinder | [outputs/columns_01/dev_v4_suite/mixed_triple/seed_801](../outputs/columns_01/dev_v4_suite/mixed_triple/seed_801) |
| inventory_tower:prism,cube,prism,prism | [outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1018](../outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1018) |
| inventory_tower:prism,cube,thin_prism | [outputs/columns_02/heldout_v22/random_tall/random_l3_d2/seed_1018](../outputs/columns_02/heldout_v22/random_tall/random_l3_d2/seed_1018) |
| inventory_tower:prism,cylinder | [outputs/columns_02/random2_v11/random_l2_d0/seed_906](../outputs/columns_02/random2_v11/random_l2_d0/seed_906) |
| inventory_tower:prism,cylinder,cube | [outputs/columns_02/tall_control/dev_geometry_cube_height/random_l3/seed_906](../outputs/columns_02/tall_control/dev_geometry_cube_height/random_l3/seed_906) |
| inventory_tower:prism,cylinder,cube,cube | [outputs/columns_02/heldout_v22/random_tall/random_l4_d2/seed_1019](../outputs/columns_02/heldout_v22/random_tall/random_l4_d2/seed_1019) |
| inventory_tower:prism,cylinder,cube,thin_prism | [outputs/columns_02/all_geometry_random4/random_l4/seed_906](../outputs/columns_02/all_geometry_random4/random_l4/seed_906) |
| inventory_tower:prism,cylinder,cylinder | [outputs/columns_02/all_geometry_v16/random_l3/seed_901](../outputs/columns_02/all_geometry_v16/random_l3/seed_901) |
| inventory_tower:prism,cylinder,cylinder,cylinder | [outputs/columns_02/all_geometry_random4/random_l4/seed_901](../outputs/columns_02/all_geometry_random4/random_l4/seed_901) |
| inventory_tower:prism,cylinder,prism | [outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1016](../outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1016) |
| inventory_tower:prism,cylinder,prism,prism | [outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1016](../outputs/columns_02/heldout_v22/random_tall/random_l4_d0/seed_1016) |
| inventory_tower:prism,cylinder,thin_prism | [outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1019](../outputs/columns_02/heldout_v22/random_tall/random_l3_d0/seed_1019) |
| inventory_tower:prism,prism | [outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1008](../outputs/columns_02/heldout_v22/random_two/random_l2_d0/seed_1008) |
| inventory_tower:prism,prism,cylinder | [outputs/columns_01/random_reserved/random/seed_826](../outputs/columns_01/random_reserved/random/seed_826) |
| legacy_unspecified:default | [outputs/visual_retry_01/baseline/seed_3002](../outputs/visual_retry_01/baseline/seed_3002) |
| prism:default | [outputs/cpu_instances_01/historical_prism](../outputs/cpu_instances_01/historical_prism) |
