# Policy sources and generated candidates

`controllers/` holds the canonical task controllers; shared perception and geometry
remain in `src/aspire_so100`. Generated standalone policies are build artifacts,
not a second maintained copy of those sources.

```bash
uv run python scripts/build_inventory_policy.py \
  --controller structure_policy --output outputs/generated/structure.py
```

Available controllers are `inventory_policy`, `column_policy`, `structure_policy`
and `cube_alignment_policy`. The builder also supports explicit pointing/refinement
options. Column/structure evaluators generate their default candidate inside the
requested output directory. Explicit `--candidate` paths remain supported.
A build never overwrites a file containing different source.

The Base/Full learning comparison uses `support_graph_visual_starter.py` and the
shared RGB-D PnP/session sources here. Old evaluated versions remain in Git history
and local experiment outputs; they are not maintained as current policy copies.
