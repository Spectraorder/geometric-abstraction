# Archived pointing experiments

These GPT/Molmo pointing and same-color recognition experiments are outside the
current Visual Base versus Full geometry comparison. Their reports, prompts,
media, scripts, policy and tests are retained here byte-for-byte. The manifest
records original paths and content hashes. Links inside archived reports retain
their original relative layout.

Archived scripts and tests assume their original repository locations. They are
historical source, not supported entry points from this archive. To reproduce
that layout without changing the current branch:

```bash
git worktree add --detach ../aspire-so100-pointing 11dafd4adebbe092f938bbadb6425052523cd495
```

Set up that worktree's environment separately; ignored rollout archives are not
included in Git. Existing local `outputs/` evidence has not been moved or deleted.
The optional pointing runtime modules remain in `src/aspire_so100` because the
shared runner and inventory-policy builder still reference them. Removing those
would change the execution environment used by frozen experiments.
