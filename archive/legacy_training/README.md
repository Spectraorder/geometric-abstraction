# Historical training and skill-promotion loop

These modules and their dedicated tests belong to the older tower-specific actor
loop. The current matched geometry experiment uses the drivers in `scripts/` and
does not import them. Source bytes are preserved, including the old module names
and internal paths. This archive is not an installed Python package.

Use commit `97d7b98152b21a1612997c9e28873b9eba7d927d` for the original package layout:

```bash
git worktree add --detach ../aspire-so100-before-source-cleanup 97d7b98152b21a1612997c9e28873b9eba7d927d
```

Configure its environment and provide the retained local evidence separately.
Historical experiment manifests hash every top-level package module, including
these unused modules. They are deliberately not rewritten after cleanup. A full
historical source audit therefore requires that original checkout, while any new
experiment in the cleaned checkout must register a new source manifest. Do not
resume a frozen run against a changed source inventory or bypass its audit gate.

Shared runtime modules, generated geometry library contents and retained result
files are not modified by this cleanup. The old loop's skills and documentation
are archived separately under `archive/legacy_skills` and `docs/history`.
