# Releasing torchwright

Versioning is SemVer, currently 0.x: breaking changes are allowed and land
on minor bumps (`0.1.0` → `0.2.0`); fixes land on patch bumps.  The version
in `pyproject.toml` is the single source of truth.

A release is a commit on `main` plus a matching tag.  Pushing the tag
triggers `.github/workflows/release.yml`, which rebuilds the package and
publishes it to PyPI via Trusted Publishing (OIDC — no stored tokens; the
`v*` tag must match the pyproject version or the workflow refuses).

## The ritual

1. **Finalize the notes.**  In `RELEASE_NOTES.md`, retitle the
   `# Unreleased` section to the version and date
   (`# 0.2.0 — 2026-08-01`), and start a fresh empty `# Unreleased`
   above it.
2. **Bump the version** in `pyproject.toml`.
3. **Refresh the locks.**  `make modal-lock` (the standalone lock records
   torchwright's own version), and `uv lock` at the umbrella root.
4. **Gate.**  `make lint` and `make test` must pass.
5. **Commit and push** (`Release 0.2.0`), wait for CI to go green on
   `main`.
6. **Tag and push the tag:**

       git tag v0.2.0
       git push origin v0.2.0

7. **Verify:** the Release workflow run succeeds and
   https://pypi.org/project/torchwright/ shows the new version.

Optionally create a GitHub Release from the tag afterwards
(`gh release create v0.2.0 --notes "..."`), pasting the version's
section from `RELEASE_NOTES.md`.

## One-time setup (already done / do once)

- PyPI → account → Publishing → add a (pending) publisher:
  project `torchwright`, owner `physicsrob`, repository `torchwright`,
  workflow `release.yml`, environment `pypi`.
- The `pypi` environment exists in the GitHub repo (Settings →
  Environments); the workflow's `environment: pypi` line binds publishes
  to it.
