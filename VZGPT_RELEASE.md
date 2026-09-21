# VZGPT official-release baseline

`vzgpt-core` targets official **v2.9.0**, not upstream `main`.
See `vzgpt-release.json` for the immutable commit and official PyPI wheel hash.
All upstream Python source files at that tag were compared byte-for-byte with
that verified wheel. The fork builds as **2.9.0+vzgpt.2** to distinguish it from
both the official distribution and the incident artifact `2.9.0+vzgpt.1`.

## Retained extension

The only runtime differences from the release, apart from version identity,
are the existing `merge_state` API implementations in five session services.
They preserve state-only writes without adding conversation events or changing
the session optimistic-concurrency marker. Their regression tests are retained.
The official release already includes several earlier fork fixes; do not replay
historical fork commits blindly.

## Incident-related differences

Post-release upstream commit `a00a2a977` introduced the `_ts_id` events index and
runtime dropping of the older `_ts` index. The incident artifact subsequently
removed dropping and added an advisory lock, but still built missing indexes
inside session initialization. This release baseline includes neither change.
It declares the official `_ts` index and retains any other existing indexes.

This is NOT validation-only initialization: official ADK still creates missing
tables/indexes at runtime. Provision production schemas through reviewed
migrations before rollout. Never treat this change as a universal DDL bypass.
Already-submitted database transactions are not cancelled by a code update.

## Updating the branch

Do not merge `upstream/main`, even if its version string says 2.9.0. Choose an
explicit official release tag, verify its source against the published artifact,
and review the complete schema and API delta before changing the manifest.
Preserve/reapply only reviewed fork extensions. Run:

```bash
uv run --no-project python scripts/verify_vzgpt_release.py
uv run pytest tests/unittests/sessions/
```

The baseline CI check rejects changes outside the reviewed source allowlist and
specific incident regressions. Changes within the allowlist still require code
review and tests; the check is not a substitute for either. Make this CI check
required through repository branch protection to enforce it for merges.
No upstream-main sync workflow was found in this repository; external sync jobs
must also follow this policy.

## Validation (2026-09-21)

- Official tag Python sources match the SHA-verified PyPI 2.9.0 wheel.
- Candidate wheel differs from the official wheel only in the five session
  service extension files and version identity.
- Session, skill, workflow, and function-flow selection: 1,633 passed,
  17 skipped, 8 xfailed; one optional Spanner dependency test deselected.
- Wheel-extracted session and skill selection: 658 passed, 16 skipped,
  3 xfailed; the same Spanner dependency test deselected.
- The initial session/skill run reproduced that Spanner failure: this environment
  lacks `google.cloud.sqlalchemy_spanner`. It was not silently counted as passing.
- All staged pre-commit hooks passed. The unit-guide exception documents that
  deleted post-release modules are being replaced with existing release modules.
- Index preservation/repeated-initialization regression uses real SQLite for
  v0/v1, not PostgreSQL. Production-scale PostgreSQL validation, full-suite tests,
  and consuming VZGPT application compatibility are not claimed.
- Tests used the existing VZGPT Python 3.13 environment with the candidate source
  (or extracted wheel) explicitly on PYTHONPATH. The SDK checkout's older local
  environment could not collect due to missing GenAI `InteractionStatus`.

## Deployment boundary

Updating this branch does not update VZGPT's vendored wheel or running image.
Build a new wheel, verify the installed artifact, update the consuming app's
pin/hash/lock and remove the old initialization patch as a separate reviewed
change. Test application compatibility: post-release skill-lifecycle and other
APIs from upstream main are intentionally absent. Do not deploy automatically.
No production database operation is performed by this change.
