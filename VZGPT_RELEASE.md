# VZGPT fork baseline

`vzgpt-core` targets the official **v2.12.0** release (tag commit
`38381f65f2bee0f70a89f938ee64e51b34916fa7`), not upstream `main`. The fork
builds as **2.12.0+vzgpt.4**. `vzgpt-release.json` records the baseline, the
official PyPI wheel hash, the reviewed source differences and the declared
session indexes.

All 795 Python files in the official PyPI 2.12.0 wheel are byte-identical to a
wheel built from the tag. 2.12.0+vzgpt.3 tracked upstream `main` (121
unreleased commits beyond 2.12.0). It was replaced because one unreleased
change, upstream `ce2fdc5aa`, changes session-state behavior: in-place writes
into state containers are now persisted, and containers are copied on read.

## Fork extensions

The built wheel differs from the official wheel only in the files listed under
`allowed_source_differences`:

1. **`merge_state`.** A state-only write API for five session services. It writes
   no events row and does not advance the session's optimistic-concurrency
   marker. Its regression tests live in `tests/unittests/sessions/`.
2. **No runtime index builds on existing server-database tables.** Upstream
   `prepare_tables()` creates every missing declared index with a plain
   `CREATE INDEX`. On PostgreSQL that blocks writes to the table for the whole
   build. Outside SQLite the fork only logs a missing index on an existing table
   (with its `CREATE INDEX` statement) and leaves it for an operator. New tables
   still get their indexes, and SQLite keeps the upstream behavior. Tests:
   `tests/unittests/sessions/migration/test_vzgpt_runtime_index_policy.py`.

Upstream schemas and upstream tests are not modified.

## Events index state

Upstream declares `idx_events_app_user_session_ts_id`
`(app_name, user_id, session_id, timestamp DESC, id DESC)`. Databases created by
2.9.0+vzgpt.2 only have `idx_events_app_user_session_ts`. The session queries
are the same in both versions and keep using the existing index. Until the new
index exists, each process logs one warning at startup.

If you adopt the new index, build it outside a deploy, UAT first:

```sql
SET statement_timeout = 0;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_app_user_session_ts_id
  ON events (app_name, user_id, session_id, timestamp DESC, id DESC);
SELECT indisvalid FROM pg_index
WHERE indexrelid = 'idx_events_app_user_session_ts_id'::regclass;
```

- A failed concurrent build leaves an invalid index with the same name. Index
  checks match by name only, so check `indisvalid`. If it is false, run
  `DROP INDEX CONCURRENTLY` and build it again.
- Keep `idx_events_app_user_session_ts` while 2.9.0+vzgpt.2 is a rollback
  target, because that wheel builds that index if it is missing.
- VZGPT application indexes registered on ADK tables are covered by the same
  policy. If one of them is missing on an existing table, it is logged rather
  than built.

## Updating the branch

1. Choose an explicit official release tag. Verify its sources against the
   published PyPI wheel. Do not merge upstream `main`.
2. Reapply the fork extensions on the tag, without replaying old fork commits
   blindly. Set `upstream_tag`, `upstream_commit`, `official_wheel_sha256` and
   `version` in `vzgpt-release.json`.
3. Run `uv run --no-project python scripts/verify_vzgpt_release.py`.
   - It rejects source changes outside the allowlist.
   - It rejects a sync that drops the index guard or `merge_state`.
   - It rejects any change to the declared session indexes. If upstream changed
     an index, plan the production build before you update
     `declared_session_indexes`.
4. Run `pytest tests/unittests/sessions` and the broader unit suite.

Make the `VZGPT release baseline` workflow a required check through branch
protection.

## Validation (2026-10-17)

- The wheel differs from a wheel built from the baseline commit only in the six
  allowlisted files.
- Unit tests ran in a fresh Python 3.11 environment with all extras.
  - `tests/unittests/sessions`: 645 passed, 16 skipped.
  - Full `tests/unittests -n auto`: 20,243 passed, 58 failed, 108 skipped.
    Rerun in isolation, 51 of the 58 passed. The same 7 failed on this branch
    and on the upstream baseline: six depend on the local machine (for example
    `OTEL_*` variables) and one timing-sensitive web UI test fails on a
    different parameter each run. No session test failed.
- Session tests against the extracted wheel in the VZGPT Python 3.13
  environment (SQLAlchemy 2.0.50): 644 passed, 16 skipped. One test was
  deselected because that environment lacks
  `google.cloud.sqlalchemy_spanner`.
- Local PostgreSQL 17 (temporary cluster, Unix socket only), with a schema
  created by 2.9.0+vzgpt.2 and 6M events (1.6 GB):
  - Starting this wheel took 0.05 s. It issued no index DDL, logged one
    warning, and concurrent inserts stayed under 10 ms.
  - Upstream `main` without the fork guard built the index while blocking
    inserts. On the same data that took 27 s.
  - `merge_state` worked on PostgreSQL, and an `append_event` from a session
    loaded before the merge still succeeded afterwards.
  - Rolling back to 2.9.0+vzgpt.2 on the same database issued no DDL.
  - On an empty database the wheel created all tables with the upstream index.
  - After an out-of-band `CREATE INDEX CONCURRENTLY`, startup issued no DDL and
    logged no warning.

## Deployment boundary

Updating this branch does not change VZGPT's vendored wheel or running images.
Adopting the wheel is a separate, reviewed VZGPT change. No database operation
is performed by this branch. The VZGPT change has to handle these items:

- The pin, hash and lock for the new wheel.
- `google-genai>=2.30`.
- `google-adk[eval]` now needs `google-cloud-aiplatform>=2.2,<3`, which
  conflicts with VZGPT's `<2.0.0` override for pyvegas.
- Upstream `main` changes the default telemetry schema to 2. The `call_llm`
  span is gone and `gcp.vertex.agent.*` attributes become `gen_ai.*`. VZGPT's
  Galileo tracing depends on the legacy attributes. Set
  `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` (supported until
  2027-03-31) or migrate the tracing first.
- Before rollout, confirm in each environment that VZGPT's own indexes on ADK
  tables (for example `idx_sessions_app_id_update_time`) already exist. This
  wheel no longer builds them on existing tables.
- Re-test the skill lifecycle APIs that 2.9.0+vzgpt.2 deliberately omitted.
