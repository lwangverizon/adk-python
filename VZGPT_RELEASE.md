# VZGPT fork baseline

`vzgpt-core` tracks upstream **`main`**, currently commit
`a6ed85905d4c1d1f285ba9d4f76a9d801b66d17f` (2026-10-17). That is release
2.12.0 plus 121 unreleased upstream commits. The fork builds as
**2.12.0+vzgpt.3**. `vzgpt-release.json` records the baseline, the reviewed
source differences and the declared session indexes.

This replaces the earlier policy of tracking official release tags only.
Upstream `main` has no published wheel, so there is no official artifact to
compare against. Compare with a wheel built from the baseline commit instead.

## Fork extensions

The built wheel differs from a wheel built from the baseline commit only in the
files listed under `allowed_source_differences`:

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

3. **NodeTool results keep their 2.9.0 behavior.** Upstream `main` changed
   three things about workflow tools. The fork keeps the 2.9.0 behavior for
   each, because VZGPT workflows end the turn with an internal acknowledgement
   and publish the customer-facing answer themselves:
   - A failed node is returned to the model as
     `Error running node <name>: ...` instead of failing the run (upstream
     b4c5272c5). Invalid input is returned as
     `Error validating input for node: ...`.
   - A NodeTool result is not appended as visible text when it sets
     `skip_summarization`. Upstream 5a0421cbc appends it, which would show
     the acknowledgement JSON to the customer.
   - Events a workflow tool publishes on its `<tool>@<call id>` sub-branch stay
     in the calling agent's history (upstream 2e06fdc81 removes them).
     Otherwise the model loses the workflow's answer on the next turn.

   Fork tests: `tests/unittests/workflow/test_vzgpt_node_tool_compat.py` and
   `tests/unittests/flows/llm_flows/context/test_vzgpt_tool_sub_branch_history.py`.

Upstream schemas and upstream test files are not modified. Upstream tests that
assert the newer NodeTool behavior are listed in
`tests/unittests/_vzgpt_fork_expectations.py` and marked strict xfail by
fork-owned `conftest.py` files, so a sync that loses a fork behavior fails the
suite.

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

1. Merge a reviewed upstream `main` commit and resolve conflicts in the fork
   files.
2. Set `upstream_commit` and `version` in `vzgpt-release.json`.
3. Run `uv run --no-project python scripts/verify_vzgpt_release.py`.
   - It rejects source changes outside the allowlist.
   - It rejects a sync that drops any fork guard (index policy, `merge_state`,
     NodeTool behavior).
   - It rejects any change to the declared session indexes. If upstream changed
     an index, plan the production build before you update
     `declared_session_indexes`.
4. Run `pytest tests/unittests/sessions` and the broader unit suite.

Make the `VZGPT release baseline` workflow a required check through branch
protection.

## Validation (2026-10-17)

- The wheel differs from a wheel built from the baseline commit only in the
  nine allowlisted files.
- Unit tests ran in a fresh Python 3.11 environment with all extras.
  - Full `tests/unittests -n auto`: 20,277 passed, 9 failed, 39 xfailed (19 of
    them are the strict-xfail upstream tests listed above).
  - Six of the 9 failures also fail on upstream `main` and depend on the local
    machine (for example `OTEL_*` variables). The other three are
    load-sensitive and pass in isolation.
  - The fork tests fail without the fork changes.
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
