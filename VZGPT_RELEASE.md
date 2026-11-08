# VZGPT fork baseline

`vzgpt-core` targets the official **v2.14.0** release (tag commit
`93814c3475916bf873236f4e30f6915b4b9c8698`), not upstream `main`. The fork
builds as **2.14.0+vzgpt.1**. `vzgpt-release.json` records the baseline, the
official PyPI wheel hash, the reviewed source differences and the declared
session indexes.

The previous baseline was v2.13.0 (2.13.0+vzgpt.1). The five fork extensions
below are reapplied on v2.14.0 unchanged in behavior. Upstream `ce2fdc5aa` and
`662b0175a` are still part of v2.14.0, so the fork keeps the 2.12.0
session-state semantics (extension 4).

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

4. **Session State keeps its 2.12.0 semantics.** Upstream `ce2fdc5aa` and
   `662b0175a` (both in v2.13.0) deep-copy every dict or list a read hands
   out and record later in-place writes into the delta. VZGPT has about 28
   functions that write into containers read from state in place and relies on
   those writes not being persisted. The copy also costs about 3 ms per
   callback for a 261 KB value. `sessions/state.py` is the v2.12.0 file plus a
   no-op `_record_nested_writes()` that `agents/context.py` calls. Only an
   assignment enters the delta.

   Fork test: `tests/unittests/sessions/test_vzgpt_state_semantics.py`.

Upstream schemas and upstream test files are not modified. Upstream tests that
assert the newer NodeTool or nested-state behavior are listed in
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

1. Choose an explicit official release tag. Verify its sources against the
   published PyPI wheel. Do not merge upstream `main`.
2. Reapply the fork extensions on the tag, without replaying old fork commits
   blindly. Set `upstream_tag`, `upstream_commit`, `official_wheel_sha256` and
   `version` in `vzgpt-release.json`.
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

## Validation (2026-11-08)

- A wheel built from the tag matches every Python file of the official PyPI
  2.14.0 wheel (sha256 `f7d8c79b…`, 830 Python files). The tag build also ships
  `README.md` files that the PyPI wheel omits. The fork wheel differs from the
  PyPI wheel in exactly the ten allowlisted Python files.
- `scripts/verify_vzgpt_release.py` passes. The declared session indexes are
  unchanged, and v2.14.0 adds no session migration.
- Reapplying the extensions on v2.14.0 needed three manual merges: the
  `merge_state` and stale-writer tests, the index-policy helpers next to the new
  `_begin_sqlite_write_transaction`, and the tool sub-branch guard in front of
  the new `_BranchPath.has_tool_segment_below`.
- One new upstream test asserts that tool sub-branch events are excluded from
  the caller's history: three cases of
  `test_tool_sub_branch_event_needs_tool_segment_below_branch` are strict xfail.
- Full `tests/unittests -n 10` in a fresh Python 3.11 environment: 23,089
  passed, 58 xfailed, 25 failed. All 25 (telemetry setup, functional schema and
  instrumentation tests and one BigQuery plugin drain test) pass when rerun with
  the shell's `OTEL_*` variables unset: 1,082 passed, 3 xfailed.

## Deployment boundary

Updating this branch does not change VZGPT's vendored wheel or running images.
Adopting the wheel is a separate, reviewed VZGPT change. No database operation
is performed by this branch. The VZGPT change has to handle these items:

- The pin, hash and lock for the new wheel. v2.14.0 only widens the
  `opentelemetry-api`/`-sdk` ceiling to `<=1.45.2`; no other core dependency
  changes.
- **PostgreSQL JSON columns (upstream `d487c5377`).** `DynamicJSON` now creates
  `json` instead of `jsonb` columns. Existing `jsonb` tables are not altered
  and still read and write. Only a table created by this wheel on an empty
  database is `json`. Queries outside ADK that use jsonb-only operators or GIN
  indexes on ADK tables would not work on such a table.
- Bytes in session state are now stored as base64 (upstream `d91dd34e8`,
  `73dca84ef`) instead of failing to serialize.
- `google-adk[eval]` still needs `google-cloud-aiplatform>=2.2,<3`, which
  conflicts with VZGPT's `<2.0.0` override for pyvegas.
- The default telemetry schema is 2. VZGPT's Galileo tracing depends on the
  legacy attributes and keeps
  `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` (supported until
  2027-03-31). v2.14.0 records the aggregated answer of a streamed model call
  on its span instead of the chunks (upstream `80a44111c`).
- Before rollout, confirm in each environment that VZGPT's own indexes on ADK
  tables (for example `idx_sessions_app_id_update_time`) already exist. This
  wheel does not build them on existing tables.
