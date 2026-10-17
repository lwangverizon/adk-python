# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VZGPT fork: no runtime index builds on existing tables of server databases.

SQLite stands in for a server dialect by emptying the allowed dialect set.
"""

import logging

from google.adk.sessions import database_session_service
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.schemas import v0
import pytest
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_LEGACY_INDEX = 'idx_events_app_user_session_ts'
_DECLARED_INDEX = 'idx_events_app_user_session_ts_id'


async def _events_index_names(db_url):
  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    indexes = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_indexes('events')
    )
  await engine.dispose()
  return {index['name'] for index in indexes}


async def _create_legacy_indexed_db(db_url, schema_version):
  """Creates a database whose events table has only the pre-2.11 index."""
  if schema_version == 'v0':
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
      await conn.run_sync(v0.Base.metadata.create_all)
    await engine.dispose()
  async with DatabaseSessionService(db_url) as service:
    await service.create_session(
        app_name='app', user_id='user', session_id='retained'
    )
  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.execute(text(f'DROP INDEX {_DECLARED_INDEX}'))
    await conn.execute(
        text(
            f'CREATE INDEX {_LEGACY_INDEX} ON events'
            ' (app_name, user_id, session_id, timestamp DESC)'
        )
    )
  await engine.dispose()


@pytest.mark.parametrize('schema_version', ['v0', 'v1'])
async def test_server_dialect_reports_but_does_not_build_missing_index(
    tmp_path, monkeypatch, caplog, schema_version
):
  db_url = f'sqlite+aiosqlite:///{tmp_path / "legacy.db"}'
  await _create_legacy_indexed_db(db_url, schema_version)
  monkeypatch.setattr(
      database_session_service, '_RUNTIME_INDEX_DDL_DIALECTS', frozenset()
  )

  statements = []
  for _ in range(2):
    async with DatabaseSessionService(db_url) as service:

      def record_sql(conn, cursor, statement, parameters, context, many):
        statements.append(statement.upper())

      event.listen(
          service.db_engine.sync_engine, 'before_cursor_execute', record_sql
      )
      with caplog.at_level(logging.WARNING):
        session = await service.get_session(
            app_name='app', user_id='user', session_id='retained'
        )
      assert session is not None

  assert not any(
      'CREATE INDEX' in sql or 'DROP INDEX' in sql for sql in statements
  )
  assert await _events_index_names(db_url) >= {_LEGACY_INDEX}
  assert _DECLARED_INDEX not in await _events_index_names(db_url)
  assert any(
      _DECLARED_INDEX in record.getMessage()
      and 'CREATE INDEX' in record.getMessage()
      for record in caplog.records
  )


async def test_server_dialect_still_indexes_newly_created_tables(
    tmp_path, monkeypatch
):
  db_url = f'sqlite+aiosqlite:///{tmp_path / "fresh.db"}'
  monkeypatch.setattr(
      database_session_service, '_RUNTIME_INDEX_DDL_DIALECTS', frozenset()
  )

  async with DatabaseSessionService(db_url) as service:
    await service.create_session(app_name='app', user_id='user')

  assert _DECLARED_INDEX in await _events_index_names(db_url)


def test_sqlite_is_the_only_runtime_index_ddl_dialect():
  assert database_session_service._RUNTIME_INDEX_DDL_DIALECTS == {'sqlite'}
