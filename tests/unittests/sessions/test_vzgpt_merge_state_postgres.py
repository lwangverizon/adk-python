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

"""VZGPT fork: merge_state works on json and jsonb PostgreSQL state columns."""

from google.adk.sessions import database_session_service as service
from google.adk.sessions.schemas.shared import DynamicJSON
from sqlalchemy.dialects import postgresql


def test_postgres_merge_casts_the_column_before_concatenating():
  # v2.14.0 creates json columns (upstream d487c5377); json has no ||
  # operator, so the stored value must be cast before the merge.
  clause = service._json_merge_set_clause(service._POSTGRESQL_DIALECT)
  assert clause == 'state = CAST(state AS JSONB) || CAST(:delta AS JSONB)'


def test_new_postgres_tables_use_json():
  impl = DynamicJSON().load_dialect_impl(postgresql.dialect())
  assert isinstance(impl, postgresql.JSON)
  assert not isinstance(impl, postgresql.JSONB)
