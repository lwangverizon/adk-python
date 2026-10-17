# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject unreviewed drift from VZGPT's upstream baseline and index policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess

_ROOT = Path(__file__).resolve().parents[1]
_SERVICE = 'src/google/adk/sessions/database_session_service.py'
_SCHEMAS = 'src/google/adk/sessions/schemas'
# Fork guard markers that a sync must not drop.
_REQUIRED_IN_SERVICE = (
    '_RUNTIME_INDEX_DDL_DIALECTS = frozenset({_SQLITE_DIALECT})',
    'if connection.dialect.name not in _RUNTIME_INDEX_DDL_DIALECTS:',
    'async def merge_state(',
)
# Destructive or serializing startup DDL from earlier incidents.
_FORBIDDEN_IN_SERVICE = ('_SUPERSEDED_INDEX_NAMES', 'pg_advisory_xact_lock')


def _index_column(node: ast.expr) -> str:
  if isinstance(node, ast.Constant):
    return str(node.value)
  if (
      isinstance(node, ast.Call)
      and getattr(node.func, 'id', None) == 'desc'
      and isinstance(node.args[0], ast.Constant)
  ):
    return f'{node.args[0].value} DESC'
  return ast.unparse(node)


def _declared_indexes(schema: Path) -> dict[str, dict[str, list[str]]]:
  """Returns {table: {index: columns}} for Index() calls in ORM classes."""
  declared: dict[str, dict[str, list[str]]] = {}
  for cls in ast.walk(ast.parse(schema.read_text(encoding='utf-8'))):
    if not isinstance(cls, ast.ClassDef):
      continue
    table = next(
        (
            stmt.value.value
            for stmt in cls.body
            if isinstance(stmt, ast.Assign)
            and any(
                getattr(target, 'id', None) == '__tablename__'
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Constant)
        ),
        None,
    )
    for node in ast.walk(cls):
      if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == (
          'Index'
      ):
        if table is None:
          raise SystemExit(f'Index outside a mapped table in {schema}')
        name, *columns = node.args
        declared.setdefault(table, {})[_index_column(name)] = [
            _index_column(column) for column in columns
        ]
  return declared


def main() -> None:
  manifest = json.loads(
      (_ROOT / 'vzgpt-release.json').read_text(encoding='utf-8')
  )
  baseline = manifest['upstream_commit']
  changed = subprocess.check_output(
      ['git', 'diff', '--name-only', baseline, '--', 'src', 'pyproject.toml'],
      cwd=_ROOT,
      text=True,
  ).splitlines()
  unexpected = set(changed) - set(manifest['allowed_source_differences'])
  if unexpected:
    raise SystemExit(f'Unreviewed source drift: {sorted(unexpected)}')

  service = (_ROOT / _SERVICE).read_text(encoding='utf-8')
  for required in _REQUIRED_IN_SERVICE:
    if required not in service:
      raise SystemExit(f'VZGPT session guard missing: {required}')
  for forbidden in _FORBIDDEN_IN_SERVICE:
    if forbidden in service:
      raise SystemExit(f'Unsafe session initialization returned: {forbidden}')

  for version, expected in manifest['declared_session_indexes'].items():
    actual = _declared_indexes(_ROOT / _SCHEMAS / f'{version}.py')
    if actual != expected:
      raise SystemExit(
          f'Declared {version} session indexes changed: {actual}. Plan the'
          ' production index build, then update vzgpt-release.json.'
      )
  print(f'Verified baseline {baseline}; reviewed paths: {changed}')


if __name__ == '__main__':
  main()
