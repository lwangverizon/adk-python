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

"""Reject unreviewed source drift from VZGPT's official ADK release baseline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

_ROOT = Path(__file__).resolve().parents[1]


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
    raise SystemExit(f'Unreviewed release drift: {sorted(unexpected)}')
  service = (
      _ROOT / 'src/google/adk/sessions/database_session_service.py'
  ).read_text(encoding='utf-8')
  for forbidden in ('pg_advisory_xact_lock', '_SUPERSEDED_INDEX_NAMES'):
    if forbidden in service:
      raise SystemExit(f'Unsafe session initialization returned: {forbidden}')
  for version in ('v0', 'v1'):
    schema = (
        _ROOT / f'src/google/adk/sessions/schemas/{version}.py'
    ).read_text(encoding='utf-8')
    if 'idx_events_app_user_session_ts_id' in schema:
      raise SystemExit(f'Post-release events index returned in {version}')
  print(f'Verified release baseline {baseline}; reviewed paths: {changed}')


if __name__ == '__main__':
  main()
