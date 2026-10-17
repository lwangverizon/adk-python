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

"""VZGPT fork: see tests/unittests/_vzgpt_fork_expectations.py."""

from tests.unittests._vzgpt_fork_expectations import mark_kept_behavior


def pytest_configure(config):
  config.addinivalue_line(
      "markers", "vzgpt_kept_at_2_9_0: upstream test of behavior the fork keeps"
  )


def pytest_collection_modifyitems(config, items):
  del config  # Unused.
  mark_kept_behavior(items)
