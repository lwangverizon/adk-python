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

"""VZGPT fork: session State keeps its 2.12.0 semantics.

Upstream ce2fdc5aa and 662b0175a deep-copy every dict or list a read hands out
and record later in-place writes into the delta. VZGPT relies on assignment
being the only way a key enters the delta, and on reads not copying.
"""

from google.adk.sessions.state import State


def test_in_place_write_into_a_read_container_is_not_recorded():
  state = State(value={'tags': ['a'], 'profile': {'tier': 1}}, delta={})

  state['tags'].append('b')
  state['profile']['tier'] = 2

  assert not state.has_delta()
  assert state.to_dict() == {'tags': ['a', 'b'], 'profile': {'tier': 2}}


def test_assignment_is_recorded():
  state = State(value={'tags': ['a']}, delta={})

  state['tags'] = state['tags'] + ['b']

  assert state.has_delta()
  assert state._delta == {'tags': ['a', 'b']}


def test_a_read_neither_copies_nor_snapshots_the_container():
  stored = {'tier': 1}
  state = State(value={'profile': stored}, delta={})

  assert state['profile'] is stored
  assert not hasattr(state, '_snapshots')


def test_record_nested_writes_is_a_no_op():
  state = State(value={'tags': ['a']}, delta={})
  state['tags'].append('b')

  state._record_nested_writes()

  assert state._delta == {}
