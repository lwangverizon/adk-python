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

"""VZGPT fork: upstream tests that assert behavior this fork keeps at 2.9.0.

Each listed upstream test asserts newer upstream behavior that the fork
deliberately does not take. Marking them strict xfail keeps upstream test files
unmodified, and the suite fails if a sync loses the fork behavior. The fork's
own tests assert the kept behavior:
- NodeTool results: tests/unittests/workflow/test_vzgpt_node_tool_compat.py
- Tool sub-branch history:
  tests/unittests/flows/llm_flows/context/test_vzgpt_tool_sub_branch_history.py
- Session State (2.12.0 semantics, no nested-write recording):
  tests/unittests/sessions/test_vzgpt_state_semantics.py
"""

import pytest

_NODE_TOOL = 'VZGPT fork keeps 2.9.0 NodeTool result behavior'
_SUB_BRANCH = 'VZGPT fork keeps tool sub-branch events in caller history'
_ENTRY_ARGS = 'test_node_tool_workflow_entry_args_are_validated_before_run'
_STATE = 'VZGPT fork keeps 2.12.0 State: in-place nested writes are not recorded'

# {test file name: {test name or parametrized node name: reason}}
KEPT_AT_2_9_0 = {
    'test_node_tool.py': {
        'test_function_node_tool_skip_summarization_returns_output_directly': (
            _NODE_TOOL
        ),
        'test_node_tool_failure_reaches_on_tool_error_callback': _NODE_TOOL,
        'test_node_tool_function_node_missing_node_input_param_returns_error_dict': (
            _NODE_TOOL
        ),
        'test_node_tool_function_node_none_arg_for_non_optional_returns_error_dict': (
            _NODE_TOOL
        ),
        'test_node_tool_function_node_validation_error_returns_error_dict': (
            _NODE_TOOL
        ),
        'test_node_tool_skip_summarization_returns_workflow_output_directly': (
            _NODE_TOOL
        ),
        'test_node_tool_unhandled_failure_propagates': _NODE_TOOL,
        'test_node_tool_validation_error_returns_error_dict': _NODE_TOOL,
        'test_node_tool_workflow_downstream_error_hits_on_tool_error': (
            _NODE_TOOL
        ),
        **{
            f'{_ENTRY_ARGS}[{case}]': _NODE_TOOL
            for case in (
                'dict_schema_state_node_input',
                'join_from_start_is_skipped',
                'node_input_entry',
                'parallel_start_distinct_coercions',
                'parallel_start_validates_all_branches',
                'parallel_worker_items',
                'primitive_request',
            )
        },
        'test_root_agent_excludes_fan_out_messages_from_node_tool_workflow': (
            _SUB_BRANCH
        ),
        'test_single_turn_workflow_agent_keeps_tool_response_after_tool_messages': (
            _SUB_BRANCH
        ),
    },
    'test_function_tool.py': {
        'test_generator_tool_intermediate_events_stay_out_of_next_turn_context': (
            _SUB_BRANCH
        ),
    },
    'test_contents_branch.py': {
        'test_root_agent_excludes_tool_sub_branch_events': _SUB_BRANCH,
    },
    # Upstream ce2fdc5aa and 662b0175a (v2.13.0).
    'test_state.py': {
        name: _STATE
        for name in (
            'test_a_nested_write_is_recorded_only_by_the_state_that_read_it',
            'test_a_read_hands_back_the_stored_container',
            'test_a_recorded_container_that_cannot_be_compared_is_withdrawn',
            'test_a_recorded_container_that_loses_its_json_form_is_withdrawn',
            'test_a_recorded_key_assigned_again_is_not_withdrawn',
            'test_a_value_another_writer_put_in_the_delta_is_not_withdrawn',
            'test_a_write_after_the_delta_is_cleared_is_still_recorded',
            'test_deleting_from_a_nested_dict_records_the_key',
            'test_writing_far_down_a_nested_dict_records_the_key',
            'test_writing_into_a_nested_dict_records_the_key',
            'test_writing_into_a_nested_dict_records_the_key_under_a_schema',
            'test_writing_into_a_nested_list_records_the_key',
        )
    },
    'test_context.py': {
        'test_actions_include_a_write_into_a_nested_state_dict': _STATE,
    },
    'test_base_agent.py': {
        'test_run_async_before_agent_callback_content_carries_nested_write': (
            _STATE
        ),
    },
    'test_model_callbacks.py': {
        'test_a_model_callback_write_into_a_nested_state_dict_is_persisted': (
            _STATE
        ),
        'test_an_on_model_error_callback_nested_state_write_is_persisted': _STATE,
    },
}


def mark_kept_behavior(items: list[pytest.Item]) -> None:
  """Marks the listed upstream tests strict xfail."""
  for item in items:
    expected = KEPT_AT_2_9_0.get(item.path.name, {})
    reason = expected.get(item.name) or expected.get(
        getattr(item, 'originalname', item.name)
    )
    if reason and not item.get_closest_marker('vzgpt_kept_at_2_9_0'):
      item.add_marker(pytest.mark.vzgpt_kept_at_2_9_0)
      item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
