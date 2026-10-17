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

"""VZGPT fork: NodeTool results keep their 2.9.0 behavior."""

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils


class _Request(BaseModel):
  q: str


async def _run(step, args):
  workflow = Workflow(
      name='wf',
      description='d',
      input_schema=_Request,
      edges=[(START, step)],
  )
  errors = []

  async def on_tool_error(tool, args, tool_context, error):
    errors.append(error)
    return None

  model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name='wf', args=args),
          types.Part.from_text(text='done'),
      ]
  )
  agent = LlmAgent(
      name='parent',
      model=model,
      tools=[workflow],
      on_tool_error_callback=on_tool_error,
  )
  runner = testing_utils.InMemoryRunner(root_agent=agent)
  events = await runner.run_async(testing_utils.get_user_content('go'))
  return events, errors, model


def _responses(events):
  return [
      part.function_response.response
      for event in events
      for part in (event.content.parts if event.content else []) or []
      if part.function_response
  ]


@pytest.mark.asyncio
async def test_failed_node_is_reported_to_the_model():
  def fail(node_input: _Request) -> dict:
    raise ValueError('boom')

  events, errors, model = await _run(fail, {'q': 'x'})

  assert _responses(events) == [
      {'result': 'Error running node wf: Dynamic node wf failed'}
  ]
  assert not errors
  assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_invalid_node_input_is_reported_as_a_result_string():
  def ok(node_input: _Request) -> dict:
    return {'answer': node_input.q}

  events, _, _ = await _run(ok, {'wrong': 1})

  (response,) = _responses(events)
  assert set(response) == {'result'}
  assert response['result'].startswith('Error validating input for node: ')


@pytest.mark.asyncio
async def test_skip_summarization_does_not_append_the_result_as_text():
  def answer(node_input: _Request, ctx: Context) -> dict:
    ctx.actions.skip_summarization = True
    return {'status': 'answer', 'agent_instruction': 'add nothing'}

  events, _, model = await _run(answer, {'q': 'x'})

  last = events[-1]
  assert last.actions.skip_summarization is True
  assert [part.text for part in last.content.parts if part.text] == []
  assert len(model.requests) == 1
