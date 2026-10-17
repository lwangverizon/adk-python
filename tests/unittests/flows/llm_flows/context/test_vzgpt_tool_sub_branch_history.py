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

"""VZGPT fork: an answer a workflow tool published stays in caller history."""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.context._contents import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from .... import testing_utils


@pytest.mark.asyncio
async def test_root_agent_keeps_tool_sub_branch_events():
  agent = Agent(model='gemini-2.5-flash', name='root_agent')
  llm_request = LlmRequest(model='gemini-2.5-flash')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.branch = None
  invocation_context.session.events = [
      Event(
          invocation_id='inv',
          author='user',
          content=types.UserContent('upgrade my phone'),
      ),
      Event(
          invocation_id='inv',
          author='root_agent',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='adk-1', name='upgrade_workflow', args={}
                      )
                  )
              ],
          ),
      ),
      Event(
          invocation_id='inv',
          author='upgrade_workflow',
          content=types.ModelContent('Verified answer'),
          branch='upgrade_workflow@adk-1',
      ),
      Event(
          invocation_id='inv',
          author='root_agent',
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='adk-1',
                          name='upgrade_workflow',
                          response={'status': 'answer'},
                      )
                  )
              ],
          ),
      ),
      Event(
          invocation_id='inv2',
          author='user',
          content=types.UserContent('and then?'),
      ),
  ]

  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  assert 'Verified answer' in str(llm_request.contents)


class _Request(BaseModel):
  q: str = ''


@pytest.mark.asyncio
async def test_workflow_tool_answer_reaches_the_next_turn():
  @node
  async def answer(node_input: _Request):
    yield Event(message='Verified answer')
    yield {'status': 'answer', 'agent_instruction': 'add nothing'}

  workflow = Workflow(
      name='answer_workflow',
      description='Answers.',
      input_schema=_Request,
      edges=[(START, answer)],
  )
  model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name='answer_workflow', args={}),
          types.Part.from_text(text='Done.'),
          types.Part.from_text(text='Still done.'),
      ]
  )
  parent = LlmAgent(name='parent', model=model, tools=[workflow])
  runner = testing_utils.InMemoryRunner(root_agent=parent)

  await runner.run_async(testing_utils.get_user_content('go'))
  await runner.run_async(testing_utils.get_user_content('and now?'))

  assert len(model.requests) == 3
  assert 'Verified answer' in str(model.requests[2].contents)
