import json
import logging
from typing import Callable

from python_to_tools.ai.base import AiModelClient
from python_to_tools.ai.generic_models import Tool, ToolParameter, TextGenerationResponse, ToolCallResponse
from typing import Annotated
from python_to_tools.utils import prompt_template_to_convo

logger = logging.getLogger(__name__)

class MissingTOolError(Exception):
    pass

class Agent:
    """
    An Agent implements a series of tools and a root prompt. it can perform actions, collect data, etc.
    """

    def __init__(
            self,
            agent_name: str,
            convo_file: str,
            model: AiModelClient
    ):
        self.agent_name = agent_name
        self.convo_file = convo_file
        self.model = model
        self.tools = {}
        self.agents = {}

    def add_tool(self, func: Callable):
        """Adds the given python function as a tool available to this agent.

        Tools will be passed along with any generation task, so the agent can work out if it needs to call a tool
        to service the request.
        """
        self.tools[func.__name__] = func

    def call_tool_from_tool_response(self, tool_call: ToolCallResponse):
        """Executes the given tool, with the given arguments, based on a tool call response object from AI"""
        if tool_call.name not in self.tools:
            raise ValueError(f"tool call {tool_call.name} is not available")

        tool = self.tools.get(tool_call.name)
        arguments = tool_call.arguments
        return tool(**arguments)

    def add_agent(self, agent: "Agent"):
        """Add another agent to the agent tree at this node."""
        self.agents[agent.agent_name] = agent

    def next_agent(
            self,
            task: Annotated[str, ToolParameter(type="string", description="The Task we were given")],
            agent_name: Annotated[
                str, ToolParameter(type="string", description="The name of the agent to forward the task to")]
    ):
        """Call the next agent to handle this message or task."""
        pass

    def _tool_call_results_to_text(self, call_results: dict[str, str]):
        """Converts the results of tool calls into plain text, suitable for passing to ai models

        In the future we will support templates for this but for now it's static"""
        r = ""
        for name, result in call_results.items():
            r += f"I called the tool {name} and it returned: {result}\n"

        return r

    def generate_text(
            self,
            text: str,
            history: list[str] = None
    ) -> TextGenerationResponse:
        """Generate text based on a previous text input.

        Will call tools, if tools are given.
        """
        convo = prompt_template_to_convo(self.convo_file, last_input=text, history=history)
        request = convo.as_text_generation_request()
        tools = []
        for name, func in self.tools.items():
            tools.append(Tool.from_func(func))

        request.tools = tools

        print(request.model_dump_json(indent=4))

        return self.model.get_response(
            request=request,
        )

    def resolve_from_text(self, text: str, history: list[str] = None) -> str:
        """
        Resolve text into either more text, or a series of tool calls.
        """
        result = self.generate_text(text, history=history)
        if result.tool_calls:
            call_results = {}
            for tool_call in result.tool_calls:
                logger.info(f"Resolving task using tool call {tool_call.name}")
                call_results[tool_call.name] = self.call_tool_from_tool_response(tool_call)

            return self._tool_call_results_to_text(call_results)

        return result.content


