import json
import logging
import pathlib
import traceback
from typing import Callable, Any

from click.testing import Result
from jinja2 import Template
from pydantic import InstanceOf

from python_to_tools.ai.base import AiModelClient
from python_to_tools.ai.generic_models import Tool, ToolParameter, TextGenerationResponse, ToolCallResponse
from typing import Annotated

from python_to_tools.context import Context
from python_to_tools.utils import ConvoLoader, DEFAULT_TEMPLATE_ENVIRONMENT, JinjaConvoLoader

logger = logging.getLogger(__name__)


class MissingToolError(Exception):
    pass


class UnserializableResponse(Exception):
    pass


class AgentRecursionDepthExceeded(Exception):
    pass

class ToolCallGotInvalidArguments(Exception):
    pass

class Agent:
    """
    An Agent implements a series of tools and a root prompt. it can perform actions, collect data, etc.
    """

    def __init__(
            self,
            agent_name: str,
            convo_loader: ConvoLoader,
            model: AiModelClient,
            description: str = "",
            max_recursion=2,
            tool_call_response_template: Template = None,
            tool_call_error_response_template: Template = None,
    ):
        """
        Create an `Agent` for servicing requests.

        Arguments:
            agent_name: The name of the agent.
            convo_loader: A ConvoLoader object, used for getting the chat message templates we use to convert to
                textual prompts for the model
            model: The AiModelClient to use for resolving requests using the agent
            description: The description of the agent itself, this is important when connecting child agents using
                `add_agent`.
            max_recursion: The maximum depth we will go when calling other agents to resolve a query
            tool_call_response_template: When a tool is called, this is the template that is used to format the
                response data into text for returning to the parent object, agent or behavior.
        """
        self.agent_name = agent_name
        self.convo_loader = convo_loader
        self.model = model
        self.tools = {}
        self.agents = {}
        self.description = description
        self.max_recursion = max_recursion

        self.tool_call_response_template = tool_call_response_template
        if not self.tool_call_response_template:
            self.tool_call_response_template = DEFAULT_TEMPLATE_ENVIRONMENT.get_template(
                "default_tool_call_response.j2"
            )

        self.tool_call_error_response_template = tool_call_error_response_template
        if not self.tool_call_error_response_template:
            self.tool_call_error_response_template = DEFAULT_TEMPLATE_ENVIRONMENT.get_template(
                "default_tool_call_error_response.j2"
            )

    @classmethod
    def from_convo_file(
            cls,
            agent_name: str,
            convo_file_path: pathlib.Path | str,
            model: AiModelClient,
            **cls_kwargs
    ) -> "Agent":
        """Helper method; makes it easier to load agents directly from a convo file path"""
        loader = JinjaConvoLoader(template_path=convo_file_path)
        return cls(agent_name, loader, model, **cls_kwargs)

    def add_tools_from_object(self, obj: Any):
        """Adds all of the class object methods from the given object as tools attached to this agent.
        """
        for method in [m for m in dir(obj) if callable(getattr(obj, m))]:
            self.add_tool(getattr(obj, method))


    def add_tool(self, func: Callable):
        """Adds the given python function as a tool available to this agent.

        Tools will be passed along with any generation task, so the agent can work out if it needs to call a tool
        to service the request.

        When adding tools, you should ensure they return data in some format that `_response_to_str` can convert
        to textual representation.
        """
        if func.__name__ in self.tools:
            logger.warning(f"Naming conflict between tools: {func.__name__} is already registered to this agent.")
        self.tools[func.__name__] = func

    def _response_to_str(self, call_results: dict[str, Any]) -> str:
        """Convert the response from a tool call into a string representation, so that it can be fed back
        into a large language model.

        This supports string responses, json serializable objects or Pydantic models.
        """
        stringified_tool_call_results = {}
        for tool_name, tool_call_result_model in call_results.items():
            if hasattr(tool_call_result_model, "model_dump"):
                result = json.dumps(tool_call_result_model.model_dump())
            elif isinstance(tool_call_result_model, str):
                result = tool_call_result_model
            else:
                try:
                    result = json.dumps(tool_call_result_model, default=str)
                except TypeError:
                    result = str(tool_call_result_model)

            stringified_tool_call_results[tool_name] = result

        return self.tool_call_response_template.render(call_results=stringified_tool_call_results)

    def _get_agent_by_name(self, agent_name: str) -> "Agent":
        """Returns the matching agent attached to this object, or raises if it's not available"""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError("Agent '{}' not found!".format(agent_name))

        return agent

    def _format_error_response(self, exception: str) -> str:
        """Formats any exception raised by a tool call so AI can process and understand it

        Currently, this just means just returning the entire thing as a string.
        """
        return self.tool_call_error_response_template.render(exception=exception)

    def call_tool_from_tool_response(self, tool_call: ToolCallResponse):
        """Executes the given tool, with the given arguments, based on a tool call response object from AI"""

        if tool_call.name not in self.tools:
            raise ValueError(f"tool call {tool_call.name} is not available")

        tool = self.tools.get(tool_call.name)
        arguments = tool_call.arguments
        try:
            return tool(**arguments)
        except Exception:
            # Catch all other errors
            return self._format_error_response(traceback.format_exc())

    def add_agent(self, agent: "Agent"):
        """Add another agent to the agent tree at this node."""
        self.agents[agent.agent_name] = agent

    def next_agent(
            self,
            task: Annotated[str, ToolParameter(type="string", description="The Task we were given")],
            agent_name: Annotated[
                str, ToolParameter(type="string", description="The name of the agent to forward the task to")]
    ):
        """Use this tool if you cannot accomplish the given task.

        This tool is useful when you cannot handle the request yourself. You can call this tool to pass it to
        another agent that may have more tool access.
        """
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
            context: Context = None
    ) -> TextGenerationResponse:
        """Generate text based on a previous text input.

        Will call tools, if tools are given.
        """
        convo = self.convo_loader.to_convo(last_input=text, context=context)
        request = convo.as_text_generation_request()
        #print(json.dumps(request.model_dump(), indent=2))
        tools = []
        if self.agents:
            agent_tool = Tool.from_func(self.next_agent)
            agent_tool.description += """You can select from the following agents:\n"""
            agent_tool.description += "\n".join([f" - {k}: {v.description}" for k, v in self.agents.items()])
            logger.info(agent_tool.description)
            tools.append(agent_tool)

        for name, func in self.tools.items():
            tools.append(Tool.from_func(func))

        if tools:
            request.tools = tools

        return self.model.get_response(
            request=request,
        )

    def resolve_from_text(
            self, text: str,
            context: Context = None,
            depth=0
    ) -> str:
        """Resolve text into either more text, or a series of tool calls.
        """
        result = self.generate_text(text, context=context)
        if result.tool_calls:
            if result.tool_calls[0].name == "next_agent":
                tool_call = result.tool_calls[0]
                next_agent = self._get_agent_by_name(tool_call.arguments.get("agent_name"))
                if depth < self.max_recursion:
                    logger.info(f"Trying to resolve using the next agent at depth {depth} (max: {self.max_recursion})")
                    depth += 1
                    return next_agent.resolve_from_text(text, context=context, depth=depth)
                else:
                    raise AgentRecursionDepthExceeded(f"Max agent recurison depth exceeded: {depth}. ")

            call_results = {}
            for tool_call in result.tool_calls:
                logger.info(f"Resolving text resulted in tool call {tool_call.name}")
                call_results[tool_call.name] = self.call_tool_from_tool_response(tool_call)

            return self._response_to_str(call_results)

        return result.content