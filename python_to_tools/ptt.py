import json

from pydantic import BaseModel

from python_to_tools.ai.base import AiModelClient
from python_to_tools.behavior import Agent
from python_to_tools.utils import logger


class TaskList(BaseModel):
    list: list[str]
    reasoning: str

class AgenticBehavior:
    """
    The main class for implementing agentic handling of python functions.

    The purpose of this class is to accept any number of functions, creating both an index of tools for AI along
    with the necessary logic structure for invoking them
    """
    def __init__(
            self,
            root_ai_model: AiModelClient,
            root_agent: Agent = None,
            handler_agent: Agent = None
    ):
        """
        Create an instance of AgenticBehavior

        You must provide a  Root AI model which functions as the base for the initial decision-making, including
        the creation of the top level task list.

        Arguments:
            root_ai_model: Instance of any class that implements `AiModelClient`
        """
        self.root_ai_model = root_ai_model

        self.root_agent = root_agent
        if not self.root_agent:
            self.root_agent = Agent(
                agent_name="root",
                convo_file="root.j2",
                model=root_ai_model
            )

        self.handler_agent = handler_agent


    def add_root_agent(self, agent: Agent):
        """Adds the root, top level agent for handling all other requests.

        The root agent should only perform fundamental task analysis and the creation of a task list for the handling
        of other agents.

        You can attach tools to the root agent, and like all agents customize the prompt to suit your needs, but
        deviating too far from this pattern will break the underlying agentic routing.
        """
        self.root_agent = agent

    def add_task_handler_agent(self, agent: Agent):
        """Adds a handler agent to this behavior object. Note that agents all function as a tree!"""
        self.handler_agent = agent

    def _task_results_to_text(self, task_results: dict[str, str]):
        """Converts the
        """
        r = []
        for name, result in task_results.items():
            r.append(f"\n<task_output task_name='{name}'>\n{result}</task_output>")

        return r

    def resolve_from_text(self, msg: str):
        """Resolve the given text into, first, a list of tasks, then close each task one by one using our associated
        handler agents."""
        logger.info("Resolving message into task list")
        task_list = TaskList(**json.loads(self.root_agent.generate_text(msg).content))
        task_results = {}
        for task in task_list.list:
            logger.info(f"resolving task: {task}")
            history = self._task_results_to_text(task_results)
            logger.info(f"Adding {len(history)} history to generation step")
            result = self.handler_agent.resolve_from_text(task, history=history)
            task_results[task] = result
        return task_results