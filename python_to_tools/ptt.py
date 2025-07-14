import json

from pydantic import BaseModel

from python_to_tools.ai.base import AiModelClient
from python_to_tools.behavior import Agent
from python_to_tools.utils import logger


class TaskList(BaseModel):
    list: list[str]
    reasoning: str

    @classmethod
    def from_str(cls, text: str):
        text = text.lstrip("```json")
        text = text.rstrip("```")
        return cls(**json.loads(text))

class TaskFlowBehavior:
    """
    This is a TaskFlow implementation of agentic behavior.

    This flow is based on the concept of implementing a lit of tasks to handle based on an original user query,
    then actioning it.

    TaskFlow is very good at handling sync operations, such as one off user messages where the total time to
    process will be short enough for the user to wait.
    """
    def __init__(
            self,
            root_ai_model: AiModelClient,
            root_agent: Agent = None,
            handler_agent: Agent = None,
            summary_agent: Agent = None
    ):
        """
        Create an instance of AgenticBehavior

        You must provide a  Root AI model which functions as the base for the initial decision-making, including
        the creation of the top level task list.

        Arguments:
            root_ai_model: Instance of any class that implements `AiModelClient`
            handler_agent: Instance of any class that implements `Agent`. The Agent is responsible for ahndling each
                individual task within the generated list, and will contain the bulk of your logic.
            summary_agent: After the tasks have completed, this agent summarizes and finally provides the ultimate
                response to the original question.
        """
        self.root_ai_model = root_ai_model

        self.root_agent = root_agent
        if not self.root_agent:
            self.root_agent = Agent(
                agent_name="root",
                convo_file="root.j2",
                model=root_ai_model
            )

        self.summary_agent = summary_agent
        if not self.summary_agent:
            self.summary_agent = Agent(
                agent_name="summary",
                convo_file="root_summary.j2",
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
        """Converts completed tasks into a textual representation.

        Within this style of Behavior handler, this step is very important as it's how AI ultimately works out
        how it did at handling the user's actual query
        """
        r = []
        for name, result in task_results.items():
            r.append(f"I completed the task: {name}, with the following result: \n{result.rstrip()}")

        return r


    def resolve_from_text(self, msg: str):
        """Resolve the given text into, first, a list of tasks, then close each task one by one using our associated
        handler agents."""
        logger.info("Resolving message into task list")
        task_list = TaskList.from_str(self.root_agent.generate_text(msg).content)
        task_results = {}
        history = []
        for task in task_list.list:
            logger.info(f"resolving task: {task}")
            history = self._task_results_to_text(task_results)
            logger.info(f"Adding {len(history)} history to generation step")
            result = self.handler_agent.resolve_from_text(task, history=history)
            task_results[task] = result

        summary = self.summary_agent.resolve_from_text(msg, history)

        return summary, task_results