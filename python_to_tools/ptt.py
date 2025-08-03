import json

from pydantic import BaseModel

from python_to_tools.ai.base import AiModelClient
from python_to_tools.ai.generic_models import MessageRoleEnum
from python_to_tools.behavior import Agent
from python_to_tools.context import Context, MemoryContext
from python_to_tools.utils import logger, JinjaConvoLoader


class TaskList(BaseModel):
    list: list[str]
    reasoning: str

    @classmethod
    def from_str(cls, text: str):
        text = text.lstrip("```json")
        text = text.rstrip("```")
        return cls(**json.loads(text))


class TaskFlowResult(BaseModel):
    summary: str
    success: bool
    task_output: dict[str, str]


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
            summary_agent: Agent = None,
            review_agent: Agent = None,
            context: Context = None
    ):
        """This behavior first creates a "task list" using the root_agent.

        The **root_agent** MUST return a JSON str in the format;

        ```json
        {
            tasks: [],
            reasoning: ""
        }
        ```

        Each task within the task list is then passed, in sequence, to teh **handler_agent**. The **handler_agent**
        will solve the task, returning the result, or pass it to it's own sub agents.

        After each response, the **review_agent** will review the resutl and determine if it was a success. If so,
        it will continue, otherwise it will stop processing further tasks.

        Regardless of success or failure, all of the tasks and their results will ultimately be summarized for the
        user along with whether the original request was solved.

        Arguments:
            root_ai_model: Instance of any class that implements `AiModelClient`,
            root_agent: Instance of any class that implements `Agent`. This agent is responsible for creating a task
                list.
            handler_agent: Instance of any class that implements `Agent`. The Agent is responsible for handling each
                individual task within the generated list, and will contain the bulk of your logic.
            summary_agent: After the tasks have completed, this agent summarizes and finally provides the ultimate
                response to the original question.
            review_agent: After each task, this agent will review the result to determine if the process has been
                successful thus far - if not, we can exit early
            context: Stores, and provides access to, context for each request.

        Examples:
            >>> from python_to_tools.ptt import TaskFlowBehavior
            >>> behavior = TaskFlowBehavior(root_ai_model=model)
            >>> behavior.resolve_from_text("Get the weather in Sydney, Australia")
        """
        self.root_ai_model = root_ai_model

        self.context = context
        if not self.context:
            self.context = MemoryContext()

        self.root_agent = root_agent
        if not self.root_agent:
            self.root_agent = Agent(
                agent_name="root",
                convo_loader=JinjaConvoLoader(
                    "task_master_agent.j2",
                ),
                model=root_ai_model
            )

        self.summary_agent = summary_agent
        if not self.summary_agent:
            self.summary_agent = Agent(
                agent_name="summary",
                convo_loader=JinjaConvoLoader(
                    "root_summary.j2",
                ),
                model=root_ai_model
            )

        self.handler_agent = handler_agent
        if not self.handler_agent:
            self.handler_agent = Agent(
                agent_name="default_handler_agent",
                convo_loader=JinjaConvoLoader(
                    "single_task_agent.j2",
                ),
                model=root_ai_model
            )

        self.review_agent = review_agent
        if not self.review_agent:
            self.review_agent = Agent(
                agent_name="reviewer_agent",
                convo_loader=JinjaConvoLoader(
                    "task_review_agent.j2",
                ),
                model=root_ai_model
            )

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

    def _task_results_to_text(self, name: str, result: str):
        """Converts completed tasks into a textual representation.

        Within this style of Behavior handler, this step is very important as it's how AI ultimately works out
        how it did at handling the user's actual query
        """
        if result:
            return f"Task result:\nTask name: {name}. Result: \n{result.rstrip()}"

        return None

    def resolve_from_text(self, msg: str):
        """Resolve the given text into, first, a list of tasks, then close each task one by one using our associated
        handler agents."""
        logger.info("Resolving message into task list")
        task_list = TaskList.from_str(self.root_agent.generate_text(msg).content)
        task_results = {}

        success = True
        for task in task_list.list:
            if success:
                logger.info(f"resolving task: {task}")
                result = self.handler_agent.resolve_from_text(task, context=self.context)
                self.context.add_message(
                    self._task_results_to_text(task, result), role=MessageRoleEnum.assistant
                )
                logger.debug(result)
                review_result = self.review_agent.resolve_from_text(result, context=self.context)
                if "YES" not in review_result:
                    logger.warning(f"Flow cannot continue: {review_result}")
                    success = False


        summary = self.summary_agent.resolve_from_text(msg, context=self.context)

        return TaskFlowResult(summary=summary, task_output=task_results, success=success)
