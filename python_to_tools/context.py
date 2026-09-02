import abc
from python_to_tools.utils import logger
from python_to_tools.ai.generic_models import Message, MessageRoleEnum


class Context(abc.ABC):
    def __init__(self):
        self.messages = []

    @abc.abstractmethod
    def last_message(self):
        pass

    @abc.abstractmethod
    def add_message(self, content, role: MessageRoleEnum = "assistant"):
        pass


class MemoryContext(Context):
    def __init__(self):
        """Basic, in memory context handler.

        The point of any Context handler is to be passed to resolution functions so that the configured agents
        or behavior tasks
        """
        super().__init__()
        self.messages: list[Message] = []

    @property
    def last_message(self):
        return self.messages[-1]

    def add_message(self, content, role: MessageRoleEnum = "assistant"):
        """Add a message to the context store"""
        if not content:
            logger.info("No content provided when adding messages to context store")
        self.messages.append(
            Message(
                content=content,
                role=role
            )
        )