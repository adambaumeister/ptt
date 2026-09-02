# User interactive flow

This example passes a custom function, `ask_user`, to retrieve information from the user using stdin.

It demonstrates how you can add user interactivity in a *predictable* way. Your behavior/agents only ask the user for
more information if it needs to.

```python title="Interactive behavior"
import logging
from typing import Annotated

from python_to_tools.ai.generic_models import ToolParameter
from python_to_tools.ai.utils import get_model_by_environment_variables
from python_to_tools.ptt import TaskFlowBehavior
from python_to_tools.utils import EnvironmentVariables, JinjaConvoLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def ask_user(question: Annotated[
    str,
    ToolParameter(type="string", description="The question to ask the user, if more information is required.")
]) -> str:
    """If information is missing, this collects it from the user using stdin."""
    return input(question + " ")

model = get_model_by_environment_variables(EnvironmentVariables.load_from_env())
"""Loads the model based on the configured environment variables"""
behavior = TaskFlowBehavior(root_ai_model=model)
"""Init the agentic behavior controller"""

behavior.handler_agent.add_tool(ask_user)
"""Add the tools to the default task handler agent"""

if __name__ == '__main__':
    print(behavior.resolve_from_text('Count the number of letters in my name').summary)
```