from typing import Annotated
from python_to_tools.ai.generic_models import ToolParameter

def example_function(
        x: Annotated[
            str, ToolParameter(
                type="string",
                description="Example argument without default",
            )
        ]):
    """This is just an example function, it doesn't do anything!"""
    pass

def test_create_tool_from_function():
    from python_to_tools.ai.generic_models import Tool
    result = Tool.from_func(example_function)