from python_to_tools.ai.generic_models import Tool, ToolCallResponse
from python_to_tools.behavior import Agent
from python_to_tools.utils import JinjaConvoLoader


def test_function_to_tool_call():
    """Test the next_agent functionality works"""
    agent = Agent(
        agent_name="root",
        convo_loader=JinjaConvoLoader("base_agent.j2"),
        model=None
    )

    agent.add_agent(
        Agent(
            agent_name="sub_agent",
            convo_loader=JinjaConvoLoader("base_agent.j2"),
            model=None
        )
    )

    assert Tool.from_func(agent.next_agent).name == "next_agent"

def test_class_to_tool_calls():
    """Test that an object can be converted to tool calls"""
    agent = Agent(
        agent_name="root",
        convo_loader=JinjaConvoLoader("base_agent.j2"),
        model=None
    )
    class ExampleClass:
        def __init__(self):
            self.z = 10

        def add(self, x, y):
            """Add two numbers together"""
            return x+y+self.z

    obj = ExampleClass()
    agent.add_tools_from_object(obj)
    result = agent.call_tool_from_tool_response(ToolCallResponse(
        name="add",
        arguments={"x": 1, "y": 2},
    ))
    assert result == 13