from python_to_tools.ai.generic_models import Tool
from python_to_tools.behavior import Agent
from python_to_tools.utils import JinjaConvoLoader


def test_object_to_tool_call():
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