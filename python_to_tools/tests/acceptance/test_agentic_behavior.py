from typing import Annotated

import pytest

from python_to_tools.ai.generic_models import ToolParameter
from python_to_tools.behavior import Agent
from python_to_tools.ptt import TaskFlowBehavior
from python_to_tools.tests.acceptance.test_ai_cloudflare_client import (
    cloudflare_text_generation_client_fixture, env_vars
)
from python_to_tools.tests.acceptance.test_ai_openai_client import openai_client_fixture
from python_to_tools.tests.acceptance.fixtures import acceptance_test_results
from python_to_tools.utils import JinjaConvoLoader


@pytest.fixture
def agentic_behavior_fixture(cloudflare_text_generation_client_fixture) -> TaskFlowBehavior:
    """Get the root Agentic Behavior object"""
    return TaskFlowBehavior(cloudflare_text_generation_client_fixture)


def get_user_location():
    """Get the user's current location"""
    return "Sydney, NSW, Australia"

def get_weather(location: Annotated[str, ToolParameter(type="string", description="The user's current location")]):
    """Get the weather in the given location"""
    return 25

def get_weather_agent(model):
    weather_agent = Agent(
        agent_name="weather_agent",
        convo_loader=JinjaConvoLoader(
            "base_agent.j2",
        ),
        model=model
    )

    weather_agent.add_tool(get_user_location)
    weather_agent.add_tool(get_weather)
    return weather_agent

def test_get_the_weather_cloudflare(
        cloudflare_text_generation_client_fixture,
        agentic_behavior_fixture,
        acceptance_test_results
):
    from python_to_tools.ptt import TaskFlowBehavior, Agent
    behavior = TaskFlowBehavior(root_ai_model=cloudflare_text_generation_client_fixture)

    weather_agent = get_weather_agent(cloudflare_text_generation_client_fixture)

    behavior.add_task_handler_agent(weather_agent)
    result = behavior.resolve_from_text("get me the weather in my current location")
    acceptance_test_results.write(f"## Cloudflare AI - test_get_the_weather_cloudflare - Get the Weather\n{result.summary}\n")

def test_get_the_weather_sub_agents_cloudflare(
        cloudflare_text_generation_client_fixture, agentic_behavior_fixture, acceptance_test_results
):
    """Same functions, but this time we delegate the location to a sub agent to test that the resolution works
    with next_agent"""
    from python_to_tools.ptt import TaskFlowBehavior, Agent
    behavior = TaskFlowBehavior(root_ai_model=cloudflare_text_generation_client_fixture)

    weather_agent = Agent(
        agent_name="weather_agent",
        convo_loader=JinjaConvoLoader(
            "base_agent.j2",
        ),
        model=cloudflare_text_generation_client_fixture
    )
    location_agent = Agent(
        agent_name="location_agent",
        convo_loader=JinjaConvoLoader(
            "base_agent.j2",
        ),
        model=cloudflare_text_generation_client_fixture,
        description="Has the ability to resolve a user location to a city and latitude/longitude coordinates."
    )
    weather_agent.add_tool(get_weather)
    weather_agent.add_agent(location_agent)
    behavior.add_task_handler_agent(weather_agent)
    result = behavior.resolve_from_text("get me the weather in my current location")
    acceptance_test_results.write(f"## Cloudflare AI - test_get_the_weather_cloudflare_sub_agents - Get the Weather with proxy to another agent\n{result.summary}\n")


def test_get_the_weather_google(
        openai_client_fixture,
        agentic_behavior_fixture,
        acceptance_test_results
):
    from python_to_tools.ptt import TaskFlowBehavior, Agent
    behavior = TaskFlowBehavior(root_ai_model=openai_client_fixture)

    weather_agent = get_weather_agent(openai_client_fixture)

    behavior.add_task_handler_agent(weather_agent)
    result = behavior.resolve_from_text("get me the weather in my current location")
    acceptance_test_results.write(f"## Google AI - test_get_the_weather_google - Get the Weather\n{result.summary}\n")