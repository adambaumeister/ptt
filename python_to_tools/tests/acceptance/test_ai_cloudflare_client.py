import sys

import pytest
from python_to_tools.ai.generic_models import MessageRoleEnum, Tool, ToolParameters, ToolParameter
from python_to_tools.ai.cloudflare.client import CloudflareClient
from python_to_tools.utils import logger
from python_to_tools.tests.acceptance.fixtures import env_vars

@pytest.fixture()
def cloudflare_text_generation_client_fixture(env_vars):
    """Client fixture for cloudflare, skips tests if AI token isn't defined"""
    if not env_vars.CLOUDFLARE_API_TOKEN:
        pytest.skip("Cloudflare is not configured")

    return CloudflareClient(
        api_token=env_vars.CLOUDFLARE_API_TOKEN,
        account_id=env_vars.CLOUDFLARE_ACCOUNT_ID,
        model_id="@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    )

def test_cloudflare_get_response(cloudflare_text_generation_client_fixture):
    """Tests connectivity to the cloudflare API platform and pydantic field validation"""
    from python_to_tools.ai.cloudflare.client import CloudflareRequest, Message
    request = CloudflareRequest(
        messages=[
            Message(
                content="You are a helpful assistant. Keep your responses short, because you're being run as part of "
                        "acceptance testing.",
                role=MessageRoleEnum.system,
            ),
            Message(
                content="Hello!",
                role=MessageRoleEnum.user,
            )
        ]
    )
    response = cloudflare_text_generation_client_fixture.get_response(request)
    assert response.success
    logger.info(response)


def test_cloudflare_execute_tool(cloudflare_text_generation_client_fixture):
    """Tests connectivity to the cloudflare API platform and pydantic field validation"""
    from python_to_tools.ai.cloudflare.client import CloudflareRequest, Message
    request = CloudflareRequest(
        messages=[
            Message(
                content="You are a helpful assistant. You will be given tools to choose from to perform an action based"
                        "on what the user has requested, and you are expected to always choose one.",
                role=MessageRoleEnum.system,
            ),
            Message(
                content="Get the weather in sydney",
                role=MessageRoleEnum.user,
            )
        ],
        tools=[
            Tool(
                name="get_weather",
                description="Gets the weather in the user provided location",
                parameters=ToolParameters(
                    properties={
                        "location": ToolParameter(
                            type="str",
                            description="The location, of which, to check the weather"
                        )
                    },
                    required=["location"]
                )
            )
        ]
    )
    response = cloudflare_text_generation_client_fixture.get_response(request)
    assert response.result.tool_calls