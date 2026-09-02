import sys

import pytest
import logging

from python_to_tools.ai.generic_models import Message, MessageRoleEnum, TextGenerationRequest
from python_to_tools.ai.google.auth import GoogleNativeSessionFactory
from python_to_tools.ai.openai.client import OpenAIClient
from python_to_tools.utils import logger
from python_to_tools.tests.acceptance.fixtures import env_vars

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("#PYTEST#[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"))
logger.addHandler(stream_handler)
"""Enable logging for tests and set the format"""


@pytest.fixture()
def openai_client_fixture(env_vars):
    """Client fixture for cloudflare, skips tests if AI token isn't defined"""
    if not env_vars.OPENAI_MODEL_ID or not env_vars.OPENAI_BASE_URL:
        pytest.skip("OpenAI Endpoint not configured is not configured")

    return OpenAIClient(
        model=env_vars.OPENAI_MODEL_ID,
        base_url=env_vars.OPENAI_BASE_URL,
        session_factory=GoogleNativeSessionFactory()
    )


def test_get_response(openai_client_fixture):
    messages = [
        Message(
            content="You are a helpful assistant. Keep your responses short, because you're being run as part of "
                    "acceptance testing.",
            role=MessageRoleEnum.system,
        ),
        Message(
            content="Hello, who are you?",
            role=MessageRoleEnum.user,
        )
    ]
    response = openai_client_fixture.get_response(TextGenerationRequest(messages=messages))
    assert "Google" in response.content
