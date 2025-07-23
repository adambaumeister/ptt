from typing import Union, Optional, Callable

import openai
import requests
from openai.types.chat import ChatCompletionToolParam

from python_to_tools.ai.base import AiModelClient, ModelTypeEnum
from python_to_tools.ai.generic_models import TextGenerationRequest, TextGenerationResponse
from python_to_tools.ai.generic_sessions import BearerAuthenticatedSessionFactory
from python_to_tools.ai.google.auth import GoogleNativeSessionFactory

class UnauthenticatedSessionFactory():
    def __init__(self, verify: bool = True):
        self.verify = verify

    def __call__(self):
        session = requests.Session()
        session.verify = self.verify

class OpenAIClient(AiModelClient):
    """OpenAI API compatible model client.
    """
    def __init__(
            self,
            base_url: str,
            session_factory: Optional[Union[
                BearerAuthenticatedSessionFactory, GoogleNativeSessionFactory, UnauthenticatedSessionFactory
            ]] = None,
            model: str = "",
            model_type: ModelTypeEnum = ModelTypeEnum.text_generation

    ):
        """This model supports any API that conforms to the OpenAI schema, which is most of them at the time of writing.

        Arguments:
            base_url: Path to the OpenAI API base URL
            model: The model ID to use
            session_factory: The session factory to use for making requests.

        Examples:

            from python_to_tools.ai.openai.client import OpenAIClient
            from python_to_tools.ai.google.auth import GoogleNativeSessionFactory
            client = OpenAIClient(
                 model=env_vars.OPENAI_MODEL_ID,
                 base_url='https://example.com/openai',
                 session_factory=GoogleNativeSessionFactory()
            )

        Example (Basic Bearer token authentication):

            from python_to_tools.ai.openai.client import OpenAIClient
            client = OpenAIClient(
                 model=env_vars.OPENAI_MODEL_ID,
                 base_url='https://example.com/openai',
                 session_factory=BearerAuthenticatedSessionFactory("my-token-here")
            )

        Example (Unauthenticated):

            from python_to_tools.ai.openai.client import OpenAIClient, UnauthenticatedSessionFactory
            client = OpenAIClient(
                model=env_vars.OPENAI_MODEL_ID,
                base_url='https://example.com/openai',
                session_factory=UnauthenticatedSessionFactory()
            )
        """
        super().__init__([model_type])
        self.base_url = base_url
        self.session_factory = session_factory
        self.model = model

    def get_client(self):
        token = self.session_factory.bearer_token
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=token,
        )

    def get_response(self, request: Union[TextGenerationRequest]) -> TextGenerationResponse:
        """Generate a response from the AI Model."""
        client = self.get_client()
        tools = None
        if request.tools:
            tools = [ChatCompletionToolParam(
                function=t.model_dump(), type="function"
            ) for t in request.tools]

        response = client.chat.completions.create(
            messages=request.messages,
            model=self.model,
            tools=tools,
        )

        d = {
            "content": response.choices[0].message.content,
            "tool_calls": []
        }

        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                d["tool_calls"].append(
                    tc.function.model_dump()
                )

        return TextGenerationResponse(
            **d,
        )
