from typing import Union, Optional, Callable

import openai
from openai.types.chat import ChatCompletionToolParam

from python_to_tools.ai.base import AiModelClient, ModelTypeEnum
from python_to_tools.ai.generic_models import TextGenerationRequest, TextGenerationResponse
from python_to_tools.ai.generic_sessions import BearerAuthenticatedSessionFactory
from python_to_tools.ai.google.auth import GoogleNativeSessionFactory

class OpenAIClient(AiModelClient):
    """Wraps the OpenAI Client to only return the generic response models"""
    def __init__(
            self,
            base_url: str,
            session_factory: Optional[Union[
                BearerAuthenticatedSessionFactory, GoogleNativeSessionFactory
            ]] = None,
            model: str = "",
            model_type: ModelTypeEnum = ModelTypeEnum.text_generation

    ):
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
