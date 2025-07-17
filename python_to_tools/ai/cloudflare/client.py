import inspect
from calendar import error
from json import JSONDecodeError
from typing import List, Optional, Union, Callable

import requests
from pydantic import BaseModel, Field

from python_to_tools.ai.base import AiModelClient, ModelTypeEnum
from python_to_tools.ai.generic_models import Message, Tool, ToolParameter, ToolParameters, TextGenerationRequest, \
    TextGenerationResponse
from python_to_tools.ai.generic_sessions import BearerAuthenticatedSessionFactory
from python_to_tools.utils import logger

class CloudflareTextGenerationResponseError(BaseModel):
    code: int
    message: str

class CloudflareTextGenerationResponseResult(BaseModel):
    response: Optional[str] = None
    tool_calls: Optional[list] = []
    usage: Optional[dict] = {}

class CloudflareTextGenerationResponse(BaseModel):
    """
    Cloudflare specific text generation response
    """
    result: Optional[CloudflareTextGenerationResponseResult] = None
    success: bool
    errors: Optional[list[CloudflareTextGenerationResponseError]] = []

    def to_text_generation_response(self) -> TextGenerationResponse:
        return TextGenerationResponse(
            content=self.result.response,
            tool_calls=self.result.tool_calls
        )

class CloudflareRequest(BaseModel):
    """
    Represents a request to the Cloudflare Workers AI API.
    
    This class validates and structures requests that include conversation messages
    and available tools for the AI model to use.
    
    Attributes:
        messages: Optional list of conversation messages
        tools: Optional list of tools available to the AI model
    """
    messages: Optional[List[Message]] = Field(default=None, description="List of conversation messages")
    tools: Optional[List[Tool]] = Field(default=None, description="List of available tools")


class CloudflareRequestError(Exception):
    pass

class CloudflareClient(AiModelClient):
    def __init__(
            self,
            account_id: str,
            api_token: str,
            model_id: str,
            model_type: ModelTypeEnum = ModelTypeEnum.text_generation
    ):
        """
        Supports AI Models hosted in the Cloudflare `Workers AI` infrastructure.

        Arguments:
            account_id (str): Your Cloudflare Account ID
            api_token (str): Your Cloudflare API Token
            model_id (str): The model to use in workers AI
            model_type (ModelTypeEnum): Type of the model, defaults to 'text_generation'

        Examples:
            >>> from python_to_tools.ai.cloudflare import CloudflareClient
            >>> client = CloudFlareClient(
            >>>     account_id='your-account-id',
            >>>     api_token='your-api-token',
            >>>     model_id='your-model-id',
            >>> )
        """
        super().__init__([model_type])
        self.account_id = account_id
        self.api_token = api_token
        self.model_id = model_id
        self.session_factory = BearerAuthenticatedSessionFactory(self.api_token)


    def _get_session(self):
        """Returns an authenticated session, for use with requests"""
        return self.session_factory()

    def _get_url(self):
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_id}"

    @staticmethod
    def _read_response(response: requests.Response, response_class: Callable):
        try:
            data = response.json()
        except JSONDecodeError as e:
            raise CloudflareRequestError("Failed to decode response from cloudflare") from e


        return response_class(**data)

    def _post(
            self,
            url: str,
            data: Union[dict, list]
    ):
        """Generic HTTP Post method."""
        return self._get_session().post(url, json=data)

    def get_response(self, request: Union[CloudflareRequest, TextGenerationRequest]) -> TextGenerationResponse:
        """
        Generate a response from the AI Model.
        """
        result = self._read_response(
            self._post(self._get_url(), data=request.model_dump()), CloudflareTextGenerationResponse
        )
        logger.debug(result)
        if not result.success:
            logger.info(f"Failed to generate response to {request.messages}")
            raise CloudflareRequestError(
                f"Cloudflare request failed: {result.errors[0].message}"
            )
        return result.to_text_generation_response()