from enum import Enum
from typing import Union

import requests

from python_to_tools.ai.generic_models import TextGenerationRequest
from python_to_tools.ai.generic_sessions import BearerAuthenticatedSessionFactory


class ModelTypeEnum(str, Enum):
    text_generation = "text_generation"
    image_generation = "image_generation"

class AiModelClient:
    def __init__(
            self,
            types: list[ModelTypeEnum],
    ):
        self.types = types
        self.session_factory = requests.Session

    def get_response(self, request: TextGenerationRequest):
        """Generate a response from the AI Model."""

    def _get_session(self):
        """Returns an authenticated session, for use with requests"""
        return self.session_factory()

    def _post(
            self,
            url: str,
            data: Union[dict, list]
    ):
        """Generic HTTP Post method."""
        return self._get_session().post(url, json=data)