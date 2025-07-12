from enum import Enum

from python_to_tools.ai.generic_models import TextGenerationRequest


class ModelTypeEnum(str, Enum):
    text_generation = "text_generation"
    image_generation = "image_generation"

class AiModelClient:
    def __init__(
            self,
            types: list[ModelTypeEnum],
    ):
        self.types = types

    def get_response(self, request: TextGenerationRequest):
        """
        Generate a response from the AI Model.
        """