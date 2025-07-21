import inspect
import json
from enum import Enum
from typing import Optional, Callable, Any, Union

from pydantic import BaseModel, Field, AliasChoices, field_validator


class ToolParameter(BaseModel):
    """
    Represents a parameter definition for a tool.

    Attributes:
        type: The data type of the parameter (e.g., 'string', 'number')
        description: Human-readable description of the parameter
    """
    type: str
    description: str


class ToolParameters(BaseModel):
    """
    Represents the parameters schema for a tool.

    Attributes:
        type: The schema type (typically 'object')
        properties: Dictionary mapping parameter names to their definitions
        required: List of required parameter names
    """
    type: str = "object"
    properties: dict[str, ToolParameter]
    required: list[str]


class Tool(BaseModel):
    """
    Represents a tool that can be called by the AI model.

    Attributes:
        name: The name identifier of the tool
        description: Human-readable description of what the tool does
        parameters: The parameters schema for the tool
    """
    name: str
    description: str
    parameters: ToolParameters

    @classmethod
    def from_func(cls, func: Callable):
        description = func.__doc__
        if not description:
            raise ValueError("Tool must have a description, did you add a docstring to your function?")
        name = func.__name__

        properties = {}
        required = []
        for k,v in inspect.signature(func).parameters.items():
            if v.annotation:
                properties[k] = v.annotation.__metadata__[0]
                if not v.default:
                    required.append(k)

        return cls(
            name=name,
            description=description,
            parameters=ToolParameters(properties=properties, required=required)
        )

class MessageRoleEnum(str, Enum):
    """Generic Enum, supported by most OpenAI compatible models"""
    user = "user"
    system = "system"
    assistant = "assistant"

class Message(BaseModel):
    """
    Represents a message in the conversation.

    Attributes:
        role: The role of the message sender (e.g., 'user', 'assistant')
        content: The text content of the message
    """
    role: MessageRoleEnum
    content: str

class ToolCallResponse(BaseModel):
    """Generic representation of a tool call by AI"""
    name: str
    arguments: dict[str, Any] = {}

    @field_validator("arguments", mode='before')
    @classmethod
    def parse_args(cls, value: Union[str, dict]):
        if isinstance(value, str):
            return json.loads(value)

        return value

class TextGenerationResponse(BaseModel):
    """Generic, API independent Text Generation Response Model"""
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCallResponse]] = None

class TextGenerationRequest(BaseModel):
    """
    Represents a generic text generation request

    Attributes:
        messages: Optional list of conversation messages
        tools: Optional list of tools available to the AI model
    """
    messages: Optional[list[Message]] = Field(default=None, description="List of conversation messages")
    tools: Optional[list[Tool]] = Field(default=None, description="List of available tools")
