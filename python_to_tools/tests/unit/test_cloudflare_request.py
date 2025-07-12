import pytest
from python_to_tools.ai.cloudflare.client import CloudflareRequest, CloudflareClient
from python_to_tools.ai.generic_models import Message, Tool, ToolParameters, ToolParameter


def test_cloudflare_request_validation():
    """Test that CloudflareRequest can validate the provided JSON structure."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "what is the weather in london?",
            },
        ],
        "tools": [
            {
                "name": "getWeather",
                "description": "Return the weather for a latitude and longitude",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "string",
                            "description": "The latitude for the given location",
                        },
                        "longitude": {
                            "type": "string",
                            "description": "The longitude for the given location",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        ],
    }
    
    request = CloudflareRequest(**data)
    
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "what is the weather in london?"
    
    assert len(request.tools) == 1
    assert request.tools[0].name == "getWeather"
    assert request.tools[0].description == "Return the weather for a latitude and longitude"
    assert request.tools[0].parameters.type == "object"
    assert "latitude" in request.tools[0].parameters.properties
    assert "longitude" in request.tools[0].parameters.properties
    assert request.tools[0].parameters.required == ["latitude", "longitude"]


def test_message_validation():
    """Test Message model validation."""
    message_data = {
        "role": "user",
        "content": "Hello world"
    }
    
    message = Message(**message_data)
    assert message.role == "user"
    assert message.content == "Hello world"


def test_tool_validation():
    """Test Tool model validation."""
    tool_data = {
        "name": "testTool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                }
            },
            "required": ["param1"]
        }
    }
    
    tool = Tool(**tool_data)
    assert tool.name == "testTool"
    assert tool.description == "A test tool"
    assert tool.parameters.type == "object"
    assert "param1" in tool.parameters.properties
    assert tool.parameters.required == ["param1"]


def test_invalid_data_raises_validation_error():
    """Test that invalid data raises ValidationError."""
    invalid_data = {
        "messages": [
            {
                "role": "user",
                # Missing required 'content' field
            },
        ],
        "tools": []
    }
    
    with pytest.raises(Exception):  # Pydantic will raise ValidationError
        CloudflareRequest(**invalid_data)


def test_cloudflare_request_with_empty_fields():
    """Test that CloudflareRequest works with empty optional fields."""
    request = CloudflareRequest()
    
    assert request.messages is None
    assert request.tools is None


def test_cloudflare_request_with_only_messages():
    """Test that CloudflareRequest works with only messages field."""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "Hello world"
            }
        ]
    }
    
    request = CloudflareRequest(**data)
    
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello world"
    assert request.tools is None


def test_cloudflare_request_with_only_tools():
    """Test that CloudflareRequest works with only tools field."""
    data = {
        "tools": [
            {
                "name": "testTool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "First parameter"
                        }
                    },
                    "required": ["param1"]
                }
            }
        ]
    }
    
    request = CloudflareRequest(**data)
    
    assert request.messages is None
    assert len(request.tools) == 1
    assert request.tools[0].name == "testTool"


def test_function_to_tool_call_basic():
    """Test converting a basic function to tool specification."""
    def get_weather(latitude: str, longitude: str):
        """Return the weather for a latitude and longitude"""
        return f"Weather for {latitude}, {longitude}"
    
    tool = CloudflareClient._function_to_tool_call(get_weather)
    
    assert tool.name == "get_weather"
    assert tool.description == "Return the weather for a latitude and longitude"
    assert tool.parameters.type == "object"
    assert len(tool.parameters.properties) == 2
    assert "latitude" in tool.parameters.properties
    assert "longitude" in tool.parameters.properties
    assert tool.parameters.properties["latitude"].type == "string"
    assert tool.parameters.properties["longitude"].type == "string"
    assert tool.parameters.required == ["latitude", "longitude"]


def test_function_to_tool_call_with_optional_params():
    """Test converting a function with optional parameters."""
    def calculate(x: int, y: int, operation: str = "add"):
        """Perform a calculation on two numbers"""
        if operation == "add":
            return x + y
        return x - y
    
    tool = CloudflareClient._function_to_tool_call(calculate)
    
    assert tool.name == "calculate"
    assert tool.description == "Perform a calculation on two numbers"
    assert len(tool.parameters.properties) == 3
    assert "x" in tool.parameters.properties
    assert "y" in tool.parameters.properties
    assert "operation" in tool.parameters.properties
    assert tool.parameters.properties["x"].type == "integer"
    assert tool.parameters.properties["y"].type == "integer"
    assert tool.parameters.properties["operation"].type == "string"
    assert tool.parameters.required == ["x", "y"]  # operation is optional


def test_function_to_tool_call_different_types():
    """Test converting a function with different parameter types."""
    def complex_function(name: str, age: int, score: float, active: bool):
        """A function with various parameter types"""
        return f"{name} is {age} years old with score {score}, active: {active}"
    
    tool = CloudflareClient._function_to_tool_call(complex_function)
    
    assert tool.parameters.properties["name"].type == "string"
    assert tool.parameters.properties["age"].type == "integer"
    assert tool.parameters.properties["score"].type == "number"
    assert tool.parameters.properties["active"].type == "boolean"
    assert len(tool.parameters.required) == 4


def test_function_to_tool_call_no_docstring():
    """Test that function without docstring raises ValueError."""
    def no_doc_function():
        pass
    
    with pytest.raises(ValueError, match="must have a docstring"):
        CloudflareClient._function_to_tool_call(no_doc_function)


def test_function_to_tool_call_with_varargs():
    """Test function with *args and **kwargs (should be ignored)."""
    def varargs_function(required: str, *args, **kwargs):
        """A function with variable arguments"""
        return f"Required: {required}"
    
    tool = CloudflareClient._function_to_tool_call(varargs_function)
    
    assert len(tool.parameters.properties) == 1
    assert "required" in tool.parameters.properties
    assert tool.parameters.required == ["required"]