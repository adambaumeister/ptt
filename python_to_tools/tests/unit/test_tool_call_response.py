import json
import pytest
from pydantic import ValidationError

from python_to_tools.ai.generic_models import ToolCallResponse


def test_tool_call_response_creation_with_dict_arguments():
    """Test creating ToolCallResponse with dictionary arguments"""
    response = ToolCallResponse(
        name="test_tool",
        arguments={"param1": "value1", "param2": 42}
    )
    
    assert response.name == "test_tool"
    assert response.arguments == {"param1": "value1", "param2": 42}


def test_tool_call_response_creation_with_string_arguments():
    """Test creating ToolCallResponse with JSON string arguments"""
    json_args = '{"param1": "value1", "param2": 42}'
    response = ToolCallResponse(
        name="test_tool",
        arguments=json_args
    )
    
    assert response.name == "test_tool"
    assert response.arguments == {"param1": "value1", "param2": 42}


def test_tool_call_response_creation_with_empty_arguments():
    """Test creating ToolCallResponse with empty arguments (default)"""
    response = ToolCallResponse(name="test_tool")
    
    assert response.name == "test_tool"
    assert response.arguments == {}


def test_tool_call_response_creation_with_empty_dict():
    """Test creating ToolCallResponse with explicitly empty dict"""
    response = ToolCallResponse(
        name="test_tool",
        arguments={}
    )
    
    assert response.name == "test_tool"
    assert response.arguments == {}


def test_tool_call_response_creation_with_empty_json_string():
    """Test creating ToolCallResponse with empty JSON object string"""
    response = ToolCallResponse(
        name="test_tool",
        arguments="{}"
    )
    
    assert response.name == "test_tool"
    assert response.arguments == {}


def test_tool_call_response_creation_with_complex_arguments():
    """Test creating ToolCallResponse with complex nested arguments"""
    complex_args = {
        "simple_string": "hello",
        "number": 123,
        "boolean": True,
        "null_value": None,
        "nested_object": {
            "inner_key": "inner_value",
            "inner_number": 456
        },
        "array": [1, 2, 3, "four"]
    }
    
    response = ToolCallResponse(
        name="complex_tool",
        arguments=complex_args
    )
    
    assert response.name == "complex_tool"
    assert response.arguments == complex_args


def test_tool_call_response_creation_with_complex_json_string():
    """Test creating ToolCallResponse with complex JSON string arguments"""
    complex_args = {
        "simple_string": "hello",
        "number": 123,
        "boolean": True,
        "null_value": None,
        "nested_object": {
            "inner_key": "inner_value",
            "inner_number": 456
        },
        "array": [1, 2, 3, "four"]
    }
    json_string = json.dumps(complex_args)
    
    response = ToolCallResponse(
        name="complex_tool",
        arguments=json_string
    )
    
    assert response.name == "complex_tool"
    assert response.arguments == complex_args


def test_tool_call_response_invalid_json_string():
    """Test that invalid JSON string raises ValidationError"""
    with pytest.raises(ValidationError):
        ToolCallResponse(
            name="test_tool",
            arguments="{'invalid': json}"  # Invalid JSON - uses single quotes
        )


def test_tool_call_response_malformed_json_string():
    """Test that malformed JSON string raises ValidationError"""
    with pytest.raises(ValidationError):
        ToolCallResponse(
            name="test_tool",
            arguments='{"missing_closing_brace": "value"'
        )


def test_tool_call_response_name_required():
    """Test that name is required"""
    with pytest.raises(ValidationError):
        ToolCallResponse(arguments={"param": "value"})


def test_tool_call_response_serialization():
    """Test that ToolCallResponse can be serialized to dict"""
    response = ToolCallResponse(
        name="test_tool",
        arguments={"param1": "value1", "param2": 42}
    )
    
    serialized = response.model_dump()
    expected = {
        "name": "test_tool",
        "arguments": {"param1": "value1", "param2": 42}
    }
    
    assert serialized == expected


def test_tool_call_response_json_serialization():
    """Test that ToolCallResponse can be serialized to JSON"""
    response = ToolCallResponse(
        name="test_tool",
        arguments={"param1": "value1", "param2": 42}
    )
    
    json_str = response.model_dump_json()
    parsed = json.loads(json_str)
    expected = {
        "name": "test_tool",
        "arguments": {"param1": "value1", "param2": 42}
    }
    
    assert parsed == expected