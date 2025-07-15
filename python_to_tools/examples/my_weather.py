import logging
from typing import Annotated

import requests
from pydantic import BaseModel

from python_to_tools.ai.generic_models import ToolParameter
from python_to_tools.ai.utils import get_model_by_environment_variables
from python_to_tools.ptt import TaskFlowBehavior
from python_to_tools.utils import EnvironmentVariables, JinjaConvoLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LocationResponse(BaseModel):
    """Response model for locations API"""
    country: str
    city: str
    lat: float
    lon: float


class WeatherHourlyTemperatureResponse(BaseModel):
    """Hourly data from weather API"""
    temperature_2m: list[float]


class WeatherResponse(BaseModel):
    """Response model for weather API"""
    hourly: WeatherHourlyTemperatureResponse

    @property
    def last_weather(self):
        return self.hourly.temperature_2m[-1]


def get_user_location():
    """Get the user's current location.

    This uses the public website 'ip-api.com' to resolve the user's current IP address to a given location

    This is an example of a function that returns a Pydantic model. The Behavior class will conver this into a
    textual representation so it can be fed back into AI, using `BaseModel.model_dump`
    """
    return LocationResponse(**requests.get("http://ip-api.com/json/").json())


def get_weather(
        latitude: Annotated[
            str, ToolParameter(type="string", description="The user's current latitude")
        ],
        longitude: Annotated[
            str, ToolParameter(type="string", description="The user's current longitude")
        ]
):
    """Get the weather in the given location, based on the provided latitude and longitude.

    Note this just returns the most recent weather."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m"
    }

    result = WeatherResponse(**requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params
    ).json())
    return str(result.last_weather)


model = get_model_by_environment_variables(EnvironmentVariables.load_from_env())
"""Loads the model based on the configured environment variables"""
behavior = TaskFlowBehavior(root_ai_model=model)
"""Init the agentic behavior controller"""

behavior.handler_agent.add_tool(get_user_location)
behavior.handler_agent.add_tool(get_weather)
"""Add the tools to the default task handler agent"""

if __name__ == '__main__':
    print(behavior.resolve_from_text('Get the weather in my current location'))
