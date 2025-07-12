import pathlib

import pytest

from python_to_tools.utils import EnvironmentVariables

@pytest.fixture
def env_vars():
    """Useful for acceptance tests - loads teh environment variables from a .env file in the root of this repository"""
    return EnvironmentVariables.load_from_env(
        dotenv_path=pathlib.Path(__file__).parent.parent.parent.parent.joinpath(".env")
    )
