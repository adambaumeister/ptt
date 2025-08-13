import pathlib

import pytest

from python_to_tools.utils import EnvironmentVariables

@pytest.fixture
def env_vars():
    """Useful for acceptance tests - loads teh environment variables from a .env file in the root of this repository"""
    return EnvironmentVariables.load_from_env(
        dotenv_path=pathlib.Path(__file__).parent.parent.parent.parent.joinpath(".env")
    )

@pytest.fixture
def acceptance_test_results():
    fp = pathlib.Path(__file__).parent.parent.parent.parent.joinpath("docs").joinpath("acceptance.md")
    return open(fp, "w")


@pytest.fixture
def image():
    fp = pathlib.Path(__file__).parent.joinpath("test_data").joinpath("night_sky.jpg")
    return fp
