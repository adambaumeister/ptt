from typing import Union

from python_to_tools.ai.cloudflare.client import CloudflareClient
from python_to_tools.utils import EnvironmentVariables


def get_model_by_environment_variables(
        environment_variables: EnvironmentVariables
) -> Union[CloudflareClient]:
    """Returns the relevant, configured AI model class based on configured environment variables."""
    if environment_variables.CLOUDFLARE_ACCOUNT_ID:
        return CloudflareClient(
            account_id=environment_variables.CLOUDFLARE_ACCOUNT_ID,
            api_token=environment_variables.CLOUDFLARE_API_TOKEN,
            model_id=environment_variables.CLOUDFLARE_MODEL_ID,
        )

    raise EnvironmentError("Environment is not configured for any model providers!")