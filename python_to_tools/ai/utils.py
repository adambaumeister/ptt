from typing import Union

from python_to_tools.ai.cloudflare.client import CloudflareClient
from python_to_tools.ai.generic_sessions import BearerAuthenticatedSessionFactory
from python_to_tools.ai.google.auth import GoogleNativeSessionFactory
from python_to_tools.ai.openai.client import OpenAIClient
from python_to_tools.utils import EnvironmentVariables
from python_to_tools.utils import logger

def get_model_by_environment_variables(
        environment_variables: EnvironmentVariables = None
) -> Union[CloudflareClient, OpenAIClient]:
    """Returns the relevant, configured AI model class based on configured environment variables."""

    if not environment_variables:
        logger.info("Loading environment variables from default environment")
        environment_variables = EnvironmentVariables().load_from_env()

    if environment_variables.CLOUDFLARE_ACCOUNT_ID:
        logger.info("Using Cloudflare AI model Client")

        return CloudflareClient(
            account_id=environment_variables.CLOUDFLARE_ACCOUNT_ID,
            api_token=environment_variables.CLOUDFLARE_API_TOKEN,
            model_id=environment_variables.CLOUDFLARE_MODEL_ID,
        )

    if environment_variables.OPENAI_MODEL_ID:
        logger.info("Using openAI AI model Client")
        client = OpenAIClient(
            base_url=environment_variables.OPENAI_BASE_URL,
            model=environment_variables.OPENAI_MODEL_ID,
        )
        if environment_variables.OPENAI_API_TOKEN:
            client.session_factory = BearerAuthenticatedSessionFactory(environment_variables.OPENAI_API_TOKEN)
        else:
            logger.info("using google native session factory for connecting to OpenAI API (via Vertex)")
            client.session_factory = GoogleNativeSessionFactory()

        return client

    raise EnvironmentError("Environment is not configured for any model providers!")