import abc
import pathlib
import re
from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from jinja2 import Environment, PackageLoader, Template

from python_to_tools.ai.generic_models import Message, TextGenerationRequest

logger = logging.getLogger(__name__)
"""Root logger"""

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() - %(levelname)s] %(message)s")
"""Set the default logging format"""

logger.setLevel(logging.DEBUG)
"""Configure a default logging level (info)"""

DEFAULT_TEMPLATE_ENVIRONMENT = Environment(loader=PackageLoader('python_to_tools', 'templates'))
"""Get the default prompt template J2 environment"""

class ConvoParseError(Exception):
    pass

class Convo(BaseModel):
    """Object representation of an entire conversation.

    `convo` objects are designed to be portable between models and types of agentic behavior. """
    messages: list[Message]
    comments: str
    metadata: dict

    def as_text_generation_request(self):
        """Convert this object into the generic, model compatible text generation request"""
        return TextGenerationRequest(
            messages=self.messages,
        )

class ConvoParser:
    def __init__(self):
        self.buffer = []
        self.i = 0
        self.handlers = {
            r'"""': self.handle_comment_start,
            r'\[[a-zA-Z]+\]': self.handle_message
        }

        self.comments: str = ""
        self.messages: list[dict] = []

    def handle_message(self, i: int, lines: list[str]):
        role_regex = r'\[([a-zA-Z\.]+)\]'
        role = re.search(role_regex, lines[i]).group(1)
        buffer = ""
        if not role:
            raise ConvoParseError(f"Error parsing convo file at line {i}. Invalid role specification "
                                  f"(role can only contain alphabetic characters)")
        i = i + 1
        for line in lines[i:]:
            if re.match(role_regex, line):
                self.messages.append({
                    "role": role,
                    "content": buffer.strip()
                })
                # In this case, the top level handler should parse the current line
                return i - 1
            buffer += line + "\n"
            i += 1

        self.messages.append({
            "role": role,
            "content": buffer
        })

        return i

    def handle_comment_start(self, i: int, lines: list[str]):
        i = i + 1
        for line in lines[i:]:
            if line.startswith('"""'):
                return i

            self.comments += line + "\n"
            i += 1

        return i


    def parse_convo_file(self, data: str):
        lines = data.splitlines()
        while self.i < len(lines):
            current_line = lines[self.i]
            for handler_match_re, handler in self.handlers.items():
                if re.match(handler_match_re, current_line):
                    self.i = handler(self.i, lines)

            self.i += 1

        return Convo(
            messages=[
                Message(**i) for i in self.messages
            ],
            metadata=dict(),
            comments=self.comments,
        )

class ConvoLoader(abc.ABC):
    @abc.abstractmethod
    def to_convo(self, **kwargs) -> Convo:
        """Convert the given loader to a `Convo` object"""
        pass

class JinjaConvoLoader(ConvoLoader):
    """Jinja2 Based Convo Loader

    The simplest method of describing and building agent behavior (i.e Prompts), using Jinja2 templates.
    """
    def __init__(
            self,
            template_name: str | None = None,
            environment: Environment | None = None,
            template_path: str = None,
    ):
        """
        Create a Jinja2 Convo loader. If an environment is not provided, we will use the default templates shipped
        with this library.

        Arguments:
            template_name (str): The name of the Jinja template to use.
            environment (Environment): The Jinja2 environment to use.
        """
        self.environment = environment
        if not self.environment:
            self.environment = DEFAULT_TEMPLATE_ENVIRONMENT
        self.template_path = template_path
        self.template_name = template_name

    def to_convo(self, **kwargs):
        return self.prompt_template_to_convo(**kwargs)

    def prompt_template_to_convo(self, **kwargs) -> Convo:
        """Convenience function, allows us to store prompt templates as simple files"""
        if self.template_name:
            template = self.environment.get_template(self.template_name)
        elif pathlib.Path(self.template_path).is_file():
            with open(self.template_path) as template_file:
                template = Template(template_file.read())
        else:
            raise EnvironmentError(f"No valid j2 template was found.")

        rendered_prompt_template = template.render(**kwargs)
        #logger.debug(rendered_prompt_template)
        parser = ConvoParser()
        return parser.parse_convo_file(rendered_prompt_template)


class EnvironmentVariables(BaseModel):
    """All environment variables that we need get defined in this class
    """
    CLOUDFLARE_API_TOKEN: Optional[str] = ""
    """Cloudflare API token"""

    CLOUDFLARE_ACCOUNT_ID: Optional[str] = ""
    """Cloudflare Account ID"""

    CLOUDFLARE_MODEL_ID: Optional[str] = ""
    """Cloudflare Model ID"""

    OPENAI_MODEL_ID: Optional[str] = ""
    """OpenAI Model ID"""

    OPENAI_BASE_URL: Optional[str] = ""
    """OpenAI Base URL"""

    OPENAI_API_TOKEN: Optional[str] = ""
    """OpenAI Compatible API token"""

    @classmethod
    def load_from_env(cls, dotenv_path: Optional[pathlib.Path] = None):
        if dotenv_path:
            logger.info(f"Loaded environment variables from {dotenv_path}")
            load_dotenv(dotenv_path)
        else:
            logger.info("Loaded environment variables from .env file")
            load_dotenv()

        return cls(**os.environ)