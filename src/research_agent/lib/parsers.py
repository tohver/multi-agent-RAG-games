"""Turning an LLM reply into a typed object."""

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from .messages import AIMessage


class OutputParser(BaseModel, ABC):
    """Base class for anything that reads an `AIMessage` into a value."""

    @abstractmethod
    def parse(self, ai_message: AIMessage) -> Any:
        """Convert the message into the parser's target type."""


class PydanticOutputParser(OutputParser):
    """Validate an LLM reply against a Pydantic model.

    Pairs with `LLM.invoke(..., response_format=Model)`, which asks the API for
    JSON matching that model; this parser is what turns the JSON into the model.

    Attributes:
        model_class: The model the reply must satisfy.
    """

    model_class: Type[BaseModel]

    def parse(self, ai_message: AIMessage) -> BaseModel:
        """Validate the message content as JSON.

        Args:
            ai_message: The reply to parse.

        Returns:
            An instance of `model_class`.

        Raises:
            pydantic.ValidationError: If the content does not match the model.
        """
        return self.model_class.model_validate_json(ai_message.content)
