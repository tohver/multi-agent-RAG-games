"""Turning an LLM reply into a typed object."""

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from .messages import AIMessage


class OutputParser(BaseModel, ABC):
    """Base class for anything that reads an `AIMessage` into a value."""

    @abstractmethod
    def parse(self, ai_message: AIMessage) -> Any:
        '''
        In plain English: the shared promise that every parser can turn a model reply
        into something usable.

        It has no body on purpose - it exists so different parsers can be swapped for
        one another without the calling code caring which is in use.

        Output: defined by whichever parser is actually doing the work.
        '''
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
        '''
        In plain English: checks that the model's reply has the shape it was asked for,
        and converts it into a proper object.

        The model was told to answer in a fixed format. Trusting that blindly is how
        programs break at three in the morning, so this validates the reply against the
        expected shape and raises if it does not match.

        Output: a validated object with real fields, so the caller can write
        `report.useful` instead of digging through text. This is what turns the judge's
        verdict into something the pipeline can branch on.
        '''
        """Validate the message content as JSON.

        Args:
            ai_message: The reply to parse.

        Returns:
            An instance of `model_class`.

        Raises:
            pydantic.ValidationError: If the content does not match the model.
        """
        return self.model_class.model_validate_json(ai_message.content)
