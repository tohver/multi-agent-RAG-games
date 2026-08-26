from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from .messages import AIMessage, BaseMessage, TokenUsage, UserMessage
from .tooling import Tool


class LLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.tools: Dict[str, Tool] = {
            tool.name: tool for tool in (tools or [])
        }

    def register_tool(self, tool: Tool):
        """Make `tool` available to the model on subsequent calls."""
        self.tools[tool.name] = tool

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Assemble the request body for the chat completions API."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.dict() for m in messages],
        }

        if self.tools:
            payload["tools"] = [tool.dict() for tool in self.tools.values()]
            payload["tool_choice"] = "auto"

        return payload

    def _as_messages(self, input: Any) -> List[BaseMessage]:
        """Normalise a string, message, or message list into a message list.

        Raises:
            ValueError: If the input is none of those.
        """
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            return input
        else:
            raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(self, 
               input: str | BaseMessage | List[BaseMessage],
               response_format: BaseModel = None,) -> AIMessage:
        """Send the conversation to the model and return its reply.

        Args:
            input: A string, a single message, or a list of messages.
            response_format: Pydantic model the reply must match; switches to
                the structured-output endpoint.

        Returns:
            The assistant message, with tool calls and token usage attached.
        """
        messages = self._as_messages(input)
        payload = self._build_payload(messages)
        if response_format:
            payload.update({"response_format": response_format})
            response = self.client.beta.chat.completions.parse(**payload)
        else:
            response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )

        return AIMessage(
            content=message.content,
            tool_calls=message.tool_calls,
            token_usage=token_usage
        )
