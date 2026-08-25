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
        '''
        In plain English: opens one connection to the language model.

        Worth creating these sparingly. Each one carries its own pool of network
        connections, so building a fresh one for every question would throw away the
        ability to reuse an open connection - noticeably slower over many calls. This
        project builds one per role and keeps it.

        Output: nothing returned; the connection is ready for `invoke`.
        '''
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.tools: Dict[str, Tool] = {
            tool.name: tool for tool in (tools or [])
        }

    def register_tool(self, tool: Tool):
        '''
        In plain English: gives this connection one more tool the model may call.

        Output: nothing returned. The tool is added to the list sent with each request.
        '''
        self.tools[tool.name] = tool

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        '''
        In plain English: assembles the request that will actually be sent.

        Which model, how creative to be, the conversation so far, and - if any tools are
        registered - their descriptions, along with permission for the model to use them.

        Output: a dictionary ready to hand to the API. Not meant to be called from
        outside; `invoke` uses it.
        '''
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.dict() for m in messages],
        }

        if self.tools:
            payload["tools"] = [tool.dict() for tool in self.tools.values()]
            payload["tool_choice"] = "auto"

        return payload

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        '''
        In plain English: accepts whatever shape the caller found convenient and turns
        it into a proper conversation.

        A bare string, a single message or a list of messages all end up as a list of
        messages, so nothing downstream has to check.

        Output: a list of messages. Raises if given something it cannot make sense of,
        rather than sending a malformed request.
        '''
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
        '''
        In plain English: sends the conversation to the model and returns its reply.
        Every model call in this whole project passes through here.

        Passing a `response_format` changes the request in an important way: instead of
        free prose, the model is required to answer in a given shape. That is what makes
        the judge's verdict something the program can branch on rather than something it
        has to interpret.

        Output: the reply, along with any tools the model wants called and how many
        tokens were used. The token count feeds the cost estimate in the evaluation.
        '''
        messages = self._convert_input(input)
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
