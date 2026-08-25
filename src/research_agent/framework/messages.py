from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

from .tooling import ToolCall


class BaseMessage(BaseModel):
    role: str
    content: Optional[str] = ""

    def dict(self) -> Dict:
        '''
        In plain English: turns a message into a plain dictionary for sending to the API.

        The rest of the code works with typed message objects, which catch mistakes
        early. The API wants plain data. This is the conversion point.

        Output: the message as a dictionary. Called once per message just before a
        request goes out.
        '''
        return dict(self)


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AIMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[List[ToolCall]] = None
    token_usage: Optional[TokenUsage] = None
