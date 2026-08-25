import inspect
import datetime
from typing import (
    Any, Callable, 
    Literal, Optional, Union, TypeAlias,
    get_type_hints, get_origin, get_args,
)
from functools import wraps
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall


# Type alias for OpenAI's tool call implementation
ToolCall: TypeAlias = ChatCompletionMessageToolCall

class Tool:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        '''
        In plain English: takes an ordinary Python function and packages it as something
        a language model can call.

        It reads the function's own name, its description and the types of its arguments,
        and turns them into a machine-readable description. The important consequence:
        the function's documentation is what the model reads to decide when to use it, so
        that text is part of the program's behaviour, not just a note for humans.

        Output: nothing returned; the wrapper is ready to be handed to a model.
        '''
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func)
        self.signature = inspect.signature(func, eval_str=True)
        self.type_hints = get_type_hints(func)

        self.parameters = [
            self._build_param_schema(key, param)
            for key, param in self.signature.parameters.items()
        ]

    def _build_param_schema(self, name: str, param: inspect.Parameter):
        '''
        In plain English: describes one argument - its name, its type, and whether it is
        required.

        An argument counts as required exactly when it has no default value.

        Output: a small description of that one argument, collected with the others.
        '''
        param_type = self.type_hints.get(name, str)
        schema = self._infer_json_schema_type(param_type)
        return {
            "name": name,
            "schema": schema,
            "required": param.default == inspect.Parameter.empty
        }

    def _infer_json_schema_type(self, typ: Any) -> dict:
        '''
        In plain English: translates a Python type into the vocabulary the model
        understands.

        Text, whole numbers, decimals, true/false, lists, dictionaries and fixed choices
        all have an equivalent. Anything unrecognised falls back to text, which is
        harmless but vague - a reason to keep tool arguments simple.

        Output: the type description for one argument.
        '''
        origin = get_origin(typ)

        # Handle Literal (enums)
        if origin is Literal:
            return {
                "type": "string",
                "enum": list(get_args(typ))
            }

        # Handle Optional[T]
        if origin is Union:
            args = get_args(typ)
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return self._infer_json_schema_type(non_none[0])
            return {"type": "string"}  # fallback

        # Handle collections
        if origin is list:
            return {
                "type": "array",
                "items": self._infer_json_schema_type(get_args(typ)[0] if get_args(typ) else str)
            }

        if origin is dict:
            return {
                "type": "object",
                "additionalProperties": self._infer_json_schema_type(get_args(typ)[1] if get_args(typ) else str)
            }

        # Primitive mappings
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            datetime.date: "string",
            datetime.datetime: "string",
        }

        return {"type": mapping.get(typ, "string")}

    def dict(self) -> dict:
        '''
        In plain English: the finished description of this tool, in the exact shape the
        API expects.

        Output: a dictionary with the tool's name, its description and its arguments.
        Sent with the request so the model knows this tool exists and how to call it.
        '''
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: param["schema"]
                        for param in self.parameters
                    },
                    "required": [
                        param["name"] for param in self.parameters if param["required"]
                    ],
                    "additionalProperties": False
                }
            }
        }

    def __call__(self, *args, **kwargs):
        '''
        In plain English: actually runs the underlying function.

        Because of this, a wrapped tool can be called exactly like the plain function it
        came from - which is why the agent's steps can call tools directly without
        unwrapping anything.

        Output: whatever the wrapped function returns.
        '''
        return self.func(*args, **kwargs)

    def __repr__(self):
        '''
        In plain English: a short readable summary of the tool, for printing.

        Output: text like `<Tool name=retrieve_game params=['query']>`.
        '''
        return f"<Tool name={self.name} params={[p['name'] for p in self.parameters]}>"

    @classmethod
    def from_func(cls, func: Callable):
        '''
        In plain English: an alternative way to wrap a function, for readers who prefer
        `Tool.from_func(f)` to `Tool(f)`.

        Output: the wrapped tool. Identical to calling `Tool(f)` directly.
        '''
        return cls(func)



def tool(func=None, *, name: str = None, description: str = None):
    '''
    In plain English: a shorthand so a function can be turned into a tool by writing
    `@tool` above it.

    Works either bare or with options, which is why it looks more complicated than
    it is.

    Output: the wrapped tool, replacing the original function under the same name.
    '''
    def wrapper(f):
        '''
        In plain English: does the actual wrapping once the options, if any, are known.

        Output: the wrapped tool.
        '''
        @wraps(f)
        def wrapped(*args, **kwargs):
            '''
            In plain English: a pass-through copy of the original function.

            Note it is created and then never used - `wrapper` returns the wrapped tool
            built directly from the original function instead. Harmless, but it is dead
            code, and worth deleting if this file is ever tidied.

            Output: whatever the original function returns.
            '''
            return f(*args, **kwargs)
        return Tool(f, name=name, description=description)
    
    # @tool ou @tool(name="foo")
    return wrapper(func) if func else wrapper