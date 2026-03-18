from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
try:
    from anthropic.types import Model
    from anthropic._types import NOT_GIVEN as NOT_GIVEN, NotGiven as NotGiven
except Exception:  # Anthropic optional
    class Model(str):
        pass

    class NotGiven:
        pass

    NOT_GIVEN = NotGiven()


AnthropicChatModel: TypeAlias = Model


class AnthropicChoice(BaseModel):
    """An LLM from Anthropic."""

    provider: Literal["anthropic"] = "anthropic"
    model: AnthropicChatModel


class AnthropicChoiceDict(TypedDict):
    """An LLM from Anthropic."""

    provider: Literal["anthropic"]
    model: AnthropicChatModel
