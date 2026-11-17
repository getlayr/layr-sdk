from layr.integrations.autogen import LayrConversableAgent
from layr.integrations.crewai import LayrCrew

__all__ = [
    "LayrAsyncCallbackHandler",
    "LayrCallbackHandler",
    "LayrConversableAgent",
    "LayrCrew",
    "require_review_tool",
]


def __getattr__(name: str) -> object:
    if name in {
        "LayrCallbackHandler",
        "LayrAsyncCallbackHandler",
        "require_review_tool",
    }:
        from layr.integrations import langchain as langchain_integration

        return getattr(langchain_integration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
