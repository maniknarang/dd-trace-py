"""Listeners of the litellm contrib events, which evaluate LiteLLM SDK calls.
The guardrail of the LiteLLM proxy is in litellm.py.
"""

from typing import Any
from typing import Callable

from ddtrace.aiguard._api_client import AIGuardAbortError
from ddtrace.aiguard._api_client import AIGuardClient
from ddtrace.aiguard._api_client import Message
from ddtrace.aiguard._common import _get
from ddtrace.aiguard._common import evaluate_auto
from ddtrace.aiguard._constants import AI_GUARD
from ddtrace.aiguard._context import is_aiguard_context_active
from ddtrace.aiguard._context import reset_aiguard_context_active
from ddtrace.aiguard._context import set_aiguard_context_active
from ddtrace.aiguard.integrations._openai_chat import _convert_openai_messages
from ddtrace.aiguard.integrations._openai_chat import _convert_openai_response
import ddtrace.internal.logger as ddlogger
from ddtrace.internal.utils import get_argument_value


logger = ddlogger.get_logger(__name__)


def _claim_call(state: dict[str, Any], kwargs: dict[str, Any]) -> bool:
    """Claim the call so the provider listeners skip it. Proxy requests (left to the guardrail) and calls
    under an active claim (of an outer call or another integration) get no claim.
    """
    if is_aiguard_context_active() or "proxy_server_request" in kwargs:
        return False
    state["claim"] = set_aiguard_context_active()
    return True


def _evaluate(client: AIGuardClient, convert: Callable[..., list[Message]], *args: Any) -> None:
    """Evaluate the messages that convert returns. Only a block propagates: other errors fail open."""
    try:
        messages = convert(*args)
        if messages:
            evaluate_auto(client, messages, AI_GUARD.INTEGRATION_LITELLM)
    except AIGuardAbortError:
        raise
    except Exception:
        logger.debug("AI Guard litellm: failed to evaluate the call", exc_info=True)


def _chat_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Message]:
    messages = get_argument_value(args, kwargs, 1, "messages", optional=True)
    return _convert_openai_messages(messages) if isinstance(messages, (list, tuple)) else []


def _chat_prompt_to_evaluate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Message]:
    # Like the OpenAI listener, evaluate a prompt that ends with a user or tool message.
    messages = _chat_prompt(args, kwargs)
    return messages if messages and messages[-1].get("role") in ("user", "tool") else []


def _chat_conversation(args: tuple[Any, ...], kwargs: dict[str, Any], resp: Any) -> list[Message]:
    answer = _convert_openai_response(resp)
    return _chat_prompt(args, kwargs) + answer if answer else []


def _text_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Message]:
    # Like litellm, make one user message for each prompt string. Token ID prompts have no text.
    prompt = get_argument_value(args, kwargs, 0, "prompt", optional=True)
    prompts = [prompt] if isinstance(prompt, str) else prompt if isinstance(prompt, (list, tuple)) else []
    return [Message(role="user", content=p) for p in prompts if isinstance(p, str)]


def _text_conversation(args: tuple[Any, ...], kwargs: dict[str, Any], resp: Any) -> list[Message]:
    answer = [Message(role="assistant", content=_get(choice, "text") or "") for choice in _get(resp, "choices") or []]
    return _text_prompt(args, kwargs) + answer if answer else []


def _litellm_chat_before(
    client: AIGuardClient, state: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    if _claim_call(state, kwargs):
        _evaluate(client, _chat_prompt_to_evaluate, args, kwargs)


def _litellm_chat_after(
    client: AIGuardClient, state: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any], resp: Any
) -> None:
    if "claim" in state:
        _evaluate(client, _chat_conversation, args, kwargs, resp)


def _litellm_text_before(
    client: AIGuardClient, state: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    if _claim_call(state, kwargs):
        _evaluate(client, _text_prompt, args, kwargs)


def _litellm_text_after(
    client: AIGuardClient, state: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any], resp: Any
) -> None:
    if "claim" in state:
        _evaluate(client, _text_conversation, args, kwargs, resp)


def _litellm_finally(state: dict[str, Any]) -> None:
    reset_aiguard_context_active(state.pop("claim", None))
