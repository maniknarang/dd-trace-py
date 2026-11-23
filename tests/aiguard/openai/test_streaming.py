"""Streaming response-evaluation AI Guard tests for the OpenAI SDK.

All "buffered" cases run with ``DD_AI_GUARD_ANALYZE_STREAM_RESPONSES_ENABLED=true``
(via the *_buffered fixtures). Flag-off cases assert request inputs are evaluated and
the stream is not buffered.
"""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from ddtrace.aiguard import AIGuardAbortError
from ddtrace.aiguard import AIGuardClientError
from ddtrace.aiguard._listener import _make_openai_stream_wrappers
from ddtrace.aiguard._streaming import BufferedAIGuardAsyncStream
from ddtrace.aiguard._streaming import BufferedAIGuardStream
from tests.aiguard.utils import mock_evaluate_response
from tests.aiguard.utils import override_ai_guard_config


CHAT_MODEL = "gpt-3.5-turbo"
RESPONSES_MODEL = "gpt-4o-mini"


def _user_messages(content="Hello"):
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Chat Completions — sync
# ---------------------------------------------------------------------------


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_sync_buffered_allow(mock_execute_request, openai_client_stream_buffered):
    """ALLOW: create(stream=True) returns a BufferedAIGuardStream; chunks replay after 2 evals."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    assert isinstance(stream, BufferedAIGuardStream)
    chunks = list(stream)

    assert chunks  # buffered chunks were replayed
    assert mock_execute_request.call_count == 2  # before-hook + buffered response


@pytest.mark.parametrize("decision", ["DENY", "ABORT"], ids=["deny", "abort"])
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_sync_buffered_block(mock_execute_request, openai_client_stream_buffered, decision):
    """Block on the response: AIGuardAbortError raised, ZERO chunks delivered."""
    mock_execute_request.side_effect = [mock_evaluate_response("ALLOW"), mock_evaluate_response(decision)]

    stream = openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    seen = []
    with pytest.raises(AIGuardAbortError):
        for chunk in stream:
            seen.append(chunk)

    assert seen == []  # security invariant: nothing reaches the caller before ALLOW
    assert mock_execute_request.call_count == 2


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_sync_read_after_caught_block(mock_execute_request, openai_client_stream_buffered):
    """Iterating a blocked stream again raises the block instead of yielding nothing, without a new evaluation."""
    mock_execute_request.side_effect = [
        mock_evaluate_response("ALLOW"),
        mock_evaluate_response("DENY"),
        mock_evaluate_response("ALLOW"),
    ]

    stream = openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    with pytest.raises(AIGuardAbortError):
        list(stream)
    with pytest.raises(AIGuardAbortError):
        list(stream)

    assert mock_execute_request.call_count == 2


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_sync_block_disabled_replays(mock_execute_request, openai_client_stream_buffered):
    """``_ai_guard_block=False`` + DENY: no raise, chunks replayed."""
    mock_execute_request.side_effect = [mock_evaluate_response("ALLOW"), mock_evaluate_response("DENY", block=True)]

    with override_ai_guard_config(dict(_ai_guard_block=False)):
        stream = openai_client_stream_buffered.chat.completions.create(
            model=CHAT_MODEL, messages=_user_messages(), stream=True
        )
        chunks = list(stream)

    assert chunks
    assert mock_execute_request.call_count == 2


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_collision_passthrough(mock_execute_request, openai_client_stream_buffered, aiguard_active_context):
    """Inside an active framework context: live passthrough, no evaluation."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    list(stream)

    mock_execute_request.assert_not_called()


@patch("ddtrace.aiguard.integrations._openai_chat_streaming.openai_construct_message_from_streamed_chunks")
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_reconstruction_failure_fails_open(
    mock_execute_request, mock_reconstruct, openai_client_stream_buffered
):
    """A reconstruction error must NOT break the stream: chunks replay, response eval skipped."""
    mock_reconstruct.side_effect = RuntimeError("boom")
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    chunks = list(stream)

    assert chunks  # replayed despite reconstruction failure
    assert mock_execute_request.call_count == 1  # only the before-hook ran


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_flag_off_not_buffered(mock_execute_request, openai_client_stream):
    """Flag off: create(stream=True) is NOT a BufferedAIGuardStream (today's behavior)."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_client_stream.chat.completions.create(model=CHAT_MODEL, messages=_user_messages(), stream=True)
    assert not isinstance(stream, BufferedAIGuardStream)
    list(stream)

    assert mock_execute_request.call_count == 1  # before-hook only


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_chat_stream_tool_calls_are_evaluated(mock_execute_request, openai_client_stream_tool_calls_buffered):
    """A streamed tool_call must survive reconstruction and be evaluated, not silently dropped."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_client_stream_tool_calls_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    list(stream)

    assert mock_execute_request.call_count == 2  # before-hook + buffered response
    # The response evaluation (2nd call) must carry the reconstructed tool call.
    response_messages = mock_execute_request.call_args_list[1].args[1]["data"]["attributes"]["messages"]
    tool_calls = [tc for msg in response_messages for tc in (msg.get("tool_calls") or [])]
    assert [tc["function"]["name"] for tc in tool_calls] == ["get_weather"]
    assert tool_calls[0]["function"]["arguments"] == '{"city": "Paris"}'


# ---------------------------------------------------------------------------
# Chat Completions — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_chat_stream_async_buffered_allow(mock_execute_request, async_openai_client_stream_buffered):
    """Async ALLOW: AsyncCompletions.create (async_wrapper path) returns BufferedAIGuardAsyncStream."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = await async_openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    assert isinstance(stream, BufferedAIGuardAsyncStream)
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert mock_execute_request.call_count == 2


@pytest.mark.asyncio
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_chat_stream_async_buffered_block(mock_execute_request, async_openai_client_stream_buffered):
    """Async block: AIGuardAbortError raised, zero chunks delivered."""
    mock_execute_request.side_effect = [mock_evaluate_response("ALLOW"), mock_evaluate_response("DENY")]

    stream = await async_openai_client_stream_buffered.chat.completions.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    seen = []
    with pytest.raises(AIGuardAbortError):
        async for chunk in stream:
            seen.append(chunk)

    assert seen == []
    assert mock_execute_request.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["ALLOW", "DENY"], ids=["allow", "deny"])
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_chat_stream_async_buffered_evaluates_off_event_loop(
    mock_execute_request, async_openai_client_stream_buffered, decision
):
    """A slow buffered-response evaluation must not stall other asyncio tasks; a block still propagates."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    ticks_during_response_eval = []

    def execute_request(*args, **kwargs):
        if mock_execute_request.call_count == 1:  # before-hook request evaluation
            return mock_evaluate_response("ALLOW")
        start = ticks
        time.sleep(0.3)
        ticks_during_response_eval.append(ticks - start)
        return mock_evaluate_response(decision)

    mock_execute_request.side_effect = execute_request
    ticker_task = asyncio.create_task(ticker())
    try:
        stream = await async_openai_client_stream_buffered.chat.completions.create(
            model=CHAT_MODEL, messages=_user_messages(), stream=True
        )
        seen = []
        if decision == "ALLOW":
            seen = [chunk async for chunk in stream]
            assert seen
        else:
            with pytest.raises(AIGuardAbortError):
                async for chunk in stream:
                    seen.append(chunk)
            assert seen == []
    finally:
        ticker_task.cancel()
        await asyncio.gather(ticker_task, return_exceptions=True)

    assert mock_execute_request.call_count == 2
    # The ticker kept running while the blocking evaluation slept in its worker thread.
    assert ticks_during_response_eval and ticks_during_response_eval[0] > 1


def _run_without_event_loop(coro):
    """Drive a coroutine that never suspends, with no asyncio loop running (as under trio)."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine suspended")


@pytest.mark.parametrize("decision", ["ALLOW", "DENY"], ids=["allow", "deny"])
def test_async_buffered_stream_without_asyncio_loop_evaluates_inline(decision):
    """Without a running asyncio loop the evaluation runs inline: allow replays, block raises."""

    async def source():
        for chunk in ("a", "b"):
            yield chunk

    def evaluate(response):
        assert response == ["a", "b"]
        if decision == "DENY":
            raise AIGuardAbortError(action="DENY", reason="blocked")

    async def consume(stream):
        return [chunk async for chunk in stream]

    stream = BufferedAIGuardAsyncStream(source(), reconstruct=list, evaluate=evaluate)
    with override_ai_guard_config(dict(_ai_guard_analyze_stream_responses_enabled=True)):
        if decision == "ALLOW":
            assert _run_without_event_loop(consume(stream)) == ["a", "b"]
        else:
            with pytest.raises(AIGuardAbortError):
                _run_without_event_loop(consume(stream))


class _FinalMessageStream:
    """Async stream with Anthropic-style accessors that return the full response."""

    def __init__(self):
        self._chunks = iter(["a", "b"])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    @property
    def text_stream(self):
        async def text():
            yield "full text"

        return text()

    async def get_final_message(self):
        return "full message"

    async def get_final_text(self):
        return "full text"

    async def until_done(self):
        return None

    @property
    def current_message_snapshot(self):
        return "full message"


class _SyncFinalMessageStream:
    """Sync twin of _FinalMessageStream."""

    def __init__(self):
        self._chunks = iter(["a", "b"])

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    @property
    def text_stream(self):
        return iter(["full text"])

    def get_final_message(self):
        return "full message"

    def get_final_text(self):
        return "full text"

    def until_done(self):
        return None

    @property
    def current_message_snapshot(self):
        return "full message"


@pytest.mark.asyncio
async def test_async_buffered_stream_concurrent_and_later_callers_share_the_verdict():
    """A caller arriving during or after a blocking evaluation must not read the exhausted stream unevaluated."""
    evaluated = []

    def evaluate(response):
        evaluated.append(response)
        time.sleep(0.2)
        # Mirrors the listeners, which skip an empty response.
        if response:
            raise AIGuardAbortError(action="DENY", reason="blocked")

    async def consume(stream):
        return [chunk async for chunk in stream]

    async def final_message_during_evaluation(stream):
        await asyncio.sleep(0.05)
        return await stream.get_final_message()

    stream = BufferedAIGuardAsyncStream(_FinalMessageStream(), reconstruct=list, evaluate=evaluate)
    with override_ai_guard_config(dict(_ai_guard_analyze_stream_responses_enabled=True)):
        results = await asyncio.gather(consume(stream), final_message_during_evaluation(stream), return_exceptions=True)
        assert [type(result) for result in results] == [AIGuardAbortError, AIGuardAbortError]
        with pytest.raises(AIGuardAbortError):
            await stream.get_final_message()

    assert evaluated and all(response == ["a", "b"] for response in evaluated)


_SYNC_READS = {
    "iterate": lambda stream: list(stream),
    "next": lambda stream: next(stream, None),
    "text_stream": lambda stream: list(stream.text_stream),
    "get_final_message": lambda stream: stream.get_final_message(),
    "get_final_text": lambda stream: stream.get_final_text(),
    "until_done": lambda stream: stream.until_done(),
    "current_message_snapshot": lambda stream: stream.current_message_snapshot,
}


async def _async_iterate(stream):
    return [chunk async for chunk in stream]


async def _async_next(stream):
    """Like next(stream, None) for an async stream."""
    try:
        return await stream.__anext__()
    except StopAsyncIteration:
        return None


async def _async_text_stream(stream):
    return [text async for text in stream.text_stream]


async def _async_snapshot(stream):
    return stream.current_message_snapshot


_ASYNC_READS = {
    "iterate": _async_iterate,
    "next": _async_next,
    "text_stream": _async_text_stream,
    "get_final_message": lambda stream: stream.get_final_message(),
    "get_final_text": lambda stream: stream.get_final_text(),
    "until_done": lambda stream: stream.until_done(),
    "current_message_snapshot": _async_snapshot,
}

# After the blocking verdict: a later verdict allows, AI Guard fails, or stream analysis is turned off.
_LATER = ["allow", "client_error", "analysis_off"]


def _block_once(later):
    """Return the recorded evaluations and a stub that blocks on its first call."""
    evaluated = []

    def evaluate(response):
        evaluated.append(response)
        if len(evaluated) == 1:
            raise AIGuardAbortError(action="DENY", reason="blocked")
        if later == "client_error":
            raise AIGuardClientError("AI Guard is unavailable")

    return evaluated, evaluate


@pytest.mark.parametrize("later", _LATER)
@pytest.mark.parametrize("read", list(_SYNC_READS))
def test_buffered_stream_read_after_block(read, later):
    """Every read after a block raises it again, and nothing evaluates the response again to release it."""
    evaluated, evaluate = _block_once(later)
    analysis = {"enabled": True}
    stream = BufferedAIGuardStream(
        _SyncFinalMessageStream(), reconstruct=list, evaluate=evaluate, enabled=lambda: analysis["enabled"]
    )

    with pytest.raises(AIGuardAbortError) as blocked:
        list(stream)
    analysis["enabled"] = later != "analysis_off"
    with pytest.raises(AIGuardAbortError) as again:
        _SYNC_READS[read](stream)

    assert type(again.value) is type(blocked.value)
    assert evaluated == [["a", "b"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("later", _LATER)
@pytest.mark.parametrize("read", list(_ASYNC_READS))
async def test_async_buffered_stream_read_after_block(read, later):
    """Async twin: every read after a block raises it again, without a new evaluation."""
    evaluated, evaluate = _block_once(later)
    analysis = {"enabled": True}
    stream = BufferedAIGuardAsyncStream(
        _FinalMessageStream(), reconstruct=list, evaluate=evaluate, enabled=lambda: analysis["enabled"]
    )

    with pytest.raises(AIGuardAbortError) as blocked:
        await _async_iterate(stream)
    analysis["enabled"] = later != "analysis_off"
    with pytest.raises(AIGuardAbortError) as again:
        await _ASYNC_READS[read](stream)

    assert type(again.value) is type(blocked.value)
    assert evaluated == [["a", "b"]]


class _NoVerdict(BaseException):
    """Ends an evaluation without a verdict, like a timeout raised inside the AI Guard call."""


@pytest.mark.parametrize("later", ["unchanged", "analysis_off", "claim"])
def test_buffered_stream_evaluates_the_buffer_again_without_verdict(later, request):
    """An evaluation that ends without a verdict keeps the drained buffer, and the next read evaluates it."""
    evaluated = []

    def evaluate(response):
        evaluated.append(response)
        if len(evaluated) == 1:
            raise _NoVerdict()

    analysis = {"enabled": True}
    stream = BufferedAIGuardStream(
        _SyncFinalMessageStream(), reconstruct=list, evaluate=evaluate, enabled=lambda: analysis["enabled"]
    )

    with pytest.raises(_NoVerdict):
        list(stream)
    # Pass-through is decided on first use, so turning analysis off or claiming the call skips nothing now.
    analysis["enabled"] = later != "analysis_off"
    if later == "claim":
        request.getfixturevalue("aiguard_active_context")
    assert list(stream) == ["a", "b"]
    assert evaluated == [["a", "b"], ["a", "b"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["allow", "deny"])
async def test_async_buffered_stream_cancelled_reader_shares_the_evaluation(verdict):
    """A reader cancelled during the evaluation leaves it running: the next reader gets its verdict, not a new one."""
    evaluated = []
    started, release = threading.Event(), threading.Event()

    def evaluate(response):
        evaluated.append(response)
        first = len(evaluated) == 1
        if first:
            started.set()
            release.wait(5)
        # The first evaluation gives the verdict under test; a second one would give the other verdict.
        if first == (verdict == "deny"):
            raise AIGuardAbortError(action="DENY", reason="blocked")

    stream = BufferedAIGuardAsyncStream(
        _FinalMessageStream(), reconstruct=list, evaluate=evaluate, enabled=lambda: True
    )
    reading = asyncio.ensure_future(stream.get_final_message())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        later = asyncio.ensure_future(stream.get_final_message())
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError):
            _ = stream.current_message_snapshot
    finally:
        release.set()

    if verdict == "deny":
        with pytest.raises(AIGuardAbortError):
            await later
        with pytest.raises(AIGuardAbortError):
            _ = stream.current_message_snapshot
    else:
        assert await later == "full message"
        assert stream.current_message_snapshot == "full message"
    assert evaluated == [["a", "b"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["analysis_off", "claim"])
async def test_async_buffered_stream_snapshot_first_passthrough(mode, request):
    """A pass-through stream returns the SDK snapshot even when the snapshot is its first read."""
    evaluated = []
    if mode == "claim":
        request.getfixturevalue("aiguard_active_context")
    stream = BufferedAIGuardAsyncStream(
        _FinalMessageStream(), reconstruct=list, evaluate=evaluated.append, enabled=lambda: mode != "analysis_off"
    )

    assert stream.current_message_snapshot == "full message"
    assert await _async_iterate(stream) == ["a", "b"]
    assert evaluated == []


@pytest.mark.asyncio
async def test_async_buffered_stream_snapshot_while_evaluation_pending():
    """Another task cannot read the snapshot while the evaluation runs; it is returned once the response is allowed."""
    started, release = threading.Event(), threading.Event()

    def evaluate(response):
        started.set()
        release.wait(5)

    stream = BufferedAIGuardAsyncStream(
        _FinalMessageStream(), reconstruct=list, evaluate=evaluate, enabled=lambda: True
    )
    reading = asyncio.ensure_future(stream.get_final_message())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(RuntimeError):
            _ = stream.current_message_snapshot
    finally:
        release.set()

    assert await reading == "full message"
    assert stream.current_message_snapshot == "full message"


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_stream_sync_buffered_allow(mock_execute_request, openai_responses_stream_client_buffered):
    """ALLOW: responses.create(stream=True) buffers; request eval + response eval."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_responses_stream_client_buffered.responses.create(model=RESPONSES_MODEL, input="Hello", stream=True)
    assert isinstance(stream, BufferedAIGuardStream)
    events = list(stream)

    assert events
    assert mock_execute_request.call_count == 2


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_stream_sync_buffered_block(mock_execute_request, openai_responses_stream_client_buffered):
    """Block on response snapshot: AIGuardAbortError raised, zero events delivered."""
    mock_execute_request.side_effect = [mock_evaluate_response("ALLOW"), mock_evaluate_response("DENY")]

    stream = openai_responses_stream_client_buffered.responses.create(model=RESPONSES_MODEL, input="Hello", stream=True)
    seen = []
    with pytest.raises(AIGuardAbortError):
        for event in stream:
            seen.append(event)

    assert seen == []
    assert mock_execute_request.call_count == 2


@pytest.mark.asyncio
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_responses_stream_async_buffered_allow(
    mock_execute_request, async_openai_responses_stream_client_buffered
):
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = await async_openai_responses_stream_client_buffered.responses.create(
        model=RESPONSES_MODEL, input="Hello", stream=True
    )
    assert isinstance(stream, BufferedAIGuardAsyncStream)
    events = [event async for event in stream]

    assert events
    assert mock_execute_request.call_count == 2


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_stream_flag_off_not_buffered(mock_execute_request, openai_responses_stream_client):
    """Flag off: streaming Responses are not buffered; only the request inputs are evaluated."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_responses_stream_client.responses.create(model=RESPONSES_MODEL, input="Hello", stream=True)
    assert not isinstance(stream, BufferedAIGuardStream)
    list(stream)

    mock_execute_request.assert_called_once()


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_stream_without_mcp_tools_unchanged_by_default_mcp(
    mock_execute_request, openai_responses_stream_client
):
    """DD_AI_GUARD_MCP_ENABLED is on by default: a stream with only function tools is still unbuffered."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = openai_responses_stream_client.responses.create(
        model=RESPONSES_MODEL, input="Hello", tools=[{"type": "function", "name": "lookup"}], stream=True
    )
    assert not isinstance(stream, BufferedAIGuardStream)
    assert list(stream)

    mock_execute_request.assert_called_once()
    assert set(mock_execute_request.call_args.args[1]["data"]["attributes"]) == {"messages", "meta"}


@pytest.mark.asyncio
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_responses_stream_async_flag_off_evaluates_input(
    mock_execute_request, async_openai_responses_stream_client
):
    """Flag off: async streaming Responses are not buffered; only the request inputs are evaluated."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    stream = await async_openai_responses_stream_client.responses.create(
        model=RESPONSES_MODEL, input="Hello", stream=True
    )
    assert not isinstance(stream, BufferedAIGuardAsyncStream)
    assert [event async for event in stream]

    mock_execute_request.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["DENY", "ABORT"], ids=["deny", "abort"])
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_responses_stream_async_flag_off_blocks_on_deny(
    mock_execute_request, async_openai_responses_stream_client, decision
):
    """Flag off: a blocking verdict on async streaming request inputs raises before the stream is returned."""
    import openai

    mock_execute_request.return_value = mock_evaluate_response(decision)

    with pytest.raises(openai.UnprocessableEntityError) as exc_info:
        await async_openai_responses_stream_client.responses.create(model=RESPONSES_MODEL, input="Hello", stream=True)

    assert isinstance(exc_info.value, AIGuardAbortError)
    mock_execute_request.assert_called_once()


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_raw_response_stream_flag_off_evaluates_input(mock_execute_request, openai_responses_stream_client):
    """Flag off: with_raw_response.create(stream=True) evaluates the request inputs once, unbuffered."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    raw = openai_responses_stream_client.responses.with_raw_response.create(
        model=RESPONSES_MODEL, input="Hello", stream=True
    )
    stream = raw.parse()
    assert not isinstance(stream, BufferedAIGuardStream)
    assert list(stream)

    mock_execute_request.assert_called_once()


@pytest.mark.parametrize("decision", ["DENY", "ABORT"], ids=["deny", "abort"])
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_responses_raw_response_stream_flag_off_blocks_on_deny(
    mock_execute_request, openai_responses_stream_client, decision
):
    """Flag off: a blocking verdict on with_raw_response.create(stream=True) inputs raises AIGuardAbortError."""
    import openai

    mock_execute_request.return_value = mock_evaluate_response(decision)

    with pytest.raises(openai.UnprocessableEntityError) as exc_info:
        openai_responses_stream_client.responses.with_raw_response.create(
            model=RESPONSES_MODEL, input="Hello", stream=True
        )

    assert isinstance(exc_info.value, AIGuardAbortError)
    mock_execute_request.assert_called_once()


# ---------------------------------------------------------------------------
# Raw response helper (chat.completions.with_raw_response)
# ---------------------------------------------------------------------------


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_raw_response_stream_sync_buffered_allow(mock_execute_request, openai_client_stream_buffered):
    """ALLOW: with_raw_response.create(stream=True).parse() yields a buffered stream."""
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    raw = openai_client_stream_buffered.chat.completions.with_raw_response.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    stream = raw.parse()
    assert isinstance(stream, BufferedAIGuardStream)
    chunks = list(stream)

    assert chunks
    assert mock_execute_request.call_count == 2  # contrib raw .before + buffered response


@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
def test_raw_response_stream_sync_buffered_block(mock_execute_request, openai_client_stream_buffered):
    """Block on raw-response stream: AIGuardAbortError, zero chunks delivered."""
    mock_execute_request.side_effect = [mock_evaluate_response("ALLOW"), mock_evaluate_response("DENY")]

    raw = openai_client_stream_buffered.chat.completions.with_raw_response.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    stream = raw.parse()
    seen = []
    with pytest.raises(AIGuardAbortError):
        for chunk in stream:
            seen.append(chunk)

    assert seen == []
    assert mock_execute_request.call_count == 2


@pytest.mark.asyncio
@patch("ddtrace.aiguard._api_client.AIGuardClient._execute_request")
async def test_raw_response_stream_async_buffered_allow(mock_execute_request, async_openai_client_stream_buffered):
    mock_execute_request.return_value = mock_evaluate_response("ALLOW")

    raw = await async_openai_client_stream_buffered.chat.completions.with_raw_response.create(
        model=CHAT_MODEL, messages=_user_messages(), stream=True
    )
    stream = raw.parse()
    if hasattr(stream, "__await__"):
        stream = await stream
    assert isinstance(stream, BufferedAIGuardAsyncStream)
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert mock_execute_request.call_count == 2


# ---------------------------------------------------------------------------
# Plain-generator streams (OpenAI SDK <1.6)
#
# SDKs older than 1.6 never produce a TracedStream — the contrib returns a raw
# (async) generator. These must still be buffered+evaluated, not forwarded live.
# Tested by driving ``_make_openai_stream_wrappers`` directly so the case is
# covered regardless of the installed SDK version in the test matrix.
# ---------------------------------------------------------------------------


def _record_eval():
    evaluated = []

    def reconstruct(chunks):
        return {"reconstructed": list(chunks)}

    def evaluate_after(client, kwargs, resp):
        evaluated.append(resp)

    return evaluated, reconstruct, evaluate_after


def test_plain_generator_sync_buffered():
    """A sync plain generator (SDK <1.6) is wrapped, evaluated once, then replayed."""
    evaluated, reconstruct, evaluate_after = _record_eval()
    sync_wrapper, _ = _make_openai_stream_wrappers(object(), reconstruct, evaluate_after)

    fake_chunks = ["a", "b", "c"]

    def func(*args, **kwargs):
        def gen():
            yield from fake_chunks

        return gen()

    with override_ai_guard_config(dict(_ai_guard_analyze_stream_responses_enabled=True)):
        result = sync_wrapper(func, None, (), {"stream": True})
        assert isinstance(result, BufferedAIGuardStream)
        replayed = list(result)  # drains, evaluates, then replays
    assert replayed == fake_chunks
    assert evaluated == [{"reconstructed": fake_chunks}]  # evaluated before replay


def test_plain_generator_sync_dispatches_async_for_async_generator():
    """A sync method returning an async generator dispatches to the async buffer."""
    _, reconstruct, evaluate_after = _record_eval()
    sync_wrapper, _ = _make_openai_stream_wrappers(object(), reconstruct, evaluate_after)

    async def agen():
        yield "z"

    result = sync_wrapper(lambda *a, **k: agen(), None, (), {"stream": True})
    assert isinstance(result, BufferedAIGuardAsyncStream)


@pytest.mark.asyncio
async def test_plain_generator_async_buffered():
    """An async plain generator (SDK <1.6) is wrapped, evaluated once, then replayed."""
    evaluated, reconstruct, evaluate_after = _record_eval()
    _, async_wrapper = _make_openai_stream_wrappers(object(), reconstruct, evaluate_after)

    fake_chunks = ["x", "y"]

    async def func(*args, **kwargs):
        async def agen():
            for chunk in fake_chunks:
                yield chunk

        return agen()

    with override_ai_guard_config(dict(_ai_guard_analyze_stream_responses_enabled=True)):
        result = await async_wrapper(func, None, (), {"stream": True})
        assert isinstance(result, BufferedAIGuardAsyncStream)
        collected = [chunk async for chunk in result]
    assert collected == fake_chunks
    assert evaluated == [{"reconstructed": fake_chunks}]


def test_non_stream_result_not_wrapped():
    """A non-stream return value (not a TracedStream, not a generator) passes through unwrapped."""
    _, reconstruct, evaluate_after = _record_eval()
    sync_wrapper, _ = _make_openai_stream_wrappers(object(), reconstruct, evaluate_after)

    sentinel = object()
    result = sync_wrapper(lambda *a, **k: sentinel, None, (), {})
    assert result is sentinel


class _Manager:
    """A stream whose context manager yields a new inner stream, like MessageStreamManager."""

    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        return self._inner

    def __exit__(self, *exc):
        return None

    async def __aenter__(self):
        return self._inner

    async def __aexit__(self, *exc):
        return None


def _enabled():
    return True


def test_manager_path_keeps_the_enabled_check():
    proxy = BufferedAIGuardStream(_Manager(iter([])), reconstruct=list, evaluate=lambda resp: None, enabled=_enabled)
    with proxy as inner:
        assert isinstance(inner, BufferedAIGuardStream)
        assert inner._self_enabled is _enabled


@pytest.mark.asyncio
async def test_async_manager_path_keeps_the_enabled_check():
    proxy = BufferedAIGuardAsyncStream(
        _Manager(iter([])), reconstruct=list, evaluate=lambda resp: None, enabled=_enabled
    )
    async with proxy as inner:
        assert isinstance(inner, BufferedAIGuardAsyncStream)
        assert inner._self_enabled is _enabled
