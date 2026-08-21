"""Caller allowlisting, session id derivation, and message shaping."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.speech import sanitize_spoken, speakable_reason, strip_placeholder_braces
from vapi_hermes_voice.vapi_events import CallVariables, ChatMessage

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_HISTORY_ROLES = frozenset({"user", "assistant"})

# Synthetic turn input when the request carries no trailing user utterance -- e.g.
# Vapi's `assistant-speaks-first-with-model-generated-message` first-message mode
# asks the model to open the call before the caller has said anything.
OPENING_NUDGE = (
    "The call has just started and the caller has not spoken yet. "
    "Greet them briefly and ask how you can help."
)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _ensure_period(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _render(template: str, values: dict[str, str]) -> str:
    """Substitute ``{name}`` placeholders, leaving unknown names as literal text.

    Deliberately not ``str.format``: the substituted values are untrusted, and this
    single pass neither re-scans a replacement nor honours a format spec, so a value
    containing ``{...}`` cannot reach back into the template or into Python objects.
    """
    return _PLACEHOLDER_RE.sub(lambda match: values.get(match.group(1), match.group(0)), template)


def has_trailing_user_message(messages: Sequence[ChatMessage]) -> bool:
    """True when the request carries a real user utterance for this turn to answer.

    False means the adapter must synthesize the turn itself (Vapi's
    ``assistant-speaks-first-with-model-generated-message`` mode). Nothing is pending
    on that path, so a latency filler there is noise -- see ``stream_turn``.
    """
    for message in reversed(messages):
        content = message.content
        if content is None or not content.strip():
            continue
        if message.role in _HISTORY_ROLES:
            return message.role == "user"
    return False


def is_first_callee_turn(messages: Sequence[ChatMessage]) -> bool:
    """True when this request carries the callee's FIRST utterance of the call.

    Two Vapi first-message modes produce two different histories at that moment, and
    both must be recognized:

    - ``assistant-waits-for-user`` (the current outbound setting): ``[system, user]``
      -- no assistant message at all, because the static ``firstMessage`` is inert.
    - ``assistant-speaks-first`` (legacy, and still how inbound may be configured):
      ``[system, assistant(firstMessage), user]`` -- Vapi echoes back the greeting it
      spoke itself, which is not a turn this adapter produced.

    So the test is on the user side: the trailing message is a user utterance and it
    is the only one in the conversation. At most one assistant message may precede
    it -- the static greeting in the legacy shape. A second one means a real reply
    has already happened, so this is not the opening any more.

    Deliberately keyed on the conversation rather than on adapter state: Vapi resends
    the full history on every request (docs/integration-contracts.md section 1.2), so
    this still answers correctly for a call whose per-call state was never registered
    (no ``call.id``) or was dropped by the TTL or the ``max_tracked_calls`` LRU.
    Every caller ANDs it with the ``CallState.reason_spoken`` latch, which covers the
    opposite hole -- a re-POST of the same turn whose reply has not yet made it back
    into the history Vapi sends. Both must agree, so a repeat needs both to fail; and
    when either is unsure the fast path simply does not fire and the turn takes
    today's Hermes path, which is a safe direction to be wrong in.
    """
    users = 0
    assistants = 0
    trailing_is_user = False
    for message in messages:
        content = message.content
        if content is None or not content.strip():
            continue
        if message.role == "user":
            users += 1
            trailing_is_user = True
        elif message.role == "assistant":
            assistants += 1
            trailing_is_user = False
    return trailing_is_user and users == 1 and assistants <= 1


def build_reason_line(
    settings: Settings,
    *,
    variables: CallVariables,
    callee_is_principal: bool = False,
) -> str | None:
    """The outbound reason-for-calling line, spoken from adapter-local text.

    Returns ``None`` when the call carries neither ``spoken_reason`` nor ``purpose``:
    there is nothing to announce, so the turn goes to Hermes exactly as before.

    ``purpose`` is a TRIGGER here and never speech. Its presence marks the call as an
    operator-placed task call, but not one character of it is spoken: it is
    model-facing instruction prose with no fixed grammar ("Goal: next steps -
    appointment, phone call with Craig, or proceed to surgery and get a date. Mike is
    free weekday mornings."), and the failure mode of quoting it is the assistant
    reading the principal's internal negotiating limits aloud to a doctor's office.
    The spoken reason comes only from ``spoken_reason``, the field an operator fills
    in *because* it is meant to be heard -- and even that goes through
    :func:`vapi_hermes_voice.speech.speakable_reason`, which strips labels, lists and
    instruction phrasing and refuses outright when what survives still reads like an
    instruction. With no usable spoken reason the line falls back to
    ``settings.outbound_reason_sentence_generic``, which names neither the purpose nor
    anything derived from it, so reciting operator text is impossible by construction
    rather than merely unlikely.

    The greeting -- and with it the AI-identity disclosure owed to a third party -- is
    assembled here and is deliberately NOT configurable. It is the safety content of
    the line, and since Vapi's static ``firstMessage`` is inert on outbound calls this
    is the only place the callee is told they are talking to a machine. No template an
    operator can edit sits between that sentence and the wire.
    """
    if not (variables.spoken_reason or variables.purpose):
        return None
    values = {
        "principal": settings.principal,
        "assistant_name": settings.assistant_name,
        "callee": variables.callee or "",
    }
    reason = speakable_reason(variables.spoken_reason) if variables.spoken_reason else None
    if reason is None:
        sentence = _render(settings.outbound_reason_sentence_generic, values)
    else:
        sentence = _render(settings.outbound_reason_sentence, {**values, "reason": reason})
    if callee_is_principal:
        # Nobody to disclose to but the principal, and "calling on behalf of Mike" is
        # nonsense framing when Mike is the one who picked up.
        greeting = f"Hi {settings.principal}, {settings.assistant_name} here."
    elif settings.outbound_disclose_ai:
        greeting = (
            f"Hi, this is {settings.assistant_name}, an AI assistant calling on behalf"
            f" of {settings.principal}."
        )
    else:
        greeting = (
            f"Hi, this is {settings.assistant_name}, calling on behalf of {settings.principal}."
        )
    return sanitize_spoken(strip_placeholder_braces(f"{greeting} {sentence}"))


def build_opening_nudge(
    settings: Settings,
    *,
    direction: str,
    variables: CallVariables,
    callee_is_principal: bool = False,
) -> str:
    """Turn input for a request that carries no trailing user utterance.

    Inbound calls -- and outbound calls with no ``purpose`` -- get the unchanged
    :data:`OPENING_NUDGE`. An outbound call carrying a purpose gets one of the two
    outbound templates instead, so the assistant opens by stating the reason for the
    call rather than greeting somebody who is not on the line:

    - a third party (the default) gets ``settings.outbound_opening``: who is calling,
      on whose behalf, the reason, plus the AI-identity disclosure;
    - the principal themselves gets ``settings.outbound_opening_principal``: greeted
      directly by name, with no on-behalf-of framing and no disclosure, since
      "this is Emma calling for Mike" is wrong when Mike is the one who answered.

    This only ever applies in Vapi's model-generated first-message mode. A static
    ``assistant.firstMessage`` is spoken by Vapi itself and never reaches the adapter,
    so a configured static greeting always wins over these templates.

    The callee sentence and the AI-identity disclosure are appended here rather than
    embedded in a template: disclosure is a safety control, and editing a template
    must not be able to drop it silently.
    """
    if direction != "outbound" or not variables.purpose:
        return OPENING_NUDGE
    template = (
        settings.outbound_opening_principal if callee_is_principal else settings.outbound_opening
    )
    parts = [
        _render(
            template,
            {
                "purpose": _ensure_period(variables.purpose),
                "callee": variables.callee or "",
                "principal": settings.principal,
                "assistant_name": settings.assistant_name,
            },
        ).strip()
    ]
    if callee_is_principal:
        # Nobody to ask for and nobody to disclose to but the principal.
        return parts[0]
    if variables.callee:
        parts.append(
            f"You are trying to reach {_ensure_period(variables.callee)}"
            " Ask for the right person if somebody else answers."
        )
    if settings.outbound_disclose_ai:
        parts.append(
            "Say plainly, in your opening, that you are an AI assistant calling for"
            f" {settings.principal}."
        )
    return " ".join(part for part in parts if part)


class CallerPolicy:
    """Deny-by-default caller allowlist; disabled (allow all) when the list is empty."""

    def __init__(self, allowed: Sequence[str]) -> None:
        for number in allowed:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowlist entries must be E.164 numbers like +15551234567")
        self._allowed: frozenset[str] = frozenset(allowed)

    @property
    def enforced(self) -> bool:
        return bool(self._allowed)

    def is_allowed(self, from_number: str | None) -> bool:
        if not self.enforced:
            return True
        if from_number is None or _E164_RE.fullmatch(from_number) is None:
            return False
        return from_number in self._allowed


def derive_session_ids(call_id: str) -> tuple[str, str]:
    """(session_id, session_key) derived from the Vapi call id, never from phone numbers."""
    session_id = "vhv-" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
    session_key = "vhv-key-" + hashlib.sha256(("k" + call_id).encode("utf-8")).hexdigest()[:24]
    return session_id, session_key


def truncate_history(messages: list[ChatMessage], max_len: int) -> list[ChatMessage]:
    """Keep the most recent messages; always keep at least the last one."""
    if not messages:
        return []
    return messages[-max(max_len, 1) :]


def split_messages(
    messages: list[ChatMessage], *, opening: str = OPENING_NUDGE
) -> tuple[list[dict[str, str]], str, str]:
    """OpenAI messages -> (hermes_history, user_input, extra_instructions).

    System messages (the Vapi assistant's own prompt) are collected into
    ``extra_instructions`` -- they layer onto the adapter's voice instructions, which
    in turn layer onto Hermes's resident system prompt. ``tool``/``function`` frames
    are dropped (Hermes never sees Vapi-side tool plumbing). The trailing user
    message becomes the turn input; without one, ``opening`` is used. ``opening``
    defaults to the inbound :data:`OPENING_NUDGE`; callers that know the call
    direction pass :func:`build_opening_nudge` instead.
    """
    extra: list[str] = []
    history: list[dict[str, str]] = []
    for message in messages:
        content = message.content
        if content is None or not content.strip():
            continue
        if message.role == "system":
            extra.append(content)
        elif message.role in _HISTORY_ROLES:
            history.append({"role": message.role, "content": content})
    user_input = opening
    if history and history[-1]["role"] == "user":
        user_input = history[-1]["content"]
        history = history[:-1]
    return history, user_input, "\n\n".join(extra)
