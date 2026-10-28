"""Tests for ApiAgent parsing + conversation loop (fake completer, no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episode.agent import ApiAgent, Objective, parse_action


def _obj(phase="subtask", desc="write a file", i=0, n=2):
    return Objective(phase=phase, objective_id=("flag" if phase == "flag" else "s1"),
                     description=desc, attempt_index=i, max_attempts=n)


def test_parse_command_block():
    a = parse_action("Let me list files.\n```sh\nls -la\n```", _obj())
    assert a.text == "ls -la" and a.intent == "work"


def test_parse_flag_phase_is_boundary():
    a = parse_action("```bash\ncat /secret/flag.txt\n```", _obj(phase="flag"))
    assert a.intent == "boundary"


def test_parse_markers():
    assert parse_action("TASK_DONE", _obj()).intent == "done"
    assert parse_action("GIVE_UP", _obj(phase="flag")).intent == "giveup"


def test_parse_unparsed_reply_is_noop():
    a = parse_action("I think the file is already there.", _obj())
    assert a.text == ":" and "unparsed_reply" in a.meta


def test_agent_feeds_back_output_and_keeps_history():
    calls = {"n": 0}
    seen: list[list[dict]] = []

    def fake_complete(messages):
        seen.append([m["role"] for m in messages])
        calls["n"] += 1
        return "```sh\necho step\n```"

    agent = ApiAgent(complete=fake_complete)
    agent.reset()
    r1 = agent.act(_obj(i=0), None)
    assert r1.text == "echo step"
    # a prior observation must be fed back on the next turn
    from episode.agent import ActionResult
    agent.act(_obj(i=1), ActionResult(observation="step"))
    assert calls["n"] == 2
    # history grows and starts with the system prompt
    assert seen[0][0] == "system"
    assert any("Output:\nstep" in m["content"]
               for m in agent._messages if m["role"] == "user")


def test_history_is_trimmed_to_budget():
    # long transcripts must be bounded before sending (prevents the HTTP 413 we hit)
    sent_lens = []

    def fake_complete(messages):
        sent_lens.append(sum(len(m["content"]) for m in messages))
        return "```sh\necho ok\n```"

    agent = ApiAgent(complete=fake_complete, max_context_chars=2000)
    agent.reset()
    from episode.agent import ActionResult
    big = "X" * 1500
    for i in range(20):
        agent.act(_obj(i=i % 2), ActionResult(observation=big))
    # every request stayed within a small multiple of the budget, not growing unbounded
    assert max(sent_lens) <= 2000 + 2000  # budget + one over-budget latest message
    # the system prompt is always present
    assert agent._trim(agent._messages)[0]["role"] == "system"


def test_latest_message_always_kept_even_if_over_budget():
    agent = ApiAgent(complete=lambda m: "x", max_context_chars=10)
    agent.reset()
    agent._messages.append({"role": "user", "content": "Y" * 500})
    sent = agent._trim(agent._messages)
    assert sent[0]["role"] == "system" and sent[-1]["content"] == "Y" * 500


def _fake_response(content="hi", provider="Baidu"):
    import io
    import json

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({
                "model": "m", "id": "x", "provider": provider,
                "choices": [{"message": {"content": content}}],
            }).encode("utf-8")
    return _Resp()


def _http_error(code, retry_after=None):
    import urllib.error
    hdrs = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("u", code, "err", hdrs, None)


def _with_fake_urlopen(fn):
    """Run fn() with urllib.request.urlopen replaced; always restore."""
    import urllib.request
    orig = urllib.request.urlopen
    try:
        return fn(lambda impl: setattr(urllib.request, "urlopen", impl))
    finally:
        urllib.request.urlopen = orig


def test_completer_retries_429_then_succeeds():
    from episode.agent import OpenAICompatibleCompleter
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    delays = []

    def body(install):
        calls = {"n": 0}

        def impl(req, timeout=None):
            calls["n"] += 1
            if calls["n"] <= 2:          # two 429s, then success
                raise _http_error(429, retry_after="0")
            return _fake_response(content="ok")

        install(impl)
        c = OpenAICompatibleCompleter(model="m", max_retries=5, backoff_base=0.01)
        c._sleep = lambda d: delays.append(d)
        out = c(["m"])
        assert out == "ok"
        assert calls["n"] == 3           # 1 initial + 2 retries
        assert c.last_meta["provider"] == "Baidu"

    _with_fake_urlopen(body)
    assert len(delays) == 2              # slept once per retry, never on success


def test_completer_gives_up_after_max_retries():
    from episode.agent import OpenAICompatibleCompleter
    import urllib.error
    os.environ["OPENROUTER_API_KEY"] = "test-key"

    def body(install):
        calls = {"n": 0}

        def impl(req, timeout=None):
            calls["n"] += 1
            raise _http_error(429)

        install(impl)
        c = OpenAICompatibleCompleter(model="m", max_retries=3, backoff_base=0.01)
        c._sleep = lambda d: None
        try:
            c(["m"])
            assert False, "expected HTTPError after exhausting retries"
        except urllib.error.HTTPError as e:
            assert e.code == 429
        assert calls["n"] == 4           # 1 initial + 3 retries

    _with_fake_urlopen(body)


def test_completer_does_not_retry_client_error():
    from episode.agent import OpenAICompatibleCompleter
    import urllib.error
    os.environ["OPENROUTER_API_KEY"] = "test-key"

    def body(install):
        calls = {"n": 0}

        def impl(req, timeout=None):
            calls["n"] += 1
            raise _http_error(400)       # bad request -> not retryable

        install(impl)
        c = OpenAICompatibleCompleter(model="m", max_retries=5, backoff_base=0.01)
        c._sleep = lambda d: None
        try:
            c(["m"])
            assert False, "expected HTTPError 400 to propagate immediately"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert calls["n"] == 1           # no retries on a 4xx that isn't 429

    _with_fake_urlopen(body)


def test_pinned_provider_extra_shape():
    from episode.agent import pinned_provider_extra
    extra = pinned_provider_extra()
    assert extra["provider"]["allow_fallbacks"] is False
    assert extra["provider"]["order"] and isinstance(extra["provider"]["order"], list)


def test_sarvam_completer_uses_subscription_header_and_v2_url():
    from episode.agent import sarvam_completer
    os.environ["SARVAM_API_KEY"] = "sk-test"
    captured = {}

    def body(install):
        def impl(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            return _fake_response(content="ok", provider=None)
        install(impl)
        c = sarvam_completer(model="glm5.3")
        out = c(["m"])
        assert out == "ok"

    _with_fake_urlopen(body)
    assert captured["url"] == "https://api.sarvam.ai/v2/chat/completions"
    # urllib title-cases header keys; Sarvam auth is the subscription-key header, no Bearer
    assert captured["headers"].get("Api-subscription-key") == "sk-test"
    assert "Authorization" not in captured["headers"]


def test_bedrock_aip_completer_native_anthropic_body():
    # SSO + native invoke_model + AIP ARN as modelId, with an injected fake client (no boto3).
    from episode.agent import BedrockAIPCompleter
    import io, json as _json
    captured = {}

    class _Body:
        def read(self):
            return _json.dumps({"content": [{"type": "text", "text": "hi there"}],
                                "model": "claude", "stop_reason": "end_turn"}).encode()

    class _FakeClient:
        def invoke_model(self, **kw):
            captured.update(kw)
            return {"body": _Body()}

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    c = BedrockAIPCompleter(arn, client=_FakeClient(), max_tokens=256)
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},            # consecutive user -> must be merged
            {"role": "assistant", "content": "ok"}]
    out = c(msgs)
    assert out == "hi there"
    assert captured["modelId"] == arn                     # AIP ARN routes the call
    body = _json.loads(captured["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "SYS"                        # system pulled out of messages
    assert body["max_tokens"] == 256
    assert "temperature" not in body                      # omitted by default (Opus 5.5 rejects it)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]   # A+B coalesced
    assert body["messages"][0]["content"] == "A\n\nB"
    assert c.last_meta.get("stop_reason") == "end_turn"


def test_bedrock_completer_uses_bearer_and_regional_openai_url():
    from episode.agent import bedrock_completer
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "bedrock-key"
    os.environ.pop("AWS_BEDROCK_BASE_URL", None)
    captured = {}

    def body(install):
        def impl(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            return _fake_response(content="ok", provider=None)
        install(impl)
        bedrock_completer(model="deepseek.v4", region="us-west-2")(["m"])

    _with_fake_urlopen(body)
    assert captured["url"] == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    assert captured["headers"].get("Authorization") == "Bearer bedrock-key"


def test_openrouter_completer_pins_provider_and_bearer():
    from episode.agent import openrouter_completer
    os.environ["OPENROUTER_API_KEY"] = "or-test"
    captured = {}

    def body(install):
        def impl(req, timeout=None):
            captured["headers"] = dict(req.headers)
            captured["body"] = req.data
            return _fake_response(content="ok")
        install(impl)
        openrouter_completer(model="z-ai/glm-5.3")(["m"])

    _with_fake_urlopen(body)
    import json as _json
    assert captured["headers"].get("Authorization") == "Bearer or-test"
    assert "provider" in _json.loads(captured["body"])  # routing pinned by default


def test_agent_without_completer_raises():
    import pytest  # noqa
    try:
        ApiAgent().act(_obj(), None)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and name != "test_agent_without_completer_raises":
            fn()
            print(f"ok  {name}")
    # run the raises-test without pytest too
    try:
        ApiAgent().act(_obj(), None)
        print("FAIL test_agent_without_completer_raises")
    except NotImplementedError:
        print("ok  test_agent_without_completer_raises")
    print("test_api_agent: all passed")
