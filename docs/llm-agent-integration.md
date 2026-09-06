# LLM Agent Integration

The LLM owner provides `LLMRoomAgent` and `OllamaChatClient`. The evaluator
owner drives it through the agent protocol in
[`evaluator-build-spec.md`](evaluator-build-spec.md); the LLM package does not
run, score, or aggregate experiments itself.

## Canonical construction

```python
from light_learning.llm_agent import LLMRoomAgent, LLMRoomAgentConfig
from light_learning.ollama import OllamaChatClient

client = OllamaChatClient("http://127.0.0.1:11434")
agent = LLMRoomAgent(
    client,
    LLMRoomAgentConfig(
        model="qwen3:1.7b",
        condition="qualitative",  # or "disclosed"
        thinking_enabled=True,
        temperature=0.6,
        top_p=0.95,
        context_tokens=4096,
        max_output_tokens=256,
        seed=0,
        schema_retries=2,
    ),
)
```

The production client uses native Ollama `/api/chat` with `stream=false`,
`think=true`, a JSON schema in `format`, and generation options. It never
pulls a model. `run_metadata()` captures the installed model digest, Ollama
show metadata, and the exact `system_prompt` plus its `system_prompt_sha256`,
so the evaluator's manifest records which prompt wording produced a run.

## Information boundary

For every action, the agent receives only static task text, chronological
on/off history, remaining budget, valid slot range, and phase-specific output
schema. It does not receive the evaluator seed, hidden theta, reward, score,
or likelihood in the qualitative condition. It has no carried state between
episodes.

Both conditions state the answer range (theta lies in `[theta_min, theta_max]`,
4 through 27 by default) and the inspectable slot range (0 through 31). The
range is a property of the answer space, not of the hidden time/light
relationship, so disclosing it in only one arm would confound in-context model
discovery with guessing outside the support. The two conditions therefore differ
by exactly one thing: whether the Bernoulli likelihood is supplied. The hidden
theta *value* remains private in both.

The terminal JSON schema still accepts any slot in `[0, slot_count - 1]` rather
than clamping to the theta support, so an estimate outside the stated range
stays visible in the trace as a diagnostic instead of being silently prevented.

Native thinking and raw JSON are retained in `AgentDecision` for diagnostic
traces. Thinking text is never included in a later prompt, preventing it from
acting as hidden cross-turn or cross-episode memory.

## Output and failure behavior

During observation turns, valid output is exactly:

```json
{"kind":"observe","time_slot":12}
```

On the terminal turn, valid output is exactly:

```json
{"kind":"estimate","theta_hat":12}
```

The adapter allows two schema-only retries. On persistent transport, JSON, or
schema failure, it returns the evaluator-compatible public-history fallback,
marks `protocol_failure` and `fallback_used`, and preserves diagnostics. The
evaluator must include that episode in reported metrics.

## Local readiness

Run `light-learning preflight` before a live experiment. It checks for at
least 10 GiB free disk, a reachable Ollama server, and both Qwen3 models; it
makes no downloads or other state changes.
