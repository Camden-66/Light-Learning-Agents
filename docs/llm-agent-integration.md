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
pulls a model. `run_metadata()` captures the installed model digest and
Ollama show metadata for the evaluator's manifest.

## Information boundary

For every action, the agent receives only static task text, chronological
on/off history, remaining budget, valid slot range, and phase-specific output
schema. It does not receive the evaluator seed, hidden theta, reward, score,
or likelihood in the qualitative condition. It has no carried state between
episodes.

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
