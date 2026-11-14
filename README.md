# free-model-aggregator-01

A deterministic free-model aggregator: one Python server (stdlib only) that
reads the trial brief, probes each simulated provider's catalog route, and
serves the OpenAI Chat Completions and Anthropic Messages APIs. Every client
request is routed to the highest-rated free model able to serve it (tools,
vision, JSON mode, minimum context), with immediate failover to the next
best candidate on 402/429/5xx, and provider budgets tracked so a refused
provider is not retried blindly. Responses are translated wire-to-wire with
exact usage and stop-reason mapping, including SSE streaming.

Run: `python3 main.py` with `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL` and
`MAKEFUN_TRIAL_SEED` set. `python3 main.py --list` prints the catalog it
would register.
