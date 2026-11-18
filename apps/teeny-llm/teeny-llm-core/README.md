# teeny-llm-core

Core LLM inference library for [teenygrad](https://teenygrad.org). It defines the
transport-independent pieces shared by
[`teeny-llm-server`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_server/) (OpenAI-compatible
HTTP API) and [`teeny-llm-chat`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_chat/)
(interactive CLI):

- `ChatMessage` / `Role`: a conversation
- `Prompt`, `SamplingParams`, `GenerationRequest`: what to generate
- `Engine`: the trait an inference backend implements (streaming generation, optional embeddings)
- `EchoEngine`: a placeholder engine that echoes the last user message, so the server and CLI can be
  exercised end to end before real model execution lands

> ⚠️ Early scaffolding: no real model is loaded yet.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
