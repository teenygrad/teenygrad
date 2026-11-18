# teeny-llm-chat

Interactive command-line chat for [teenygrad](https://teenygrad.org). It runs the model in-process
through [`teeny-llm-core`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_core/), so no
server is needed, and streams replies as they're generated.

> ⚠️ Early scaffolding: only the placeholder `EchoEngine` (model `teeny-echo`) is available, which
> echoes your last message back.

## Getting started

```bash
cargo install teeny-llm-chat
teeny-llm-chat --system "You are a helpful assistant."
```

Commands at the prompt: `/clear` (forget the conversation), `/help`, `/exit` (or Ctrl-D).

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
