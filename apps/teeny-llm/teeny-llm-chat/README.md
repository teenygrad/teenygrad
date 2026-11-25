# teeny-llm-chat

Interactive command-line chat for [teenygrad](https://teenygrad.org). It runs the model in-process
through [`teeny-llm-core`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_core/), so no
server is needed, and streams replies as they're generated.

## Getting started

The binary runs the models it is built with; each model is a feature, with its backend:

```bash
cargo install teeny-llm-chat --features functiongemma3,cuda
teeny-llm-chat --model functiongemma3 --tools tools.json
```

`--tools` takes OpenAI-style function declarations; the model's calls are shown in the
transcript, and you type each result. Weights come from the local Hugging Face cache, or
`--model-dir`. Without `--model`, it chats with `teeny-echo`, which echoes your message.

Commands at the prompt: `/clear` (forget the conversation), `/help`, `/exit` (or Ctrl-D).

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
