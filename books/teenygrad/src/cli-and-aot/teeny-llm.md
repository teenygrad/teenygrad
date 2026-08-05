# `teeny-llm`

`teeny-llm` is the workspace's LLM inference application — vLLM-style serving, aimed at 2x the
throughput (see the [roadmap](../appendix/faq-and-roadmap.md)).

> ⚠️ **Early scaffolding.** The `teeny-llm` binary currently only prints a placeholder banner.
> The planned direction is a single binary offering both an interactive mode and an
> OpenAPI-compatible HTTP server mode — earlier separate `agent`/`console` binaries were dropped
> in favor of this single-binary design.

```bash
cargo install teeny-llm
teeny-llm
```

> **TODO**: fill this chapter in as `teeny-llm` gains real functionality — request/response API
> shape, model loading, batching/scheduling behavior.
