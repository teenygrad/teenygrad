# docs-site

Builds and serves teenygrad's self-hosted docs (API reference + the [mdBook](../book/) SDK book)
as one static site — this is what's deployed to `docs.teenygrad.org`.

Self-hosted (rather than relying solely on docs.rs) because `teeny-cuda` requires the CUDA
toolkit to even build its docs, which docs.rs's sandbox can't provide.

## Build

```bash
# from the repository root
./docs-site/build.sh
```

Requires: the CUDA toolkit (for `teeny-cuda`'s docs), and `mdbook`/`mdbook-mermaid`
(`cargo install mdbook mdbook-mermaid`). Produces `docs-site/dist/` (gitignored).

## Run locally

```bash
docker build -t teenygrad-docs-site docs-site
docker run --rm -p 8080:80 teenygrad-docs-site
```

Then visit `http://localhost:8080/` — `/api/` for rustdoc, `/book/` for the mdBook.

## Deployment

Not yet wired up — see the deploy-CI and Kamal-config follow-up tasks. Intended to mirror
`../../spinorml-cdn`'s Kamal setup (nginx serving static files, no application logic).
