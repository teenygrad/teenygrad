# docs-site

Builds and serves teenygrad's self-hosted docs (API reference + the [mdBook](../books/teenygrad/) SDK book)
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

Not currently wired up. The site was previously deployed to `docs.teenygrad.org` via Kamal
(nginx serving static files, mirroring `../../spinorml-cdn`); that approach has been retired
and its config removed. `build.sh` still produces a self-contained `dist/` that any static
host can serve.
