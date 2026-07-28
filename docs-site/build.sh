#!/bin/sh
# Builds the self-hosted docs site (API reference + mdBook) into
# docs-site/dist/, ready to be served as static files (see Dockerfile).
#
# Requires the CUDA toolkit (teeny-cuda's build.rs hard-requires it) and
# `mdbook`/`mdbook-mermaid` (`cargo install mdbook mdbook-mermaid`, then
# `mdbook-mermaid install book` once, already done in this repo).
set -e

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"

dist="docs-site/dist"
rm -rf "$dist"
mkdir -p "$dist/api"

# Clean target/doc first: it accumulates output from whatever was last built
# there (including excluded/unpublished crates), and we only want to publish
# exactly the publishable-crate set below.
rm -rf target/doc

# API reference for every publishable crate. teeny-torch and teeny-fxgraph
# are excluded on purpose (publish = false -- PyTorch compat layer, not part
# of the public Rust SDK).
cargo doc --workspace --no-deps \
  --exclude teeny-torch --exclude teeny-fxgraph

cp -r target/doc/. "$dist/api/"

# cargo doc doesn't generate a top-level crate index -- build one.
{
  echo "<!doctype html><meta charset=utf-8><title>API Reference</title>"
  echo "<h1>teenygrad API Reference</h1><ul>"
  for d in "$dist"/api/teeny_*/; do
    name="$(basename "$d")"
    echo "<li><a href=\"${name}/index.html\">${name}</a></li>"
  done
  echo "</ul>"
} > "$dist/api/index.html"

# mdBook SDK book.
( cd book && mdbook build )
cp -r book/book "$dist/book"

cp docs-site/index.html "$dist/index.html"

echo "Docs site built at $dist"
