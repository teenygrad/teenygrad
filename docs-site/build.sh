#!/bin/sh
# Builds the self-hosted docs site (API reference + mdBook) into
# docs-site/dist/, ready to be served as static files (see Dockerfile).
#
# Requires the CUDA toolkit (teeny-cuda's build.rs hard-requires it) and
# `mdbook`/`mdbook-mermaid` (`cargo install mdbook mdbook-mermaid`, then
# `mdbook-mermaid install books/teenygrad` once, already done in this repo).
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
doc_excludes="--exclude teeny-torch --exclude teeny-fxgraph"

# teeny-cuda's build.rs hard-requires the CUDA toolkit; teeny-kernels'
# default features enable teeny-cuda. If it's not on this machine (e.g. a
# CI runner without it provisioned), skip both rather than fail the whole
# build -- better an incomplete site than no site. Run this script on a
# CUDA-capable host (as this repo's own dev sandbox is) to get full
# coverage.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "docs-site/build.sh: nvcc not found -- skipping teeny-cuda/teeny-kernels docs" >&2
  doc_excludes="$doc_excludes --exclude teeny-cuda --exclude teeny-kernels"
fi

# shellcheck disable=SC2086
cargo doc --workspace --no-deps $doc_excludes

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
( cd books/teenygrad && mdbook build )
cp -r books/teenygrad/book "$dist/book"

cp docs-site/index.html "$dist/index.html"

echo "Docs site built at $dist"
