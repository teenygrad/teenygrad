# Contributing to this book

The book is mdbook source under `books/kernels/`. It is published twice: by
`mdbook serve` locally, and by the docs site at
[docs.teenygrad.org/kernels](https://docs.teenygrad.org/kernels), which renders
the same files. Keep it valid mdbook input and both work.

```
books/kernels/
  book.toml
  src/SUMMARY.md      the table of contents — a chapter not listed here is not published
  src/**.md           chapters
  OUTLINE.md          the plan: every chapter's goal and the API it uses
  KNOWN-GAPS.md       what the SDK cannot yet demonstrate
  API-FRICTION.md     where the Rust API costs more than the Python equivalent

kernels/teeny-triton/examples/     runnable examples written for the book
kernels/teeny-kernels/src/         the library kernels later chapters teach from
books/check-includes.py            verifies the book and the code stay connected
books/check-triton-table.py        verifies the reference table covers the whole DSL
```

Chapters include from both. Prefer a library kernel when one already does the
job — it is code that ships, reviewed and tested by people who are not writing a
book. Write a new example when you need something small enough to hold in one
page, or a complete program a reader can run start to finish.

Including from a library kernel means adding `// ANCHOR:` comments to it. They
are inert comments; keep them tight around the region a chapter shows.

## Adding a chapter

1. Write `src/<part>/<chapter>.md`. Start with a single `# Heading` — it becomes
   the page title and is rendered by the site's page shell, so do not repeat it.
2. Add it to `src/SUMMARY.md` under the right part. A chapter with no link
   (`- [Title]()`) is a draft: it shows in the sidebar, unlinked, and publishes
   nothing.
3. Run `mdbook serve` and read it.

House style, in short: plain English, short sentences, second person, one new
idea per chapter, why before how, no marketing. Callouts (`> `) only for things
that will actually bite the reader. The full brief is in `OUTLINE.md`.

Every chapter ends with code that runs.

## Adding an example

Examples are cargo examples in `kernels/teeny-triton/examples/` — one complete,
runnable program per chapter. Nothing is retyped into the prose.

1. Write `kernels/teeny-triton/examples/<name>.rs`. It is a whole program: the
   kernel, compiling it, allocating buffers, launching, checking the answer.
2. Register it in `kernels/teeny-triton/Cargo.toml`, behind the `cuda` feature:

   ```toml
   [[example]]
   name = "<name>"
   required-features = ["cuda"]
   ```

   Every example needs a device, so every example is gated. Without the feature
   cargo skips the target entirely, which is what keeps the crate checkable on a
   machine with no CUDA toolkit.

3. Mark the regions the chapter shows with mdbook anchors:

   ```rust
   // ANCHOR: kernel
   #[kernel]
   pub fn vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(
       // ...
   ) { }
   // ANCHOR_END: kernel
   ```

   Use several — a chapter usually wants the kernel and the launch separately,
   with prose between them.

4. Pull them into the chapter:

   ````markdown
   ```rust
   {{#include ../../../../kernels/teeny-triton/examples/vector_add.rs:kernel}}
   ```
   ````

   Four `..` from `books/kernels/src/<part>/` reaches the repo root. Line ranges
   (`file.rs:10:20`) work too, but anchors survive edits and line numbers do not.

5. Run it, and put its real output in the chapter:

   ```bash
   cargo run -p teeny-triton --features cuda --example <name>
   ```

An include naming a missing file or anchor fails the build on both renderers,
and `python3 books/check-includes.py` catches it before you push. That is the
point: the book cannot drift from code that runs.

### Why the examples are not in the book directory

They were, briefly. They live in the crate instead because
`kernels/teeny-triton` is where the DSL they exercise is defined, so they are
kept honest by the same CI and the same reviewers as the code they demonstrate.

The docs site fetches them alongside the markdown — see `includePaths` in the
docs repo's `books.config.mjs`. If you add examples somewhere else, that list
needs the new directory or the published pages lose their code.

## Showing output

Show real numbers, and name the hardware they came from. "Much faster" is not a
measurement. If you have not run it, leave the table empty and say it is not yet
measured — an invented number is worse than a gap.

## Before you open a PR

```bash
mdbook build                                  # no warnings
python3 books/check-includes.py               # every include still resolves
python3 books/check-triton-table.py           # the reference table is complete
cargo check -p teeny-triton                   # the crate builds without CUDA
```

And, on a machine with a GPU:

```bash
cargo run -p teeny-triton --features cuda --example <name>
```

CI runs the first four. It cannot run the last one — GitHub's runners have
neither the CUDA toolkit nor a device — so an example is only ever verified by
someone running it.

If the chapter needed something the SDK does not have, add it to
`KNOWN-GAPS.md`. If it needed something the SDK has but makes awkward, add it to
`API-FRICTION.md`. Both are deliverables, not scratch files.
