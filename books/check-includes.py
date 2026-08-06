#!/usr/bin/env python3
"""Check that every {{#include}} in the books resolves.

The books keep their code samples in the crates they document and pull them in
by anchor, so a chapter can never drift from code that compiles. The cost is a
new way to break: rename an example, move an anchor, and the markdown still
looks fine while the published page loses its code.

The docs site catches that too, and is the authority — this runs the same rules
in this repo so the failure lands on the pull request that caused it rather than
on a deploy hours later. Kept deliberately small; if the two ever disagree,
scripts/includes.mjs in teenygrad/docs is correct.

    python3 books/check-includes.py

Exits non-zero and lists every unresolved include.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIRECTIVE = re.compile(r"\{\{#(include|rustdoc_include|playground)\s+([^}\s][^}]*?)\s*\}\}")
ANCHOR_START = re.compile(r"ANCHOR:\s*([\w-]+)")
ANCHOR_END = re.compile(r"ANCHOR_END:\s*([\w-]+)")


def parse_spec(spec):
    """Split `file.rs:anchor` / `file.rs:10:20` into a path and what to take."""
    parts = spec.split(":")
    rest = [p.strip() for p in parts[1:]]
    if len(rest) == 1 and rest[0] and not rest[0].isdigit():
        return parts[0].strip(), rest[0]
    return parts[0].strip(), None


def check(chapter, problems):
    for kind, spec in DIRECTIVE.findall(chapter.read_text(encoding="utf-8")):
        if kind == "playground":
            problems.append(f"{chapter.relative_to(REPO)}: {{{{#playground}}}} is not supported by the docs site")
            continue

        path, anchor = parse_spec(spec)
        target = (chapter.parent / path).resolve()

        try:
            target.relative_to(REPO)
        except ValueError:
            problems.append(f"{chapter.relative_to(REPO)}: {path} resolves outside the repo")
            continue

        if not target.is_file():
            problems.append(f"{chapter.relative_to(REPO)}: {path} does not exist")
            continue

        if anchor:
            body = target.read_text(encoding="utf-8")
            opened = any(m.group(1) == anchor for m in ANCHOR_START.finditer(body))
            closed = any(m.group(1) == anchor for m in ANCHOR_END.finditer(body))
            if not opened:
                problems.append(f"{chapter.relative_to(REPO)}: {path} has no `ANCHOR: {anchor}`")
            elif not closed:
                problems.append(f"{chapter.relative_to(REPO)}: {path} never closes `ANCHOR: {anchor}`")


def main():
    chapters = sorted(REPO.glob("books/*/src/**/*.md"))
    if not chapters:
        print("no chapters found under books/*/src/ — has the layout changed?", file=sys.stderr)
        return 1

    problems = []
    for chapter in chapters:
        check(chapter, problems)

    if problems:
        print(f"{len(problems)} unresolved include(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"all includes resolve ({len(chapters)} chapters checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
