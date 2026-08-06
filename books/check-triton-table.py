#!/usr/bin/env python3
"""Check the book's Python-Triton-to-Rust table against the `Triton` trait.

The table in books/kernels/src/reference/translation-table.md is meant to be
every method a kernel author can call. Prose cannot be generated from source,
but completeness can be checked against it — so adding a method to the trait
without documenting it fails here rather than leaving a hole a reader finds.

    python3 books/check-triton-table.py

Exits non-zero, naming what is missing or stale.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIT = REPO / "kernels/teeny-triton/src/triton/mod.rs"
TABLE = REPO / "books/kernels/src/reference/translation-table.md"

# Methods declared at trait-body indentation, e.g. "    fn program_id(".
TRAIT_METHOD = re.compile(r"^    fn ([a-z_0-9]+)", re.MULTILINE)
# Mentions in the table, written as `T::name(` or `T::name::<`.
TABLE_METHOD = re.compile(r"`T::([a-z_0-9]+)")


def main():
    for path in (TRAIT, TABLE):
        if not path.is_file():
            print(f"missing {path.relative_to(REPO)} — has the layout changed?", file=sys.stderr)
            return 1

    declared = set(TRAIT_METHOD.findall(TRAIT.read_text(encoding="utf-8")))
    documented = set(TABLE_METHOD.findall(TABLE.read_text(encoding="utf-8")))

    undocumented = sorted(declared - documented)
    stale = sorted(documented - declared)

    if undocumented:
        print(f"{len(undocumented)} trait method(s) missing from the translation table:", file=sys.stderr)
        for name in undocumented:
            print(f"  T::{name}", file=sys.stderr)
    if stale:
        print(f"{len(stale)} table entr(y/ies) name a method the trait no longer has:", file=sys.stderr)
        for name in stale:
            print(f"  T::{name}", file=sys.stderr)

    if undocumented or stale:
        print(
            f"\n{TABLE.relative_to(REPO)} must list every method in "
            f"{TRAIT.relative_to(REPO)}.",
            file=sys.stderr,
        )
        return 1

    print(f"translation table covers all {len(declared)} Triton methods")
    return 0


if __name__ == "__main__":
    sys.exit(main())
