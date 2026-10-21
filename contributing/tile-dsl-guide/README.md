# Tile DSL guide — LaTeX source

Source for `TILE_DSL_GUIDE.pdf` (repo root): the tile DSL, reverse tile-shape
inference worked on conv2d and batch norm, and a Welder schedule for
conv2d → batch_norm → silu. Written against `feature/anduin-v2` at `7da014031`
plus PR #103.

Build (TeX Live; needs tikz, listings, tcolorbox, xltabular, url):

```sh
pdflatex main.tex && pdflatex main.tex   # twice, for the contents and cross-references
```

`main.tex` includes `s1_overview.tex` … `s10_refs.tex`, one per chapter. Two
preamble details to keep in mind when editing:

- `#` is catcode "other" inside the document body, so `\code{#[tile(...)]}`
  prints one `#`. Do not define macros with parameters after `\begin{document}`.
- `\code{..}` is a `url`-style command, so it breaks at `_ / ( , =`. Write its
  contents literally: no `\{`, no `$..$` inside it.

`numbers.py` and `tiles.py` compute the traffic and footprint tables in §6.4 and
§9.4–§9.5 (`python3 numbers.py`, `python3 tiles.py`).
