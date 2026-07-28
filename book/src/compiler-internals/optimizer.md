# The egg/z3 Optimizer

`teeny-compiler` depends on [`egg`](https://docs.rs/egg) (e-graph based equality saturation) for
graph-level optimization/rewriting as part of the FXGraph compilation pipeline (see
[Compilation Flow](../core-concepts/compilation-flow.md)). `teeny-fxgraph` and `teeny-torch` also
depend on `egg`.

`z3` (SMT solving) is referenced in project documentation as a planned complement to `egg` for
optimization, but is not currently wired up as a dependency of any workspace crate — treat it as
roadmap, not shipped functionality, until a `z3` dependency actually appears in a `Cargo.toml`.

> **TODO**: document the specific e-graph rewrite rules and what they optimize for, once this
> layer has enough surface area to be worth a dedicated walkthrough.
