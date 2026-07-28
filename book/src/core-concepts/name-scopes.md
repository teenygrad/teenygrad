# Name Scopes

`teeny-core::name_scope` ([API docs](https://docs.teenygrad.org/api/teeny-core/name_scope/)),
available under the `std` feature, provides scoped naming for graph nodes/parameters —
useful for giving generated graphs (and, downstream, debug output, checkpoints, profiler traces)
human-readable, hierarchical names instead of anonymous IDs.

```rust,ignore
use teeny_core::name_scope::name_scope;

let _guard = name_scope("encoder/layer0");
// anything built while `_guard` is alive is implicitly namespaced under "encoder/layer0"
```

`current_scope()` returns the active scope name, if any (`None` outside of any `name_scope`
guard).

Because this depends on `std` (thread-local or similar scope-tracking machinery), it's gated
behind `teeny-core`'s `std` feature — unavailable in the default `no_std` build.

> **TODO**: document nested scopes and how scope names interact with parameter naming once that's
> wired into the `nn` layer API.
