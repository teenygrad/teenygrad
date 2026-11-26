# teeny-kvcache

Pluggable key/value caches for [teenygrad](https://teenygrad.org)'s autoregressive models.

A transformer decoding token by token keeps each layer's keys and values for the positions it
has seen, so a step only computes the new ones. This crate holds that state behind the
`KvCache` trait, so a model's session works with any cache:

- `KvCache`: the layout (layers, row width, capacity), the positions held, `append`,
  `truncate`, each layer's keys and values, and the `PastLen` handle a compiled step's
  attention ops read the length from (the cache keeps it up to date);
- `HostKvCache`: contiguous host memory, allocated up front, each layer one `[capacity, row]`
  buffer;
- `PrefixCache`: any cache plus the tokens it holds. `reuse` keeps the longest prefix a new
  prompt shares with them, so re-rendering a conversation each turn only runs what changed.

```rust
use teeny_kvcache::{HostKvCache, KvLayout, PrefixCache};

let layout = KvLayout { layers: 18, row: 256, capacity: 1024 };
let mut cache = PrefixCache::new(Box::new(HostKvCache::new(layout)));
let keep = cache.reuse(&[2, 105, 2364], 1); // positions kept from the last turn
assert_eq!(keep, 0);
```

teeny-llm's FunctionGemma runs on it. Other caches (device-resident, paged, quantized, ring
buffers for sliding-window layers) implement the same trait.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
