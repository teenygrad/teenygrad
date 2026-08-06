# Atomics

Every kernel so far has had a simple guarantee: each program writes to memory
that no other program writes to. Programs never disagree, because they never
touch the same address.

Some problems cannot be written that way. This chapter is about the tool for
those, and about why it is the last tool to reach for.

## The problem

Suppose several programs each need to add something to the same counter.

```text
program A: read count (5) ... add 1 ... write 6
program B: read count (5) ... add 1 ... write 6
```

Both read 5. Both write 6. One increment vanished. This is a **race**, and on a
GPU with thousands of programs in flight it is not a rare edge case — it is what
happens.

The read, the modify and the write have to be one indivisible step. That is what
an atomic is.

```rust,ignore
T::atomic_add(ptr.add_offsets(offsets), values, None, None, None);
```

## Where they are genuinely needed

**Scatter.** When the *output* index is computed from data rather than from
`program_id`, two programs can land on the same place and you cannot prove
otherwise. Negative log-likelihood loss is exactly this: the gradient goes to
the target class, and the target comes from a label tensor.

The library's kernel does it in two steps — a normal masked store for the bulk
of the row, then an atomic for the one data-dependent position:

```rust,ignore
// Step 2: subtract dy at target position via atomic_add(-dy)
let base: T::Tensor<i32> = T::full::<i32>(&[1], row_base);
let flat_off: T::Tensor<i32> = base + tgt;
let neg_dy = T::full(&[1], -1.0_f32) * dy;
T::atomic_add(dx_ptr.add_offsets(flat_off), neg_dy, None, None, None);
```

**Gradient accumulation in a backward pass.** If a value was read by many
outputs in the forward pass, its gradient is the sum of many contributions.
Convolution backward is the standard case, and several of this tree's `conv`
and `pad` backward kernels use atomics for exactly this.

**Histograms and counters.** Bin from data, increment. The definition of the
problem is a race.

## The full set

| Method | Operation | Dtypes |
|---|---|---|
| `atomic_add` | `*p += v` | numeric |
| `atomic_max`, `atomic_min` | `*p = max/min(*p, v)` | numeric |
| `atomic_and`, `atomic_or`, `atomic_xor` | bitwise | integer |
| `atomic_xchg` | `*p = v`, returns old | any |
| `atomic_cas` | compare and swap | any |

All return the **previous** value, which is what makes `atomic_add` usable as a
"claim me a slot" primitive: the value you get back is your index.

All except `atomic_cas` take a mask, so only the lanes you want participate.

## Ordering and scope

The last two arguments control how strongly the operation is ordered against
everything else.

`MemSem` — memory semantics:

| Value | Meaning |
|---|---|
| `Relaxed` | Atomic, but no ordering guarantee about anything else |
| `Acquire` | Later operations cannot move before this one |
| `Release` | Earlier operations cannot move after this one |
| `AcqRel` | Both. The default |

`MemScope` — who has to agree:

| Value | Meaning |
|---|---|
| `Cta` | Only programs in this block |
| `Gpu` | All programs on this device. The default |
| `Sys` | The whole system, including the host |

Passing `None` for both gives `AcqRel` and `Gpu`, which is correct and is what
every kernel in this tree does. Weakening them is a real optimisation — a
`Relaxed` counter is cheaper than an `AcqRel` one — but it is the kind of change
to make when you are measuring and can explain why it is safe.

## What they cost

**Contention.** When many programs hit the same address, the hardware serialises
them. A histogram where 90% of values land in one bin runs at roughly the speed
of one program. The fix is usually to reduce first and then use one atomic per
program rather than one per lane.

**Non-determinism.** Floating-point addition is not associative, so atomics
arriving in a different order give a different sum in the last bits. Run the same
kernel twice on the same input and you may get answers that differ by an ulp.

That is worth stating clearly because of what it does to your tests. A backward
pass using atomics is not bit-reproducible, so exact-equality assertions will
fail intermittently. Compare with a tolerance, as the tests in this tree do.

## The alternatives, first

Before an atomic, check these:

**Can you change who owns the output?** Often a race exists because the kernel
is parallelised over inputs. Parallelise over *outputs* instead — one program
owning each output element — and the race disappears. This is the single most
common fix.

**Can you use two kernels?** Partial results per program, then a second kernel
to combine them. More memory and another launch, but deterministic and usually
faster under contention.

**Can you reduce within the block first?** If all 128 lanes are adding to the
same place, `T::sum` them and do one atomic instead of 128.

The reasonable default is: parallelise over outputs, and use atomics only when
the output index genuinely comes from the data.

Next: making one kernel serve several dtypes.
