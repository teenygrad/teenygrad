"""Ceiling on overlapping the K3's two clusters: list-schedule each layer graph with
one kernel at a time on the A100s and one on the X100s, against running every node
in turn as the runtime does now.

Run in a directory holding, from one board run of the chat with INKLING2_IME=1:
  dag.tsv  from INKLING2_DUMP_DAG=dag.tsv
  ov.tsv   from TEENY_RUNTIME_PROFILE=ov.tsv
and pass the number of steps it ran (prompt steps plus generated tokens) as the
first argument. It ignores memory contention between the clusters, so the saving is
an upper bound."""
import collections, re, sys

prof, launches = collections.defaultdict(float), collections.defaultdict(int)
for line in open("ov.tsv", encoding="latin-1"):
    ms, n, name = line.rstrip("\n").split("\t")
    name = name.replace("\0", "")
    prof[name] += float(ms); launches[name] += int(n)
avg = {k: prof[k] / launches[k] for k in prof if launches[k]}

graphs = collections.defaultdict(list)
for line in open("dag.tsv"):
    layer, node, name, a100, parents = line.rstrip("\n").split("\t")
    graphs[int(layer)].append((int(node), name, a100 == "1", [int(p) for p in parents.split(",") if p]))

missing = set()
def dur(name):
    key = "unnamed" if name.startswith("#") else name
    if key not in avg:
        missing.add(key); return 0.0
    return avg[key]

core = range(6, 18)
serial_tok = overlap_tok = a_tok = x_tok = 0.0
for layer, nodes in sorted(graphs.items()):
    visits = 3 if layer in core else 1
    done = {}  # node -> finish time
    free = {True: 0.0, False: 0.0}
    remaining = list(nodes)  # already in run (topological) order
    serial = sum(dur(n) for _, n, _, _ in nodes)
    ops = {i for i, _, _, _ in nodes}
    while remaining:
        # The earliest-startable ready node on either cluster.
        best = None
        for k, (i, name, a, parents) in enumerate(remaining):
            if all(p in done or p not in ops for p in parents):
                start = max([free[a]] + [done[p] for p in parents if p in done])
                if best is None or start < best[0]:
                    best = (start, k)
        start, k = best
        i, name, a, parents = remaining.pop(k)
        done[i] = start + dur(name)
        free[a] = done[i]
    makespan = max(done.values())
    serial_tok += serial * visits; overlap_tok += makespan * visits
    a_tok += sum(dur(n) for _, n, a, _ in nodes if a) * visits
    x_tok += sum(dur(n) for _, n, a, _ in nodes if not a) * visits

tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 98
wall = avg["(run) total"] * launches["(run) total"] / tokens
print(f"missing from profile: {sorted(missing)[:8]}")
print(f"layer kernels in turn:   {serial_tok:7.1f} ms per token (A100 {a_tok:.1f}, X100 {x_tok:.1f})")
print(f"two clusters overlapped: {overlap_tok:7.1f} ms per token")
print(f"saving:                  {serial_tok - overlap_tok:7.1f} ms per token")
print(f"runtime's measured run time per token: {wall:.1f} ms (includes embed, head, messages, host work)")
