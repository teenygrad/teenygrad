# Week ending - 03/Aug/2025

## Sprint Goal

Get Qwen3 working on CPU using ndarray.

### Tasks

1. Qwen3 graph completed with safetensors.
2. Compiler to convert the graph to ndarray implementation.
3. Fix all the issues that arise.

### Tech debt

- Shape tracked has been removed during this sprint, to take a step back and design a better shape tracking system.
- Config clone: required because of safetenor lifetime issues and references, will revisit once everything works
- The API is not very ergonomic at the moment, will need to revisit once the whole thing works and I have a clearer idea of what the right traits and structs are.

### Sprint Review

- Good
  - To Do

- Bad
  - To Do

- Ugly
  - To Do
