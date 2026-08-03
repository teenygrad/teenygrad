# Agent Instructions

## Branching

**Never work directly on `main`.** Check the current branch (`git branch --show-current`) before your first edit; if it is `main`, create and switch to a branch first.

- `feature/<short-description>` for changes, fixes, and new work
- `release/<version>` for release preparation

Changes reach `main` only through a reviewed pull request. A `PreToolUse` hook in `.claude/settings.json` blocks file edits while the default branch is checked out.

## Rust Engineering Standards

When making Rust changes, prefer correctness, clarity, and maintainability over cleverness.

### Idiomatic Rust

- Write code that is straightforward to read and follows standard Rust conventions.
- Prefer iterators, pattern matching, and enums over ad-hoc state flags and complex control flow.
- Minimize cloning and allocations; pass references where practical and use ownership intentionally.
- Keep functions focused and small; extract helpers when logic becomes hard to scan.
- Avoid panics in non-test code unless truly unrecoverable; return structured errors instead.

### Safety and Best Practices

- Do not use `unwrap()`/`expect()` in production paths when errors can be handled or propagated.
- Keep `unsafe` blocks minimal, documented with invariants, and covered by tests.
- Prefer explicit types when inference hurts readability.
- Use existing abstractions in the codebase before introducing new patterns or dependencies.
- Preserve backward compatibility and existing behavior unless the issue explicitly requires change.

### Testing Expectations

- Add or update tests for behavior changes and bug fixes.
- Prefer targeted tests first (unit or focused integration tests), then broader suites as needed.
- Include negative/error-path tests for new logic where applicable.
- Avoid brittle tests; assert semantics, not incidental formatting or implementation details.

### Code Quality Gates

Run relevant checks before finishing work (scope to changed areas when possible):
