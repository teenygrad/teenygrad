# Agent Instructions

## Branching

**Never work directly on `main`.** Check the current branch (`git branch --show-current`) before your first edit; if it is `main`, create and switch to a branch first.

- `feature/<short-description>` for changes, fixes, and new work
- `release/<version>` for release preparation

Changes reach `main` only through a reviewed pull request. A `PreToolUse` hook in `.claude/settings.json` blocks file edits while the default branch is checked out.

The one exception is the orphan `beads` branch (see Issue Tracking below): it holds only issue data, shares no history with `main`, and is committed and pushed directly, never through a pull request.

## Issue Tracking (beads)

Issues are tracked with **br** (beads_rust). The tracker lives on an orphan `beads` branch of this repo, checked out as a git worktree at `.beads/`. `.beads` is gitignored on code branches, so issues never travel with feature branches and never conflict in their merges.

One-time setup per clone:

```bash
git fetch origin beads
git worktree add .beads beads
```

Day-to-day:

```bash
br ready                      # find available work
br show <id>                  # view an issue
br create ... --parent <id>   # file an issue (under an epic)
br update <id> --claim        # claim work
br close <id>                 # complete work
br sync --flush-only          # export the DB to .beads/issues.jsonl
git -C .beads add issues.jsonl
git -C .beads commit -m "sync beads: <summary>"
git -C .beads push
```

- Only `issues.jsonl`, `config.yaml`, `metadata.json` and `.gitignore` are tracked on the `beads` branch; `beads.db`, lock files and `last-touched` stay local.
- `br` never runs git itself: after `br sync --flush-only`, commit and push from `.beads/` as above.
- Never add `.beads` to a code branch, and never merge `beads` into `main` (or the reverse).
- The pre-push hook skips clippy when only the `beads` branch is pushed.
- This repo is public, so everything written in an issue is public.

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
