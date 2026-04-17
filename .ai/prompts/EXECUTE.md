# Execute one linear issue

**Linear issue id for this invocation:** $ISSUE_ID

(The `agent` script replaces `$ISSUE_ID` before sending this file to Claude. If you still see the literal text `$ISSUE_ID`, run: `./agent EXECUTE <issue-id>`.)

## Your role

You are a coding agent. For this invocation you read **one** issue identified by
`ISSUE_ID` and carry out the work described there. Keep work focused to that one
issue. Do not run fixed iteration caps or retry "loops." Do not handle merge or
review-comment workflows unless the issue text itself asks for it.

You do **one** issue per invocation, then stop and report.

---

## Input

- `ISSUE_ID` (required) — the Linear issue id to execute

If `ISSUE_ID` is missing or empty, stop and report:
`Missing required parameter: ISSUE_ID`.

---

## Prerequisites

- `ISSUE_ID` is provided
- `linear` is available if you use it to load the issue (optional but typical)

---

## Workflow

### 1. Load the issue

```bash
linear issue view "$ISSUE_ID" --json
```

Use the provided `ISSUE_ID` exactly. Do not pick a different issue.

If the issue does not exist, report that and stop.

Read the title, description, acceptance criteria, dependencies, and recent
comments. If something is ambiguous, add a short clarification comment and stop
rather than guessing, for example:

```bash
linear issue comment add "$ISSUE_ID" --body "Need clarification on <specific point> before implementation."
```

### 2. Claim (optional but recommended)

Prefer a non-blocking claim attempt with your team key (`ART`).
If your workspace uses a different key, replace `ART` in the command below.

```bash
set -euo pipefail
linear issue start "$ISSUE_ID" --team ART || echo "Could not start issue; continuing."
```

Do not fail the task solely because claiming failed.

### 3. Implement

- Start from a clean, up-to-date `main` before creating your issue branch:
  - If `git status --porcelain` is not empty, stop and report a dirty working tree.
  - Switch to `main` (`git switch main`).
  - Update local `main` from remote (`git pull --ff-only origin main`).
- Create the issue branch from that exact `main` head. Name it `feat/linear-<issue-id>`.
- Inspect the relevant code before changing it.
- Stay within the issue’s scope.
- Run checks/tests appropriate to the change (project defaults, e.g. `./x.py
  check` and targeted tests when they match the issue).
- Commit with clear messages when you have coherent units of work.

Pull request behavior:

- If you made code changes, you must open or update a PR for this issue.
- If fully complete, PR description should state complete scope and verification.
- If partially complete, PR description must clearly list what is incomplete and why.
- Add a Linear comment linking the PR. If partial, include the next concrete step.
- If no code changes were needed (for example duplicate/already fixed), do not open a PR and explain why in the report and a Linear comment.

Do **not** run a fixed "try N times then give up" loop; fix issues until the
work matches the issue or you hit a genuine blocker you report.

### 4. Report

Summarize briefly:

- Task id and title
- What you did
- How you verified (commands, outcome)
- Blockers, if any (with enough detail to continue later)
- PR status (created/updated/not needed) and link when applicable
- If incomplete: the exact reason, Linear comment confirmation, and next step

---

## Hard rules

- Use only the provided `ISSUE_ID`; do not auto-select from `linear issue list` or lists.
- One issue per invocation; do not chain the next issue automatically.
- Do not force-push to protected/default branches unless the issue explicitly
  requires it and you have agreement to do so.

---

## Exit tokens (optional)

- When the invocation finishes successfully for the issue as described, you may
  print: `TASK_SUCCESS`
- On failure, missing issue, or abort: `TASK_FAILED`
