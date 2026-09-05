# Git hooks

Hooks in this directory aren't active until you point git at them (git
doesn't do this automatically, and `.git/hooks` isn't version-controlled):

```sh
git config core.hooksPath .githooks
```

Run that once per clone (or worktree, since `core.hooksPath` is a local
config setting). `setup_ubuntu.sh` does this for you.

- `pre-push` — runs the same `cargo clippy -- -D warnings` check as CI's
  "Build, test, clippy" job, so a lint regression is caught locally before
  it fails the pipeline.
