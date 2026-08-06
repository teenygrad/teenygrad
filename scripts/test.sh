#!/bin/sh
#
# Runs the workspace test suite with a clean teenyc kernel cache.
#
# Several kernel-compiling tests share $TEENYC_CACHE_DIR (default
# /tmp/teenyc_cache) as a scratch/object-cache directory. Under `cargo test`'s
# default parallel test execution, two tests compiling different kernels at
# the same time can race on that shared directory (e.g. one test's compiler
# invocation renames/removes a file another test's invocation is still
# writing), causing spurious "could not copy ... No such file or directory"
# failures unrelated to the code under test. Starting from a clean, empty
# cache directory doesn't eliminate that race, but it does make failures
# reproducible from a known state rather than depending on whatever a
# previous run left behind.

set -e

cache_dir="${TEENYC_CACHE_DIR:-/tmp/teenyc_cache}"
rm -rf "$cache_dir"

exec cargo test "$@"
