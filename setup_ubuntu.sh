#!/bin/sh

set -e

# install build tools
sudo apt-get install build-essential z3 libz3-dev lld

RELOGIN_REQUIRED=false

# checkif rustc is installed, if not install it
if ! command -v rustc &> /dev/null
then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    . "$HOME/.cargo/env"

    cargo install cargo-expand
    cargo install cargo-watch

    RELOGIN_REQUIRED=true
else
    echo "Rust is installed, updating it"
    rustup update
fi


if [ "$RELOGIN_REQUIRED" = true ]; then
    echo "Please logout and login again, as new tools are installed"
    exit 0
fi
