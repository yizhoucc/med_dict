#!/bin/zsh
set -euo pipefail

# 190 MiB OS-level file-size ceiling. The Swift verifier adds a 10 MiB result
# ceiling and deletes its temporary PDF on both success and failure.
ulimit -f 389120
exec swift tests/verify_workshop_print.swift
