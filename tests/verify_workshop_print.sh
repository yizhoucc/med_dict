#!/bin/zsh
set -euo pipefail

# 190 MiB OS-level file-size ceiling. The Swift verifier adds a 10 MiB result
# ceiling and deletes its temporary PDF on both success and failure.
ulimit -f 389120
swift tests/verify_workshop_print.swift

# Prove that a failure occurring after the PDF has been written still removes
# that exact file before the process exits. The EXIT trap is only a final
# safeguard; the existence check runs first and fails if Swift cleanup did not.
fault_pdf="/private/tmp/med_dict_workshop_print_fault_$$.pdf"
rm -f -- "$fault_pdf"
trap 'rm -f -- "$fault_pdf"' EXIT
set +e
fault_output="$(MED_DICT_PRINT_OUTPUT_PATH="$fault_pdf" MED_DICT_PRINT_FAULT_AFTER_WRITE=1 swift tests/verify_workshop_print.swift 2>&1)"
fault_status=$?
set -e
if (( fault_status == 0 )); then
  print -u2 -- "Fault injection unexpectedly succeeded"
  exit 1
fi
if [[ -e "$fault_pdf" ]]; then
  fault_bytes=$(stat -f '%z' "$fault_pdf")
  rm -f -- "$fault_pdf"
  print -u2 -- "Fault injection left a PDF behind (${fault_bytes} bytes)"
  exit 1
fi
if [[ "$fault_output" != *"Injected post-write validation failure"* ]]; then
  print -u2 -- "$fault_output"
  print -u2 -- "Fault injection did not reach the post-write failure"
  exit 1
fi
print -- "WK_FAULT_CLEANUP_OK post-write failure left no PDF"
