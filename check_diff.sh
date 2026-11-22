#!/usr/bin/env bash
# Compare GPU greedy outputs with the CPU reference, request by request.
#   ./check_diff.sh 12      # output/<M>_12_gpu.txt and output/<M>_12_gpu_pp.txt vs output/<M>_12.txt
# Exit code 1 if any output file is missing or any request differs.
set -uo pipefail
TAG="${1:-12}"
cd "$(dirname "$0")"
status=0
for m in 15M 42M 110M; do
  ref="output/${m}_${TAG}.txt"
  for v in gpu gpu_pp; do
    out="output/${m}_${TAG}_${v}.txt"
    if [[ ! -f "$ref" || ! -f "$out" ]]; then echo "$m $v: missing $ref or $out"; status=1; continue; fi
    python3 -I - "$ref" "$out" "$m $v" <<'PY' || status=1
import sys
ref = open(sys.argv[1]).read().splitlines()[1:]
out = open(sys.argv[2]).read().splitlines()[1:]
same = sum(a == b for a, b in zip(ref, out)) if len(ref) == len(out) else 0
msg = f"{sys.argv[3]}: {same}/{len(ref)} requests identical to CPU"
firsts = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
          for a, b in zip(ref, out) if a != b]
if firsts:
    msg += f" (the others first differ at char {min(firsts)}..{max(firsts)})"
print(msg)
sys.exit(0 if same == len(ref) else 1)
PY
  done
done
exit $status
