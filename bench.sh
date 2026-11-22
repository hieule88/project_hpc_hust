#!/usr/bin/env bash
# End-to-end benchmark + correctness check: CPU reference vs GPU implementations.
#
#   ./bench.sh                    # all models found in model/, input/12.txt, 256 steps
#   MODELS="stories110M" STEPS=128 INPUT=input/32.txt ./bench.sh      (or: make bench)
#
# Every implementation runs the same test-mode loop (harness.h) on the same
# prompts, sampler and seed, and prints one RESULT line. This script:
#   1. builds bin/runtest, bin/runtest_omp and bin/runcu
#   2. throughput: runs each implementation on each model (temperature 1.0, top-p 0.9)
#   3. correctness: runs the CPU reference and each GPU version greedily (-t 0) and
#      compares the generated text request by request
#   4. writes profiling/bench_<host>.csv and profiling/bench_<host>.md
# To add an implementation (e.g. gpu_optimized), add a line to IMPLS below; its binary
# must accept: <model> -m test -f <input> -o <output> -n <steps> -t <temp> -z <tokenizer>.
set -euo pipefail
export LC_ALL=C   # decimal point, not comma, in printf/awk

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
INPUT="${INPUT:-input/12.txt}"
STEPS="${STEPS:-256}"
CPU_THREADS="${NUM_THREADS:-6}"   # fastest on the i5-14400 (6 P-cores), see README
MODELS="${MODELS:-$(ls model/*.bin 2>/dev/null | xargs -n1 basename | sed 's/\.bin$//' | sort -V)}"
TOKENIZER="$ROOT/tokenizer.bin"
OUT="$ROOT/profiling"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
HOST="$(hostname)"

# name | command prefix (model path and test-mode args are appended)
IMPLS=(
  "cpu_1t|$ROOT/bin/runtest"
  "cpu_omp|env OMP_NUM_THREADS=$CPU_THREADS $ROOT/bin/runtest_omp"
  "gpu_naive|$ROOT/bin/runcu"
)
REFERENCE=cpu_1t   # -O3, no fast-math: numerical reference for the correctness check

echo "== build"
make bin/runtest bin/runtest_omp bin/runcu >/dev/null

run_impl() {  # name cmd model temp outfile -> prints RESULT line
  local name="$1" cmd="$2" model="$3" temp="$4" outfile="$5"
  local extra=""
  [[ "$name" == gpu_* ]] && extra="-m test"
  $cmd "model/$model.bin" $extra -f "$INPUT" -o "$outfile" -n "$STEPS" -t "$temp" -z "$TOKENIZER" \
    | grep '^RESULT' | sed "s/impl=[^ ]*/impl=$name/"
}

CSV="$OUT/bench_$HOST.csv"
echo "impl,model,reqs,steps,fwd_tokens,gen_tokens,elapsed_s,fwd_tok_s,gen_tok_s" > "$CSV"
declare -A TPS

echo "== throughput (input=$INPUT, steps=$STEPS, temperature=1.0, top-p=0.9)"
for model in $MODELS; do
  for spec in "${IMPLS[@]}"; do
    name="${spec%%|*}"; cmd="${spec#*|}"
    line="$(run_impl "$name" "$cmd" "$model" 1.0 "$TMP/$name-$model.txt")"
    echo "  $line"
    # RESULT impl=.. model=.. reqs=.. -> csv row
    echo "$line" | sed 's/^RESULT //; s/[a-z_]*=//g; s/ /,/g; s/\.bin,/,/' >> "$CSV"
    TPS["$name|$model"]="$(echo "$line" | grep -o 'fwd_tok_s=[0-9.]*' | cut -d= -f2)"
  done
done

echo "== correctness (greedy, generated text vs $REFERENCE)"
declare -A MATCH
for model in $MODELS; do
  ref_cmd=""; for spec in "${IMPLS[@]}"; do [[ "${spec%%|*}" == "$REFERENCE" ]] && ref_cmd="${spec#*|}"; done
  run_impl "$REFERENCE" "$ref_cmd" "$model" 0 "$TMP/greedy-ref-$model.txt" >/dev/null
  for spec in "${IMPLS[@]}"; do
    name="${spec%%|*}"; cmd="${spec#*|}"
    [[ "$name" == "$REFERENCE" ]] && continue
    run_impl "$name" "$cmd" "$model" 0 "$TMP/greedy-$name-$model.txt" >/dev/null
    MATCH["$name|$model"]="$(python3 -I - "$TMP/greedy-ref-$model.txt" "$TMP/greedy-$name-$model.txt" <<'PY'
import sys
ref = open(sys.argv[1]).read().splitlines()[1:]
out = open(sys.argv[2]).read().splitlines()[1:]
same = sum(a == b for a, b in zip(ref, out))
diverge = []
for a, b in zip(ref, out):
    if a != b:
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        diverge.append(n)
msg = f"{same}/{len(ref)} identical"
if diverge:
    msg += f"; others first differ at char {min(diverge)}..{max(diverge)}"
print(msg)
PY
)"
    echo "  $name $model: ${MATCH["$name|$model"]}"
  done
done

MD="$OUT/bench_$HOST.md"
{
  echo "# Benchmark on $HOST"
  echo
  echo "- Date: $(date -u +%Y-%m-%d)"
  echo "- CPU: $(lscpu | sed -n 's/^Model name: *//p'), cpu_omp uses $CPU_THREADS threads"
  echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
  echo "- Workload: \`$INPUT\`, up to $STEPS positions per request, temperature 1.0, top-p 0.9, seed 314028"
  echo "- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded"
  echo
  printf "| model |"; for spec in "${IMPLS[@]}"; do printf " %s tok/s |" "${spec%%|*}"; done
  printf " gpu_naive vs cpu_1t | gpu_naive vs cpu_omp |\n|---|"
  for spec in "${IMPLS[@]}"; do printf -- "---|"; done; printf -- "---|---|\n"
  for model in $MODELS; do
    printf "| %s |" "$model"
    for spec in "${IMPLS[@]}"; do printf " %.1f |" "${TPS["${spec%%|*}|$model"]}"; done
    awk -v g="${TPS["gpu_naive|$model"]}" -v c1="${TPS["cpu_1t|$model"]}" -v co="${TPS["cpu_omp|$model"]}" \
      'BEGIN { printf " %.2fx | %.2fx |\n", g / c1, g / co }'
  done
  echo
  echo "Correctness (greedy decoding, generated text compared with $REFERENCE):"
  echo
  for key in "${!MATCH[@]}"; do echo "- ${key%%|*} on ${key#*|}: ${MATCH[$key]}"; done | sort
} > "$MD"
echo "== wrote $CSV and $MD"
cat "$MD"
