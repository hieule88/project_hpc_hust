#!/usr/bin/env bash
# End-to-end benchmark + correctness check: CPU reference vs GPU implementations.
#
#   ./bench.sh                    # all models found in model/, input/12.txt, 256 steps
#   MODELS="stories110M" STEPS=128 INPUT=input/32.txt ./bench.sh      (or: make bench)
#   ONLY='^gpu' INPUT=input/128.txt ./bench.sh   # only implementations whose name matches
#
# Every implementation runs the same workload (same prompts, sampler, seed, stop rule and
# output format, see harness.h) and prints one RESULT line. This script:
#   1. builds the CPU and GPU binaries of this branch, and the naive GPU version from the
#      `main` branch in a git worktree (.naive/), so naive and optimized run side by side
#   2. throughput: runs each implementation on each model (temperature 1.0, top-p 0.9)
#   3. correctness: runs the CPU reference and each other version greedily (-t 0) and
#      compares the generated text request by request
#   4. writes profiling/bench_<host>_in<input>.csv and .md
# To add an implementation, add a line to IMPLS: "name|binary|extra args"; the binary is
# called as: <binary> <model> <extra args> -f <input> -o <output> -n <steps> -t <temp> -z <tokenizer>
set -euo pipefail
export LC_ALL=C   # decimal point, not comma, in printf/awk

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
INPUT="${INPUT:-input/12.txt}"
STEPS="${STEPS:-256}"
BATCH="${BATCH:-10}"
CPU_THREADS="${NUM_THREADS:-6}"   # fastest on the i5-14400 (6 P-cores), see README
NAIVE_REF="${NAIVE_REF:-main}"    # branch holding the naive GPU version
ONLY="${ONLY:-.}"
MODELS="${MODELS:-$(ls model/*.bin 2>/dev/null | xargs -n1 basename | sed 's/\.bin$//' | sort -V)}"
TOKENIZER="$ROOT/tokenizer.bin"
OUT="$ROOT/profiling"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
HOST="$(hostname)"
OPT="gpu_optim_b$BATCH"

ALL_IMPLS=(
  "cpu_1t|$ROOT/bin/runtest|"
  "cpu_omp|env OMP_NUM_THREADS=$CPU_THREADS $ROOT/bin/runtest_omp|"
  "gpu_naive|$ROOT/.naive/bin/runcu|-m test"
  "gpu_optim_b1|$ROOT/bin/runcu|-m test -b 1"
  "$OPT|$ROOT/bin/runcu|-m test -b $BATCH"
  "gpu_pp_4x$BATCH|$ROOT/bin/runpp|-m test -b $BATCH"
  "gpu_paged_b$BATCH|$ROOT/bin/runpgattn|-m test -b $BATCH"
)
IMPLS=()
for spec in "${ALL_IMPLS[@]}"; do [[ "${spec%%|*}" =~ $ONLY ]] && IMPLS+=("$spec"); done
REFERENCE=cpu_1t   # -O3, no fast-math: numerical reference for the correctness check

echo "== build"
make bin/runtest bin/runtest_omp bin/runcu bin/runpp bin/runpgattn >/dev/null
if git rev-parse -q --verify "$NAIVE_REF" >/dev/null; then
  [ -d .naive ] || git worktree add -q --detach .naive "$NAIVE_REF"
  git -C .naive checkout -q --detach "$NAIVE_REF"
  make -C .naive bin/runcu >/dev/null
else
  echo "  (no branch $NAIVE_REF: skipping gpu_naive)"
  IMPLS=("${IMPLS[@]/gpu_naive|*/}")
fi

run_impl() {  # spec model temp outfile -> prints RESULT line
  local name cmd args
  IFS='|' read -r name cmd args <<< "$1"
  $cmd "model/$2.bin" $args -f "$INPUT" -o "$4" -n "$STEPS" -t "$3" -z "$TOKENIZER" \
    | grep '^RESULT' | sed "s/impl=[^ ]*/impl=$name/"
}

TAG="$(basename "$INPUT" .txt)"
CSV="$OUT/bench_${HOST}_in${TAG}.csv"
echo "impl,model,reqs,steps,fwd_tokens,gen_tokens,elapsed_s,fwd_tok_s,gen_tok_s" > "$CSV"
declare -A TPS

echo "== throughput (input=$INPUT, steps=$STEPS, temperature=1.0, top-p=0.9)"
for model in $MODELS; do
  for spec in "${IMPLS[@]}"; do
    [ -z "$spec" ] && continue
    name="${spec%%|*}"
    line="$(run_impl "$spec" "$model" 1.0 "$TMP/$name-$model.txt")"
    echo "  $line"
    echo "$line" | sed 's/^RESULT //; s/[a-z_]*=//g; s/ /,/g; s/\.bin,/,/' >> "$CSV"
    TPS["$name|$model"]="$(echo "$line" | grep -o 'fwd_tok_s=[0-9.]*' | cut -d= -f2)"
  done
done

echo "== correctness (greedy, generated text vs $REFERENCE)"
declare -A MATCH
for model in $MODELS; do
  ref_spec="$REFERENCE|$ROOT/bin/runtest|"
  run_impl "$ref_spec" "$model" 0 "$TMP/greedy-ref-$model.txt" >/dev/null
  for spec in "${IMPLS[@]}"; do
    [ -z "$spec" ] && continue
    name="${spec%%|*}"
    [[ "$name" == "$REFERENCE" ]] && continue
    run_impl "$spec" "$model" 0 "$TMP/greedy-$name-$model.txt" >/dev/null
    MATCH["$name|$model"]="$(python3 -I - "$TMP/greedy-ref-$model.txt" "$TMP/greedy-$name-$model.txt" <<'PY'
import sys
ref = open(sys.argv[1]).read().splitlines()[1:]
out = open(sys.argv[2]).read().splitlines()[1:]
same = sum(a == b for a, b in zip(ref, out))
diverge = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
           for a, b in zip(ref, out) if a != b]
msg = f"{same}/{len(ref)} identical"
if diverge:
    msg += f"; others first differ at char {min(diverge)}..{max(diverge)}"
print(msg)
PY
)"
    echo "  $name $model: ${MATCH["$name|$model"]}"
  done
done

ratio() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { if (a > 0 && b > 0) printf "%.1fx", a / b; else printf "n/a" }'; }
MD="$OUT/bench_${HOST}_in${TAG}.md"
{
  echo "# Benchmark on $HOST"
  echo
  echo "- Date: $(date -u +%Y-%m-%d)"
  echo "- CPU: $(lscpu | sed -n 's/^Model name: *//p'), cpu_omp uses $CPU_THREADS threads"
  echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
  echo "- Workload: \`$INPUT\`, up to $STEPS positions per request, temperature 1.0, top-p 0.9, seed 314028"
  echo "- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded; 1 run per cell"
  echo
  printf "| model |"
  for spec in "${IMPLS[@]}"; do [ -n "$spec" ] && printf " %s |" "${spec%%|*}"; done
  printf " %s vs cpu_omp | %s vs gpu_naive |\n|---|" "$OPT" "$OPT"
  for spec in "${IMPLS[@]}"; do [ -n "$spec" ] && printf -- "---|"; done; printf -- "---|---|\n"
  for model in $MODELS; do
    printf "| %s |" "$model"
    for spec in "${IMPLS[@]}"; do [ -n "$spec" ] && printf " %.1f |" "${TPS["${spec%%|*}|$model"]:-0}"; done
    printf " %s | %s |\n" "$(ratio "${TPS["$OPT|$model"]:-}" "${TPS["cpu_omp|$model"]:-}")" \
                          "$(ratio "${TPS["$OPT|$model"]:-}" "${TPS["gpu_naive|$model"]:-}")"
  done
  echo
  echo "Correctness (greedy decoding, generated text compared with $REFERENCE):"
  echo
  for key in "${!MATCH[@]}"; do echo "- ${key%%|*} on ${key#*|}: ${MATCH[$key]}"; done | sort
} > "$MD"
echo "== wrote $CSV and $MD"
cat "$MD"
