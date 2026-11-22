#!/usr/bin/env bash
# Download the TinyStories checkpoints from karpathy/tinyllamas into models/.
set -euo pipefail
cd "$(dirname "$0")"
for m in stories15M stories42M stories110M; do
  [ -f "$m.bin" ] || wget -q --show-progress "https://huggingface.co/karpathy/tinyllamas/resolve/main/$m.bin" -O "$m.bin"
done
ls -lh *.bin
