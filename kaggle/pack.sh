#!/usr/bin/env bash
# Package the project into two Kaggle objects and push them.
#
#   ./kaggle/pack.sh              # build kaggle/upload + metadata, do not push
#   ./kaggle/pack.sh --push       # create/version the dataset, then run the kernel
#
# The dataset carries the code; the kernel trains and writes the generated firmware into
# /kaggle/working. Both are private by default.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/kaggle/upload"
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p' | tr -d '[:space:]')"
[ -n "$USER_NAME" ] || { echo "kaggle CLI not authenticated (kaggle config view)" >&2; exit 1; }

rm -rf "$UP"
mkdir -p "$UP"
cp "$ROOT"/{synth.py,model.py,train.py,export_c.py,compare.py,probe.py,data_real.py,data_stl.py,data_trash.py,requirements.txt} "$UP"/

cat > "$UP/dataset-metadata.json" <<EOF
{
  "title": "prh-tiny-code",
  "id": "$USER_NAME/prh-tiny-code",
  "licenses": [{"name": "MIT"}],
  "isPrivate": true
}
EOF

cat > "$ROOT/kaggle/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/prh-tiny-train",
  "title": "prh-tiny-train",
  "code_file": "train_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["$USER_NAME/prh-tiny-code"],
  "competition_sources": [],
  "model_sources": [],
  "kernel_sources": []
}
EOF

echo "packed $UP and kaggle/kernel-metadata.json for $USER_NAME"

if [ "${1:-}" = "--push" ]; then
  if kaggle datasets list -m -s prh-tiny-code 2>/dev/null | grep -q "prh-tiny-code"; then
    kaggle datasets version -p "$UP" -r zip -m "update $(date +%Y-%m-%dT%H:%M)"
  else
    kaggle datasets create -p "$UP" -r zip
  fi
  kaggle kernels push -p "$ROOT/kaggle"
  echo "pushed. status: kaggle kernels status $USER_NAME/prh-tiny-train"
fi
