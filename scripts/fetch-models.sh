#!/usr/bin/env sh
# Downloads the ONNX models into src-tauri/models.
#   ./scripts/fetch-models.sh            OpenKeyScan KeyNet + DeepRhythm (small, MIT / AGPL)
#   ./scripts/fetch-models.sh --maest    also the MAEST genre model as maest.onnx (~350 MB, check its licence first!)
set -e
BASE=https://raw.githubusercontent.com/fdrv161/openkeyscan-reaper/main/models
MAEST_URL=${MAEST_URL:-https://huggingface.co/Xenova/discogs-maest-30s-pw-73e-ts/resolve/main/onnx/model.onnx}
mkdir -p src-tauri/models
for f in keynet.onnx deeprhythm.onnx; do curl -L --fail -o "src-tauri/models/$f" "$BASE/$f"; done
if [ "$1" = "--maest" ]; then curl -L --fail -o src-tauri/models/maest.onnx "$MAEST_URL"; fi
