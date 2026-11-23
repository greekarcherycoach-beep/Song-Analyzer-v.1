# Downloads the ONNX models into src-tauri\models.
#   ./scripts/fetch-models.ps1           OpenKeyScan KeyNet + DeepRhythm (small)
#   ./scripts/fetch-models.ps1 -Maest    also the MAEST genre model as maest.onnx (~350 MB, check its licence first!)
param([switch]$Maest, [string]$MaestUrl = "https://huggingface.co/Xenova/discogs-maest-30s-pw-73e-ts/resolve/main/onnx/model.onnx")
$ErrorActionPreference = "Stop"
$base = "https://raw.githubusercontent.com/fdrv161/openkeyscan-reaper/main/models"
New-Item -ItemType Directory -Force src-tauri\models | Out-Null
foreach ($f in "keynet.onnx", "deeprhythm.onnx") { Invoke-WebRequest "$base/$f" -OutFile "src-tauri\models\$f" }
if ($Maest) { Invoke-WebRequest $MaestUrl -OutFile "src-tauri\models\maest.onnx" }
