# Song Analyzer · Genre Tagger

Desktop app (Windows 10/11, Tauri 2 + Rust) that scans music folders and computes for every track:

* **Genre / style** – Essentia MAEST, **native Rust** when `maest.onnx` is present (otherwise automatic fallback to Transformers.js in the WebView)
* **Key** (Camelot code) – OpenKeyScan KeyNet CNN, **native Rust** (symphonia decode → log-CQT → tract/ONNX)
* **Tempo** – DeepRhythm CNN + onset-autocorrelation refinement, **native Rust**
* Energy, danceability, mood, loudness, dynamic range, duration – **native Rust** (`core/src/features.rs`)

Plus: compare tracks, build harmonic-mix playlists, edit/bulk-edit tags, write **Genre / BPM / Key** into MP3 (ID3v2) and
FLAC/OGG/Opus (Vorbis comments), M4A (MP4) and WAV/AIFF with one-time `.bak` backups (**native Rust, `lofty`**), export CSV/JSON, results cache.

In the desktop app the folder picker is native (full paths): scanning, tag reading and writing, and restore-from-`.bak` all go
through Rust. In a plain browser the same page still works through the File System Access API (MP3/FLAC only).

## Download

From **Releases**: `…-setup.exe` (installer, per-user or all-users) or `SongAnalyzerTagger-portable-win-x64.zip`.

**Portable:** unzip anywhere and run `song-analyzer-tagger.exe`. The file `portable.txt` next to the exe makes the app keep
everything in `./data` and look for models in `./models`. Delete `portable.txt` to use the normal profile folder instead.

Requirements: Windows 10 or later with the **WebView2 runtime** (the installer downloads it if missing; the portable build
needs it already installed, which is the case on almost every up-to-date Windows 10/11).

## Build from source

```powershell
# prerequisites: Rust (stable, MSVC), Visual Studio Build Tools (C++), WebView2
cargo install tauri-cli --version "^2" --locked
./scripts/fetch-models.ps1        # downloads keynet.onnx + deeprhythm.onnx into src-tauri/models
cargo test -p satcore --release   # core tests (key detection on a synthetic signal)
cargo tauri dev                   # run
cargo tauri build                 # exe + NSIS installer -> target/release/bundle/nsis
```

The core crate also runs without Tauri (any OS): `cargo run -p satcore --release --example key_cli -- src-tauri/models/keynet.onnx song.mp3`

## Releasing

Push a tag (`git tag v0.1.0 && git push --tags`). The workflow in `.github/workflows/release.yml` builds on `windows-latest`,
runs the core tests, and attaches the installer and the portable zip to the GitHub Release.

## Layout

```
core/        satcore: decoding, resampling, KeyNet (key), DeepRhythm (tempo), MAEST (genre), DSP features, lofty tags, folder scan — pure Rust
src-tauri/   Tauri shell: commands (analyze, scan_dir, read_file, write_tags, restore_backup…), portable-mode logic, bundling config
ui/          the interface (single HTML file; uses the native key engine when running inside Tauri)
```

## Status

* Everything runs in Rust (`core/`): folder scan, tag read/write/restore, decoding, resampling, key, tempo, energy/danceability/loudness, MAEST features and inference. The WebView only draws the interface.
* **MAEST model:** not bundled by default (large, and its licence is usually non-commercial). Put `maest.onnx` into `models/` (next to the exe) or `data/models/`:
  `./scripts/fetch-models.ps1 -Maest` downloads the Xenova ONNX export. Without it the app falls back to Transformers.js in the WebView (needs internet once).
  Run the release workflow manually with *bundle_maest* to include it in the build.
* **Verified here:** the Rust features are bit-identical to the JavaScript originals (energy/danceability/… and the MAEST mel-spectrogram), key/tempo/tags have tests, MAEST inference plumbing is tested with a small stand-in model.
  **Not verified:** MAEST inference with the real model in `tract` (transformer operator support) — if it fails, the app logs it and uses the WebView fallback automatically.
* Tests: `cargo test -p satcore` (needs `scripts/fetch-models.*` for the key/tempo tests).
* Unsigned binaries trigger Windows SmartScreen; sign with a code-signing certificate for public distribution.

## License

AGPL-3.0 (see `LICENSE` and `THIRD_PARTY_NOTICES.md`).
