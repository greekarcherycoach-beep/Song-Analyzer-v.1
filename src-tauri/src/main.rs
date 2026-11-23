// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::{
    ipc::{InvokeBody, Request, Response},
    AppHandle, Manager,
};

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_default()
}

/// Portable mode: a file named `portable.txt` next to the .exe keeps all data in `./data`.
fn is_portable() -> bool {
    exe_dir().join("portable.txt").exists()
}

fn data_dir(app: &AppHandle) -> PathBuf {
    if is_portable() {
        exe_dir().join("data")
    } else {
        app.path().app_data_dir().unwrap_or_else(|_| exe_dir().join("data"))
    }
}

/// Model lookup order: <data>/models, <exe>/models, bundled resources.
fn find_model(app: &AppHandle, name: &str) -> Result<PathBuf, String> {
    let mut c = vec![data_dir(app).join("models").join(name), exe_dir().join("models").join(name)];
    if let Ok(r) = app.path().resource_dir() {
        c.push(r.join("models").join(name));
    }
    c.into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| format!("model {name} not found (looked in data/models, models/ and bundled resources)"))
}

/// Run blocking work off the UI thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

#[derive(Serialize)]
struct Info {
    native: bool,
    portable: bool,
    data_dir: String,
    key_model: bool,
    tempo_model: bool,
    genre_model: bool,
}

#[tauri::command]
fn app_info(app: AppHandle) -> Info {
    Info {
        native: true,
        portable: is_portable(),
        data_dir: data_dir(&app).to_string_lossy().into_owned(),
        key_model: find_model(&app, "keynet.onnx").is_ok(),
        tempo_model: find_model(&app, "deeprhythm.onnx").is_ok(),
        genre_model: find_model(&app, "maest.onnx").is_ok(),
    }
}

/// Recursive scan of a folder: audio files + their current Genre tag.
#[tauri::command]
async fn scan_dir(path: String) -> Result<Vec<satcore::scan::Entry>, String> {
    blocking(move || satcore::scan::scan_dir(Path::new(&path)).map_err(|e| e.to_string())).await
}

/// Raw file bytes (sent to the UI as an ArrayBuffer, no JSON overhead).
#[tauri::command]
async fn read_file(path: String) -> Result<Response, String> {
    blocking(move || std::fs::read(&path).map_err(|e| format!("{path}: {e}"))).await.map(Response::new)
}

fn models(app: &AppHandle) -> satcore::Models {
    satcore::Models {
        key: find_model(app, "keynet.onnx").ok(),
        tempo: find_model(app, "deeprhythm.onnx").ok(),
        maest: find_model(app, "maest.onnx").ok(),
    }
}

/// Everything in one decode (key, tempo, genre, energy/danceability/loudness…). Body = raw file bytes;
/// headers: `x-ext`, `x-key`, `x-tempo`, `x-genre`, `x-feats` ("1"), `x-nwin`.
#[tauri::command]
async fn analyze(app: AppHandle, request: Request<'_>) -> Result<satcore::Analysis, String> {
    let bytes = match request.body() {
        InvokeBody::Raw(b) => b.clone(),
        _ => return Err("expected raw bytes".into()),
    };
    let h = request.headers();
    let flag = |n: &str| h.get(n).and_then(|v| v.to_str().ok()) == Some("1");
    let req = satcore::Req {
        key: flag("x-key"),
        tempo: flag("x-tempo"),
        genre: flag("x-genre"),
        feats: flag("x-feats"),
        nwin: h.get("x-nwin").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(1),
    };
    let ext = h.get("x-ext").and_then(|v| v.to_str().ok()).map(|s| s.to_lowercase());
    let m = models(&app);
    blocking(move || Ok(satcore::analyze_bytes(&m, &req, bytes, ext.as_deref()))).await
}

/// Same as `analyze`, but the file is read by Rust (no bytes cross the IPC boundary).
#[tauri::command]
async fn analyze_path(app: AppHandle, path: String, req: satcore::Req) -> Result<satcore::Analysis, String> {
    let m = models(&app);
    blocking(move || Ok(satcore::analyze_path(&m, &req, Path::new(&path)))).await
}

#[tauri::command]
async fn read_genre(path: String) -> Result<String, String> {
    blocking(move || satcore::tags::read_genre(Path::new(&path)).map_err(|e| e.to_string())).await
}

/// Writes Genre and optionally BPM/Key; makes a one-time `.bak` copy when `backup` is true. Returns the genre read back.
#[tauri::command]
async fn write_tags(path: String, genre: Option<String>, bpm: Option<f64>, key: Option<String>, backup: bool) -> Result<String, String> {
    blocking(move || satcore::tags::write_tags(Path::new(&path), genre.as_deref(), bpm, key.as_deref(), backup).map_err(|e| e.to_string())).await
}

#[tauri::command]
fn has_backup(path: String) -> bool {
    satcore::tags::has_backup(Path::new(&path))
}

#[tauri::command]
async fn restore_backup(path: String) -> Result<bool, String> {
    blocking(move || satcore::tags::restore_backup(Path::new(&path)).map_err(|e| e.to_string())).await
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![app_info, scan_dir, read_file, analyze, analyze_path, read_genre, write_tags, has_backup, restore_backup])
        .run(tauri::generate_context!())
        .expect("error while running Song Analyzer Tagger");
}
