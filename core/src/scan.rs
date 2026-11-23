//! Recursive folder scan for audio files, reading the current Genre tag of each.
use crate::tags;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;

pub const AUDIO_EXT: [&str; 11] = ["mp3", "flac", "m4a", "mp4", "aac", "ogg", "oga", "opus", "wav", "aif", "aiff"];

#[derive(Serialize, Debug)]
pub struct Entry {
    pub path: String,
    pub rel: String,
    pub name: String,
    pub ext: String,
    pub size: u64,
    pub mtime: u64,
    pub genre: String,
}

pub fn scan_dir(root: &Path) -> Result<Vec<Entry>> {
    let base = root.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut out = Vec::new();
    walk(root, &base, &mut out);
    out.sort_by(|a, b| a.rel.to_lowercase().cmp(&b.rel.to_lowercase()));
    Ok(out)
}

fn walk(dir: &Path, rel: &str, out: &mut Vec<Entry>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let p = e.path();
        let rel2 = format!("{rel}/{name}");
        if p.is_dir() {
            walk(&p, &rel2, out);
        } else if let Some(ext) = p.extension().and_then(|x| x.to_str()).map(|x| x.to_lowercase()) {
            if AUDIO_EXT.contains(&ext.as_str()) {
                let genre = tags::read_genre(&p).unwrap_or_else(|_| "?".into());
                out.push(Entry { path: p.to_string_lossy().into_owned(), rel: rel2, name, ext, size: e.metadata().map(|m| m.len()).unwrap_or(0), mtime: e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as u64).unwrap_or(0), genre });
            }
        }
    }
}
