//! Tag reading/writing (Genre, BPM, Key) with lofty: MP3 (ID3v2), FLAC/OGG/Opus (Vorbis comments), M4A (MP4), WAV/AIFF.
use anyhow::{anyhow, Result};
use lofty::{config::WriteOptions, id3::v1::GENRES, prelude::*, tag::{ItemKey, Tag}};
use std::path::{Path, PathBuf};

fn bak_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".bak");
    PathBuf::from(s)
}

/// ID3 numeric genres such as "(17)" or "17" -> "Rock".
fn decode_genre(g: &str) -> String {
    let t = g.trim();
    let digits: String = t.trim_start_matches('(').chars().take_while(|c| c.is_ascii_digit()).collect();
    let numeric = !digits.is_empty() && (t.starts_with('(') || digits.len() == t.len());
    if numeric {
        if let Some(n) = GENRES.get(digits.parse::<usize>().unwrap_or(usize::MAX)) {
            let rest = t.split(')').nth(1).unwrap_or("").trim();
            return if t.starts_with('(') && !rest.is_empty() { rest.to_string() } else { n.to_string() };
        }
    }
    t.to_string()
}

pub fn read_genre(path: &Path) -> Result<String> {
    let tf = lofty::read_from_path(path)?;
    let tag = tf.primary_tag().or_else(|| tf.first_tag());
    Ok(tag.and_then(|t| t.genre().map(|g| decode_genre(&g))).unwrap_or_default())
}

pub fn has_backup(path: &Path) -> bool {
    bak_path(path).exists()
}

/// Writes any of Genre / BPM / Key. The `.bak` copy is made once and keeps the original file.
/// Returns the genre read back from the file (for verification).
pub fn write_tags(path: &Path, genre: Option<&str>, bpm: Option<f64>, key: Option<&str>, backup: bool) -> Result<String> {
    if backup && !has_backup(path) {
        std::fs::copy(path, bak_path(path))?;
    }
    let mut tf = lofty::read_from_path(path)?;
    if tf.primary_tag().is_none() {
        let tt = tf.primary_tag_type();
        tf.insert_tag(Tag::new(tt));
    }
    {
        let tag = tf.primary_tag_mut().ok_or_else(|| anyhow!("cannot create a tag for this format"))?;
        if let Some(g) = genre.filter(|g| !g.trim().is_empty()) {
            tag.set_genre(g.trim().to_string());
        }
        if let Some(b) = bpm {
            tag.insert_text(ItemKey::Bpm, format!("{}", b.round() as i64));
        }
        if let Some(k) = key.filter(|k| !k.is_empty()) {
            tag.insert_text(ItemKey::InitialKey, k.to_string());
        }
    }
    tf.save_to_path(path, WriteOptions::default())?;
    read_genre(path)
}

/// Copies `<file>.bak` over the file. Returns false when there is no backup.
pub fn restore_backup(path: &Path) -> Result<bool> {
    let b = bak_path(path);
    if !b.exists() {
        return Ok(false);
    }
    std::fs::copy(b, path)?;
    Ok(true)
}
