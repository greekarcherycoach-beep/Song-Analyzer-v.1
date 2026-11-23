//! satcore: native analysis core used by the Tauri app (and testable on its own).
pub mod audio;
pub mod dsp;
pub mod features;
pub mod key;
pub mod maest;
pub mod scan;
pub mod tags;
pub mod tempo;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Default, Clone)]
pub struct Models {
    pub key: Option<PathBuf>,
    pub tempo: Option<PathBuf>,
    pub maest: Option<PathBuf>,
}

/// What to compute for a file.
#[derive(Deserialize, Default, Debug, Clone)]
pub struct Req {
    #[serde(default)]
    pub key: bool,
    #[serde(default)]
    pub tempo: bool,
    #[serde(default)]
    pub genre: bool,
    #[serde(default)]
    pub feats: bool,
    #[serde(default)]
    pub nwin: usize,
}

#[derive(Serialize, Default, Debug)]
pub struct Analysis {
    pub key: Option<key::KeyResult>,
    pub tempo: Option<tempo::TempoResult>,
    pub genre: Option<Vec<maest::Score>>,
    pub feats: Option<features::Features>,
    pub errors: Vec<String>,
}

/// Decode once, then run the requested engines. A failure of one engine does not hide the others' results.
pub fn analyze_bytes(models: &Models, req: &Req, bytes: Vec<u8>, ext: Option<&str>) -> Analysis {
    let mut out = Analysis::default();
    let d = match audio::decode_full(bytes, ext) {
        Ok(d) => d,
        Err(e) => {
            out.errors.push(format!("decode: {e}"));
            return out;
        }
    };
    let dur = d.y.len() as f64 / d.sr as f64;
    let mut y44 = audio::resample(&d.y, d.sr, 44100);
    drop(d.y);
    if req.genre || req.feats {
        let y16 = audio::resample(&y44, 44100, maest::SR as u32);
        if req.feats {
            out.feats = Some(features::analyze(&y16, maest::SR as f64, d.sr, d.ch, dur));
        }
        if req.genre {
            match &models.maest {
                None => out.errors.push("genre: model maest.onnx not found".into()),
                Some(m) => match maest::classify(m, &y16, req.nwin) {
                    Ok(r) => out.genre = Some(r),
                    Err(e) => out.errors.push(format!("genre: {e}")),
                },
            }
        }
    }
    if req.key || req.tempo {
        audio::normalize(&mut y44);
        if req.key {
            match &models.key {
                None => out.errors.push("key: model keynet.onnx not found".into()),
                Some(m) => match key::detect(m, &y44) {
                    Ok(r) => out.key = Some(r),
                    Err(e) => out.errors.push(format!("key: {e}")),
                },
            }
        }
        if req.tempo {
            match &models.tempo {
                None => out.errors.push("tempo: model deeprhythm.onnx not found".into()),
                Some(m) => match tempo::detect(m, &y44) {
                    Ok(r) => out.tempo = Some(r),
                    Err(e) => out.errors.push(format!("tempo: {e}")),
                },
            }
        }
    }
    out
}

pub fn analyze_path(models: &Models, req: &Req, path: &Path) -> Analysis {
    match std::fs::read(path) {
        Ok(b) => analyze_bytes(models, req, b, path.extension().and_then(|e| e.to_str())),
        Err(e) => Analysis { errors: vec![format!("{}: {e}", path.display())], ..Default::default() },
    }
}
