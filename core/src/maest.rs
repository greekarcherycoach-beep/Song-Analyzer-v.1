//! Genre / style classification with Essentia MAEST (ONNX, run with tract).
//! Feature extraction is a port of MAEST's `feature_extraction_maest.py` (and of the UI's JavaScript version).
use anyhow::{anyhow, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;
use std::{
    f64::consts::PI,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};
use tract_onnx::prelude::*;

pub const SR: usize = 16000;
const WIN: usize = 30 * SR;
const NFFT: usize = 512;
const HOP: usize = 256;
const MELS: usize = 96;
const MAXLEN: usize = 1876;
const MEAN: f64 = 2.06755686098554;
const STD: f64 = 1.268292820667291;

const LABELS_400: &str = include_str!("../data/discogs400.txt");
const LABELS_519: &str = include_str!("../data/discogs519.txt");

#[derive(Serialize, Debug, Clone)]
pub struct Score {
    pub label: String,
    pub score: f32,
}

fn sl_hz2mel(f: f64) -> f64 {
    if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + (f / 1000.0).ln() * (27.0 / 6.4f64.ln()) }
}
fn sl_mel2hz(m: f64) -> f64 {
    if m < 15.0 { 200.0 * m / 3.0 } else { 1000.0 * ((27.0 / 6.4f64.ln()) * (m - 15.0)).exp() }
}

fn filterbank() -> &'static Vec<Vec<f64>> {
    static FB: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    FB.get_or_init(|| {
        let nb = NFFT / 2 + 1;
        let (m0, m1) = (sl_hz2mel(0.0), sl_hz2mel(SR as f64 / 2.0));
        let ff: Vec<f64> = (0..MELS + 2).map(|i| sl_mel2hz(m0 + (m1 - m0) * i as f64 / (MELS + 1) as f64)).collect();
        (0..MELS)
            .map(|m| {
                let enorm = 2.0 / (ff[m + 2] - ff[m]);
                (0..nb)
                    .map(|k| {
                        let f = k as f64 * (SR as f64 / 2.0) / (nb - 1) as f64;
                        let down = -(ff[m] - f) / (ff[m + 1] - ff[m]);
                        let up = (ff[m + 2] - f) / (ff[m + 2] - ff[m + 1]);
                        down.min(up).max(0.0) * enorm
                    })
                    .collect()
            })
            .collect()
    })
}

/// Log-mel spectrogram, layout [MAXLEN][MELS], normalised like MAEST expects. `audio`: mono 16 kHz.
pub fn features(audio: &[f32]) -> Vec<f32> {
    let fb = filterbank();
    let half = NFFT / 2;
    let nb = half + 1;
    let n_frames = 1 + audio.len() / HOP;
    let used = n_frames.min(MAXLEN);
    let win: Vec<f64> = (0..NFFT).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / NFFT as f64).cos()).collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(NFFT);
    let mut buf = vec![Complex::new(0f64, 0f64); NFFT];
    let pad = ((0.0 - MEAN) / (STD * 2.0)) as f32;
    let mut out = vec![pad; MAXLEN * MELS];
    let mut pw = vec![0f64; nb];
    for t in 0..used {
        let start = (t * HOP) as i64 - half as i64;
        for i in 0..NFFT {
            let idx = start + i as i64;
            let v = if idx >= 0 && (idx as usize) < audio.len() { audio[idx as usize] as f64 } else { 0.0 };
            buf[i] = Complex::new(v * win[i], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..nb {
            pw[k] = buf[k].norm_sqr();
        }
        for m in 0..MELS {
            let s: f64 = fb[m].iter().zip(&pw).map(|(a, b)| a * b).sum();
            out[t * MELS + m] = (((1.0 + s.max(1e-30) * 10000.0).log10() - MEAN) / (STD * 2.0)) as f32;
        }
    }
    out
}

/// 30 s windows: the middle one, or three (20% / 50% / 80%) of the track.
pub fn segments(a: &[f32], nwin: usize) -> Vec<&[f32]> {
    if a.len() <= WIN {
        return vec![a];
    }
    let fr: &[f64] = if nwin <= 1 { &[0.5] } else { &[0.2, 0.5, 0.8] };
    fr.iter()
        .map(|f| {
            let s = ((f * a.len() as f64 - WIN as f64 / 2.0).max(0.0).min((a.len() - WIN) as f64)).floor() as usize;
            &a[s..s + WIN]
        })
        .collect()
}

type Runnable = std::sync::Arc<TypedRunnableModel<TypedModel>>;
static CACHE: Mutex<Option<(PathBuf, Runnable)>> = Mutex::new(None);

fn runnable(model: &Path) -> Result<Runnable> {
    let mut g = CACHE.lock().map_err(|_| anyhow!("model cache poisoned"))?;
    if let Some((p, r)) = g.as_ref() {
        if p == model {
            return Ok(r.clone());
        }
    }
    let r = tract_onnx::onnx()
        .model_for_path(model)?
        .with_input_fact(0, f32::fact([1usize, MAXLEN, MELS]).into())?
        .into_optimized()?
        .into_runnable()?;
    let r = std::sync::Arc::new(r);
    *g = Some((model.to_path_buf(), r.clone()));
    Ok(r)
}

fn labels(n: usize) -> Vec<String> {
    let src = if n == 519 { LABELS_519 } else { LABELS_400 };
    src.lines().map(|s| s.to_string()).collect()
}

/// Averaged softmax over the windows; returns the top 10 labels ("Genre---Style").
pub fn classify(model: &Path, y16: &[f32], nwin: usize) -> Result<Vec<Score>> {
    let run = runnable(model)?;
    let segs = segments(y16, nwin);
    let mut acc: Vec<f64> = Vec::new();
    for s in &segs {
        let input = Tensor::from_shape(&[1, MAXLEN, MELS], &features(s))?;
        let out = run.run(tvec!(input.into()))?;
        let lg: Vec<f32> = out[0].to_array_view::<f32>()?.iter().cloned().collect();
        if acc.is_empty() {
            acc = vec![0.0; lg.len()];
        }
        let m = lg.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f64> = lg.iter().map(|v| ((v - m) as f64).exp()).collect();
        let z: f64 = e.iter().sum();
        for (i, v) in e.iter().enumerate() {
            acc[i] += v / z / segs.len() as f64;
        }
    }
    let names = labels(acc.len());
    let mut idx: Vec<usize> = (0..acc.len()).collect();
    idx.sort_by(|a, b| acc[*b].partial_cmp(&acc[*a]).unwrap());
    Ok(idx.into_iter().take(10).map(|i| Score { label: names.get(i).cloned().unwrap_or_else(|| i.to_string()), score: acc[i] as f32 }).collect())
}
