//! Musical key detection with the OpenKeyScan KeyNet CNN (ONNX, run with tract).
use crate::{audio, dsp};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::path::Path;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

pub const NAMES: [&str; 24] = [
    "Ab minor", "Eb minor", "Bb minor", "F minor", "C minor", "G minor", "D minor", "A minor", "E minor", "B minor", "F# minor", "C# minor",
    "B major", "F# major", "Db major", "Ab major", "Eb major", "Bb major", "F major", "C major", "G major", "D major", "A major", "E major",
];
const MAX_SECONDS: usize = 600;

#[derive(Serialize, Debug, Clone)]
pub struct KeyResult {
    pub cls: usize,
    pub conf: f32,
    pub name: String,
    pub camelot: String,
}

fn plan() -> &'static dsp::Plan {
    static P: OnceLock<dsp::Plan> = OnceLock::new();
    P.get_or_init(dsp::Plan::new)
}

/// `y44`: mono, 44.1 kHz, peak-normalised (see `audio::load_44k`).
pub fn detect(model: &Path, y44: &[f32]) -> Result<KeyResult> {
    let sr = dsp::SR as usize;
    if y44.len() < sr * 3 {
        return Err(anyhow!("track too short for key detection (< 3 s)"));
    }
    let start = (y44.len().saturating_sub(MAX_SECONDS * sr)) / 2;
    let part = &y44[start..(start + MAX_SECONDS * sr).min(y44.len())];
    let (spec, t) = plan().spec(part);
    let runnable = tract_onnx::onnx()
        .model_for_path(model)?
        .with_input_fact(0, f32::fact([1usize, 1, dsp::NBINS, t]).into())?
        .into_optimized()?
        .into_runnable()?;
    let input = Tensor::from_shape(&[1, 1, dsp::NBINS, t], &spec)?;
    let out = runnable.run(tvec!(input.into()))?;
    let lg = out[0].to_array_view::<f32>()?;
    let lg: Vec<f32> = lg.iter().cloned().collect();
    let (mut ci, mut mx) = (0usize, f32::MIN);
    for (i, v) in lg.iter().enumerate() {
        if *v > mx {
            mx = *v;
            ci = i;
        }
    }
    let z: f32 = lg.iter().map(|v| (v - mx).exp()).sum();
    Ok(KeyResult {
        cls: ci,
        conf: 1.0 / z,
        name: NAMES[ci].to_string(),
        camelot: format!("{}{}", ci % 12 + 1, if ci < 12 { 'A' } else { 'B' }),
    })
}

/// Convenience: raw file bytes -> key.
pub fn detect_bytes(model: &Path, bytes: Vec<u8>, ext: Option<&str>) -> Result<KeyResult> {
    detect(model, &audio::load_44k(bytes, ext)?)
}
