//! Decoding (symphonia) to mono f32 and resampling.
use anyhow::{anyhow, Result};
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SErr;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct Decoded {
    pub y: Vec<f32>,
    pub sr: u32,
    pub ch: u32,
}

/// Decode an in-memory audio file to (mono samples, sample rate). `ext` is only a probing hint.
pub fn decode_mono(bytes: Vec<u8>, ext: Option<&str>) -> Result<(Vec<f32>, u32)> {
    let d = decode_full(bytes, ext)?;
    Ok((d.y, d.sr))
}

/// Like `decode_mono`, also reporting the original channel count. Samples are not normalised.
pub fn decode_full(bytes: Vec<u8>, ext: Option<&str>) -> Result<Decoded> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    if let Some(e) = ext {
        if !e.is_empty() {
            hint.with_extension(e);
        }
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| anyhow!("unsupported or corrupt audio: {e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no audio track"))?;
    let id = track.id;
    let sr = track.codec_params.sample_rate.ok_or_else(|| anyhow!("unknown sample rate"))?;
    let mut dec = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut out: Vec<f32> = Vec::new();
    let mut nch = 1u32;
    let mut sb: Option<SampleBuffer<f32>> = None;
    loop {
        let pkt = match format.next_packet() {
            Ok(p) => p,
            Err(SErr::IoError(_)) | Err(SErr::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if pkt.track_id() != id {
            continue;
        }
        match dec.decode(&pkt) {
            Ok(buf) => {
                let spec = *buf.spec();
                let ch = spec.channels.count().max(1);
                nch = ch as u32;
                if sb.is_none() {
                    sb = Some(SampleBuffer::<f32>::new(buf.capacity() as u64, spec));
                }
                let s = sb.as_mut().unwrap();
                s.copy_interleaved_ref(buf);
                for fr in s.samples().chunks(ch) {
                    out.push(fr.iter().sum::<f32>() / ch as f32);
                }
            }
            Err(SErr::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }
    if out.is_empty() {
        return Err(anyhow!("no audio decoded"));
    }
    Ok(Decoded { y: out, sr, ch: nch })
}

/// Windowed-sinc resampler (Hann window, 16 zero crossings) with a pre-computed, interpolated kernel table.
pub fn resample(y: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || y.is_empty() {
        return y.to_vec();
    }
    const RES: usize = 256;
    let ratio = from as f64 / to as f64;
    let cutoff = (1.0 / ratio).min(1.0) * 0.97;
    let half = 16.0 / cutoff;
    let hw = half.ceil() as usize;
    let pi = std::f64::consts::PI;
    let tab: Vec<f32> = (0..=2 * hw * RES + 1)
        .map(|i| {
            let x = i as f64 / RES as f64 - hw as f64;
            if x.abs() >= half {
                return 0.0;
            }
            let s = if x.abs() < 1e-12 { 1.0 } else { (pi * cutoff * x).sin() / (pi * cutoff * x) };
            (s * (0.5 + 0.5 * (pi * x / half).cos()) * cutoff) as f32
        })
        .collect();
    let n_out = (y.len() as f64 / ratio).ceil() as usize;
    let mut out = vec![0f32; n_out];
    for (k, o) in out.iter_mut().enumerate() {
        let c = k as f64 * ratio;
        let a = (c - half).ceil().max(0.0) as usize;
        let b = ((c + half).floor() as usize).min(y.len() - 1);
        let mut acc = 0f32;
        for j in a..=b {
            let p = (j as f64 - c + hw as f64) * RES as f64;
            let i = p as usize;
            let f = (p - i as f64) as f32;
            acc += y[j] * (tab[i] + (tab[i + 1] - tab[i]) * f);
        }
        *o = acc;
    }
    out
}

/// Peak-normalise to 0.95 (as OpenKeyScan does).
pub fn normalize(y: &mut [f32]) {
    let pk = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    if pk > 0.0 {
        let g = 0.95 / pk;
        y.iter_mut().for_each(|v| *v *= g);
    }
}

/// Mono 44.1 kHz, peak-normalised.
pub fn load_44k(bytes: Vec<u8>, ext: Option<&str>) -> Result<Vec<f32>> {
    let (y, sr) = decode_mono(bytes, ext)?;
    let mut y = resample(&y, sr, 44100);
    normalize(&mut y);
    Ok(y)
}
