//! Tempo with DeepRhythm (ONNX, run with tract): HCQM features -> CNN (coarse BPM) -> onset-autocorrelation refinement.
//! Port of the okscan/deeprhythm reference implementation (AGPL-3.0, see THIRD_PARTY_NOTICES.md).
use crate::dsp;
use anyhow::{anyhow, Result};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::Serialize;
use std::{f64::consts::PI, path::Path, sync::{Arc, OnceLock}};
use tract_onnx::prelude::*;

const SR: usize = 22050;
const CLIP: usize = 176_400; // 8 s
const N: usize = 2048;
const HOP: usize = 512;
const NB: usize = 240;
const BANDS: usize = 8;
const MAXCLIPS: usize = 40;
const REFINE_S: usize = 240;

#[derive(Serialize, Debug, Clone)]
pub struct TempoResult {
    pub bpm: f64,
    pub conf: f32,
    pub coarse: f64,
}

struct Frame {
    i0: usize,
    i1: usize,
    re: Vec<Vec<f32>>,
    im: Vec<Vec<f32>>,
}
struct Harm {
    frames: Vec<Frame>,
    n_f: usize,
    scale: Vec<f64>,
}
struct Plan {
    bands: Vec<(usize, usize, f64)>,
    t: usize,
    harm: Vec<Harm>,
}

fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect()
}

fn plan() -> &'static Plan {
    static P: OnceLock<Plan> = OnceLock::new();
    P.get_or_init(|| {
        let nbin = N / 2 + 1;
        let mut lb: Vec<usize> = (0..=BANDS).map(|i| (10f64.powf(i as f64 * (nbin as f64).log10() / BANDS as f64) - 1.0).round_ties_even() as usize).collect();
        lb.sort();
        lb.dedup();
        let bands = (0..BANDS)
            .map(|i| {
                let s = lb[i];
                let e = if i < BANDS - 1 { lb[i + 1] } else { nbin };
                (s, e, 1.0 / (e - s) as f64)
            })
            .collect();
        let t = 1 + CLIP / HOP;
        let hop = CLIP / HOP;
        let srq = (CLIP / (HOP * 8)) as f64;
        let mut harm = Vec::new();
        for h in [0.5f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let fmin = 32.7 * h / 60.0;
            let fmax = srq / 2.0;
            let bpo = NB as f64 / (fmax / fmin).log2();
            let q = 1.0 / (2f64.powf(1.0 / bpo) - 1.0);
            let fr: Vec<f64> = (0..NB).map(|k| fmin * 2f64.powf(k as f64 / bpo)).collect();
            let ln: Vec<f64> = fr.iter().map(|f| (q * srq / f).ceil()).collect();
            let maxl = ln.iter().cloned().fold(0.0, f64::max);
            let width = 1usize << (maxl.floor().log2().ceil() as usize);
            let pad = width / 2;
            let n_f = (t + 2 * pad - width) / hop + 1;
            let mut frames = Vec::new();
            for f in 0..n_f {
                let start = (f * hop) as i64 - pad as i64;
                let i0 = start.max(0) as usize;
                let i1 = (start + width as i64).min(t as i64) as usize;
                let (mut re, mut im) = (Vec::new(), Vec::new());
                for k in 0..NB {
                    let l = ln[k];
                    let ll = l as i64;
                    let ks = (width as f64 / 2.0 - l / 2.0).ceil() as i64 - if l % 2.0 == 1.0 { 1 } else { 0 };
                    let lo = (-l / 2.0).floor() as i64;
                    let mut r = vec![0f32; i1 - i0];
                    let mut m = vec![0f32; i1 - i0];
                    for tt in i0..i1 {
                        let j = tt as i64 - start - ks;
                        if j < 0 || j >= ll {
                            continue;
                        }
                        let th = (j + lo) as f64 * 2.0 * PI * fr[k] / srq;
                        let a = 2.0 * (0.5 - 0.5 * (2.0 * PI * j as f64 / ll as f64).cos()) / l;
                        r[tt - i0] = (a * th.cos()) as f32;
                        m[tt - i0] = (a * th.sin()) as f32;
                    }
                    re.push(r);
                    im.push(m);
                }
                frames.push(Frame { i0, i1, re, im });
            }
            harm.push(Harm { frames, n_f, scale: ln.iter().map(|l| l.sqrt()).collect() });
        }
        Plan { bands, t, harm }
    })
}

fn band_env(x: &[f32], p: &Plan, fft: &Arc<dyn Fft<f64>>, win: &[f64]) -> Vec<f32> {
    let l = x.len() as i64;
    let mut out = vec![0f32; BANDS * p.t];
    let mut buf = vec![Complex::new(0f64, 0f64); N];
    for t in 0..p.t {
        for i in 0..N {
            let mut q = (t * HOP + i) as i64 - (N / 2) as i64;
            if q < 0 {
                q = -q;
            } else if q >= l {
                q = 2 * (l - 1) - q;
            }
            buf[i] = Complex::new(x[q as usize] as f64 * win[i], 0.0);
        }
        fft.process(&mut buf);
        for (b, &(s, e, w)) in p.bands.iter().enumerate() {
            let acc: f64 = (s..e).map(|k| buf[k].norm()).sum();
            out[b * p.t + t] = (acc * w) as f32;
        }
    }
    out
}

fn hcqm(clips: &[&[f32]]) -> Vec<f32> {
    let p = plan();
    let t = p.t;
    let b = clips.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let win = hann(N);
    let mut bands = Vec::with_capacity(b * BANDS * t);
    for c in clips {
        bands.extend(band_env(c, p, &fft, &win));
    }
    let mut db: Vec<f32> = bands.iter().map(|v| 10.0 * v.max(1e-10).log10()).collect();
    let mx = db.iter().cloned().fold(f32::MIN, f32::max);
    db.iter_mut().for_each(|v| *v = v.max(mx - 80.0));
    let rows = b * BANDS;
    let mut osf = vec![0f32; rows * t];
    for r in 0..rows {
        for tt in 3..t {
            osf[r * t + tt] = (db[r * t + tt - 2] - db[r * t + tt - 3]).max(0.0);
        }
    }
    let mut out = vec![0f32; b * 6 * NB * BANDS];
    for (hi, h) in p.harm.iter().enumerate() {
        for r in 0..rows {
            let (c, band) = (r / BANDS, r % BANDS);
            for k in 0..NB {
                let mut tot = 0f64;
                for f in &h.frames {
                    let (mut sr, mut si) = (0f64, 0f64);
                    let (rr, ii) = (&f.re[k], &f.im[k]);
                    for tt in f.i0..f.i1 {
                        let v = osf[r * t + tt] as f64;
                        sr += v * rr[tt - f.i0] as f64;
                        si += v * ii[tt - f.i0] as f64;
                    }
                    tot += (sr * h.scale[k]).hypot(si * h.scale[k]);
                }
                let fl = band * NB + k;
                out[((c * 6 + hi) * NB + (fl >> 3)) * BANDS + (fl & 7)] = (tot / h.n_f as f64) as f32;
            }
        }
    }
    out
}

fn net(model: &Path, y22: &[f32]) -> Result<(f64, f32)> {
    let n = y22.len() / CLIP;
    let idx: Vec<usize> = if n <= MAXCLIPS { (0..n).collect() } else { (0..MAXCLIPS).map(|i| ((i * (n - 1)) as f64 / (MAXCLIPS - 1) as f64).round() as usize).collect() };
    let clips: Vec<&[f32]> = idx.iter().map(|&i| &y22[i * CLIP..(i + 1) * CLIP]).collect();
    let data = hcqm(&clips);
    let runnable = tract_onnx::onnx()
        .model_for_path(model)?
        .with_input_fact(0, f32::fact([1usize, 6, NB, BANDS]).into())?
        .into_optimized()?
        .into_runnable()?;
    let per = 6 * NB * BANDS;
    let mut probs = vec![0f64; 256];
    for c in 0..clips.len() {
        let input = Tensor::from_shape(&[1, 6, NB, BANDS], &data[c * per..(c + 1) * per])?;
        let out = runnable.run(tvec!(input.into()))?;
        let lg: Vec<f32> = out[0].to_array_view::<f32>()?.iter().cloned().collect();
        let m = lg.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f64> = lg.iter().map(|v| ((v - m) as f64).exp()).collect();
        let z: f64 = e.iter().sum();
        for k in 0..256.min(e.len()) {
            probs[k] += e[k] / z / clips.len() as f64;
        }
    }
    let (mut cls, mut best) = (0usize, 0f64);
    for (k, v) in probs.iter().enumerate() {
        if *v > best {
            best = *v;
            cls = k;
        }
    }
    Ok((30.0 + cls as f64, best as f32))
}

fn mel_filters() -> Vec<(Vec<usize>, Vec<f64>)> {
    let (fsp, mlh) = (200.0 / 3.0, 1000.0);
    let mlm = mlh / fsp;
    let ls = 6.4f64.ln() / 27.0;
    let h2m = |f: f64| if f >= mlh { mlm + (f / mlh).ln() / ls } else { f / fsp };
    let m2h = |m: f64| if m >= mlm { mlh * (ls * (m - mlm)).exp() } else { fsp * m };
    let m1 = h2m(SR as f64 / 2.0);
    let mf: Vec<f64> = (0..130).map(|i| m2h(m1 * i as f64 / 129.0)).collect();
    (0..128)
        .map(|i| {
            let (mut ks, mut ws) = (vec![], vec![]);
            for k in 0..=1024usize {
                let fk = k as f64 * SR as f64 / 2048.0;
                let w = ((fk - mf[i]) / (mf[i + 1] - mf[i])).min((mf[i + 2] - fk) / (mf[i + 2] - mf[i + 1])).max(0.0) * 2.0 / (mf[i + 2] - mf[i]);
                if w > 0.0 {
                    ks.push(k);
                    ws.push(w);
                }
            }
            (ks, ws)
        })
        .collect()
}

fn onset(y: &[f32]) -> Vec<f32> {
    let h = 128usize;
    let t = 1 + y.len() / h;
    let filt = mel_filters();
    let win = hann(N);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let mut buf = vec![Complex::new(0f64, 0f64); N];
    let mut s = vec![0f32; 128 * t];
    let mut mx = f32::MIN;
    for tt in 0..t {
        for i in 0..N {
            let p = (tt * h + i) as i64 - (N / 2) as i64;
            buf[i] = Complex::new(if p >= 0 && (p as usize) < y.len() { y[p as usize] as f64 * win[i] } else { 0.0 }, 0.0);
        }
        fft.process(&mut buf);
        for (m, (ks, ws)) in filt.iter().enumerate() {
            let a: f64 = ks.iter().zip(ws).map(|(&k, w)| w * buf[k].norm_sqr()).sum();
            let v = 10.0 * (a as f32).max(1e-10).log10();
            s[m * t + tt] = v;
            mx = mx.max(v);
        }
    }
    let fl = mx - 80.0;
    let mut env = vec![0f32; t];
    for tt in 9..t {
        let mut a = 0f32;
        for m in 0..128 {
            let (p, q) = (s[m * t + tt - 8].max(fl), s[m * t + tt - 9].max(fl));
            if p > q {
                a += p - q;
            }
        }
        env[tt] = a / 128.0;
    }
    env
}

fn refine(env: &[f32], coarse: f64) -> f64 {
    let fps = SR as f64 / 128.0;
    let n = env.len();
    let mean = env.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let a0 = (coarse - 1.0).max(20.0);
    let nc = ((coarse + 2.0 - a0) / 0.01).ceil() as usize;
    let max_lag = (60.0 * fps / a0 * 16.0).ceil() as usize + 2;
    if max_lag >= n {
        return coarse;
    }
    let m = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let mut buf = vec![Complex::new(0f64, 0f64); m];
    for i in 0..n {
        buf[i] = Complex::new(env[i] as f64 - mean, 0.0);
    }
    fft.process(&mut buf);
    buf.iter_mut().for_each(|c| *c = Complex::new(c.norm_sqr(), 0.0));
    fft.process(&mut buf);
    let (mut best, mut bc) = (f64::MIN, coarse);
    for c in 0..nc {
        let bpm = a0 + c as f64 * 0.01;
        let mut sc = 0.0;
        for b in 1..=16 {
            let x = 60.0 * fps / bpm * b as f64;
            let i = x.floor() as usize;
            let f = x - i as f64;
            sc += (buf[i].re * (1.0 - f) + buf[i + 1].re * f) / m as f64;
        }
        if sc > best {
            best = sc;
            bc = bpm;
        }
    }
    bc
}

/// `y44`: mono, 44.1 kHz, peak-normalised (see `audio::load_44k`).
pub fn detect(model: &Path, y44: &[f32]) -> Result<TempoResult> {
    let y22 = dsp::half(y44, 1.0);
    if y22.len() < CLIP {
        return Err(anyhow!("track too short for tempo detection (< 8 s)"));
    }
    let (coarse, conf) = net(model, &y22)?;
    let st = y22.len().saturating_sub(REFINE_S * SR) / 2;
    let part = &y22[st..(st + REFINE_S * SR).min(y22.len())];
    let bpm = refine(&onset(part), coarse);
    Ok(TempoResult { bpm: (bpm * 100.0).round() / 100.0, conf, coarse })
}
