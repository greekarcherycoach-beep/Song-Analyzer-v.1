//! Heuristic audio features (energy, danceability, mood, loudness, dynamic range, brightness, waveform, chroma).
//! Port of the `SA.analyze` JavaScript routine of the UI; works on 16 kHz mono audio.
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;
use std::f64::consts::PI;

const NOTE: [&str; 12] = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
const CAM: [u32; 12] = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const MAJ: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MIN: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
const N: usize = 1024;
const H: usize = 256;

#[derive(Serialize, Debug, Clone)]
pub struct Features {
    pub bpm: f64,
    pub key: String,
    pub cam: String,
    pub energy: i32,
    pub dance: i32,
    pub valence: i32,
    pub lufs: f64,
    pub peak: f64,
    pub dr: f64,
    pub cent: i32,
    pub bright: i32,
    pub conf: i32,
    pub dur: f64,
    pub sr: u32,
    pub ch: u32,
    pub wave: Vec<f32>,
    pub chroma: Vec<f32>,
    pub root: usize,
}

fn clamp(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}
/// JavaScript's Math.round (halves go up).
fn jr(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// `x`: mono audio at `sr` (16 kHz), `orig_sr`/`ch`/`dur` only describe the source file.
pub fn analyze(x: &[f32], sr: f64, orig_sr: u32, ch: u32, dur: f64) -> Features {
    let n = x.len();
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let win: Vec<f64> = (0..N).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / N as f64).cos()).collect();
    let bin = sr / N as f64;
    let mut pc = vec![-1i32; N / 2];
    let mut fr = vec![0f64; N / 2];
    for k in 1..N / 2 {
        let hz = k as f64 * bin;
        fr[k] = hz;
        if hz > 60.0 && hz < 2000.0 {
            pc[k] = (((jr(12.0 * (hz / 440.0).log2() + 69.0) as i64 % 12) + 12) % 12) as i32;
        }
    }
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N);
    let mut buf = vec![Complex::new(0f64, 0f64); N];
    let mut prev = vec![0f64; N / 2];
    let t_n = if n >= N { (n - N) / H } else { 0 };
    let (mut flux, mut rms) = (vec![0f64; t_n], vec![0f64; t_n]);
    let mut chroma = [0f64; 12];
    let (mut cent, mut flux_sum, mut mag_sum) = (0f64, 0f64, 0f64);
    for t in 0..t_n {
        let o = t * H;
        let mut e = 0f64;
        for i in 0..N {
            let v = x[o + i] as f64;
            buf[i] = Complex::new(v * win[i], 0.0);
            e += v * v;
        }
        rms[t] = (e / N as f64).sqrt();
        fft.process(&mut buf);
        let (mut fx, mut ms, mut cn) = (0f64, 0f64, 0f64);
        for k in 1..N / 2 {
            let m = buf[k].norm().sqrt();
            let d = m - prev[k];
            if d > 0.0 {
                fx += d;
            }
            prev[k] = m;
            ms += m;
            cn += m * fr[k];
            if pc[k] >= 0 {
                chroma[pc[k] as usize] += m * m;
            }
        }
        flux[t] = fx;
        flux_sum += fx;
        mag_sum += ms;
        cent += if ms > 0.0 { cn / ms } else { 0.0 };
    }
    let fps = sr / H as f64;
    // BPM (autocorrelation of the onset envelope, tempo prior around 115 BPM)
    let w = fps.round() as usize;
    let mut env = vec![0f64; t_n];
    for i in 0..t_n {
        let (j0, j1) = (i.saturating_sub(w), (i + w).min(t_n));
        let a: f64 = flux[j0..j1].iter().sum();
        env[i] = (flux[i] - a / (j1 - j0) as f64).max(0.0);
    }
    let r0: f64 = env.iter().map(|v| v * v).sum();
    let lo = (fps * 60.0 / 200.0).floor() as usize;
    let hi = (fps * 60.0 / 60.0).ceil() as usize;
    let mut r = vec![0f64; hi + 2];
    for l in lo.saturating_sub(1)..=hi + 1 {
        let mut s = 0.0;
        for i in 0..t_n.saturating_sub(l) {
            s += env[i] * env[i + l];
        }
        r[l] = s;
    }
    let (mut best, mut bl, mut bs) = (0f64, lo, 0f64);
    for l in lo..=hi {
        let bp = 60.0 * fps / l as f64;
        let wgt = (-0.5 * ((bp / 115.0).log2() / 1.0).powi(2)).exp();
        let v = r[l] * wgt;
        if v > best && r[l] >= r[l - 1] && r[l] >= r[l + 1] {
            best = v;
            bl = l;
            bs = r[l];
        }
    }
    let (a, b, c) = (r[bl - 1], r[bl], r[bl + 1]);
    let den = a - 2.0 * b + c;
    let off = if den != 0.0 { 0.5 * (a - c) / den } else { 0.0 };
    let bpm = 60.0 * fps / (bl as f64 + off.max(-1.0).min(1.0));
    let beat_str = if r0 > 0.0 { bs / r0 } else { 0.0 };
    // key (Krumhansl-Schmuckler correlation)
    let (mut bk, mut bm, mut bc) = (0usize, 0usize, -9f64);
    let mc = chroma.iter().sum::<f64>() / 12.0;
    for m in 0..2 {
        let pr = if m == 1 { &MIN } else { &MAJ };
        let pm = pr.iter().sum::<f64>() / 12.0;
        for k in 0..12 {
            let (mut sxy, mut sx, mut sy) = (0f64, 0f64, 0f64);
            for i in 0..12 {
                let u = chroma[(k + i) % 12] - mc;
                let v = pr[i] - pm;
                sxy += u * v;
                sx += u * u;
                sy += v * v;
            }
            let d = (sx * sy).sqrt();
            let cc = sxy / if d == 0.0 { 1.0 } else { d };
            if cc > bc {
                bc = cc;
                bk = k;
                bm = m;
            }
        }
    }
    let cam = if bm == 1 { format!("{}A", CAM[(bk + 3) % 12]) } else { format!("{}B", CAM[bk]) };
    // loudness / dynamics / mood
    let ms: f64 = rms.iter().map(|v| v * v).sum();
    let db = 20.0 * ((ms / (t_n.max(1)) as f64).sqrt() + 1e-9).log10();
    let mut rdb: Vec<f64> = rms.iter().map(|v| 20.0 * (v + 1e-9).log10()).collect();
    rdb.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let dr = if rdb.is_empty() { 0.0 } else { rdb[(rdb.len() as f64 * 0.95).floor() as usize] - rdb[(rdb.len() as f64 * 0.1).floor() as usize] };
    let cavg = cent / t_n.max(1) as f64;
    let bright = clamp((cavg - 500.0) / 3000.0);
    let flux_rel = if mag_sum != 0.0 { flux_sum / mag_sum } else { 0.0 };
    let loud_n = clamp((db + 32.0) / 22.0);
    let energy = jr(100.0 * clamp(0.5 * loud_n + 0.25 * bright + 0.25 * clamp(flux_rel / 0.35)));
    let dance = jr(100.0 * clamp(beat_str / 0.45) * if (80.0..=170.0).contains(&bpm) { 1.0 } else { 0.8 });
    let valence = jr(100.0 * clamp(0.3 + if bm == 1 { -0.05 } else { 0.2 } + clamp((bpm - 70.0) / 90.0) * 0.25 + bright * 0.2 + loud_n * 0.1));
    // waveform + chroma for the detail view
    let seg = (n / 300).max(1);
    let mut wave: Vec<f64> = (0..300).map(|i| x[(i * seg).min(n)..((i + 1) * seg).min(n)].iter().fold(0f32, |m, v| m.max(v.abs())) as f64).collect();
    let wm = wave.iter().cloned().fold(0.0, f64::max);
    let wm = if wm == 0.0 { 1.0 } else { wm };
    wave.iter_mut().for_each(|v| *v /= wm);
    let cm = chroma.iter().cloned().fold(0.0, f64::max);
    let cm = if cm == 0.0 { 1.0 } else { cm };
    Features {
        bpm: jr(bpm * 10.0) / 10.0,
        key: format!("{}{}", NOTE[bk], if bm == 1 { " minor" } else { " major" }),
        cam,
        energy: energy as i32,
        dance: dance as i32,
        valence: valence as i32,
        lufs: jr(db * 10.0) / 10.0,
        peak: jr(20.0 * (peak + 1e-9).log10() * 10.0) / 10.0,
        dr: jr(dr * 10.0) / 10.0,
        cent: jr(cavg) as i32,
        bright: jr(bright * 100.0) as i32,
        conf: jr(clamp((bc - 0.4) / 0.45) * 100.0) as i32,
        dur,
        sr: orig_sr,
        ch,
        wave: wave.iter().map(|v| *v as f32).collect(),
        chroma: chroma.iter().map(|v| (*v / cm) as f32).collect(),
        root: bk,
    }
}
