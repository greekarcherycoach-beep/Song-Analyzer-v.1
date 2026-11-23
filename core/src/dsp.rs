//! log-CQT front-end of OpenKeyScan (port of okscan/dsp.py; librosa-compatible plan).
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{collections::HashMap, f64::consts::PI, sync::Arc};

pub const SR: f64 = 44100.0;
pub const HOP: usize = 8820;
pub const FMIN: f64 = 65.0;
pub const NBINS: usize = 105;
pub const BPO: usize = 24;

struct Octave {
    nfft: usize,
    hop: usize,
    idx: Vec<Vec<u32>>,
    re: Vec<Vec<f32>>,
    im: Vec<Vec<f32>>,
    down: bool,
}
pub struct Plan {
    oct: Vec<Octave>,
    norm: Vec<f64>,
    ffts: HashMap<usize, Arc<dyn Fft<f64>>>,
}

fn i0(x: f64) -> f64 {
    let (mut s, mut t) = (1.0, 1.0);
    for k in 1..40 {
        t *= (x / 2.0 / k as f64).powi(2);
        s += t;
    }
    s
}

/// Kaiser low-pass + decimate by 2. `scale` = sqrt(2) for librosa(scale=True).
pub fn half(y: &[f32], scale: f32) -> Vec<f32> {
    const L: usize = 48;
    let fc = 0.25;
    let mut h = vec![0f64; 2 * L + 1];
    let mut sum = 0.0;
    for n in -(L as i64)..=(L as i64) {
        let x = if n == 0 { 2.0 * fc } else { (2.0 * PI * fc * n as f64).sin() / (PI * n as f64) };
        let w = i0(8.0 * (1.0 - (n as f64 / L as f64).powi(2)).sqrt()) / i0(8.0);
        h[(n + L as i64) as usize] = x * w;
        sum += x * w;
    }
    h.iter_mut().for_each(|v| *v /= sum);
    let n = y.len();
    let m = (n + 1) / 2;
    (0..m)
        .map(|k| {
            let c = 2 * k;
            let a = c.saturating_sub(L);
            let b = (c + L).min(n - 1);
            let mut acc = 0f64;
            for j in a..=b {
                acc += y[j] as f64 * h[c + L - j];
            }
            (acc * scale as f64) as f32
        })
        .collect()
}

impl Plan {
    pub fn new() -> Plan {
        let nb = NBINS;
        let n_oct = (nb + BPO - 1) / BPO;
        let n_filt = BPO.min(nb);
        let mut freqs: Vec<f64> = Vec::new();
        for o in 0..n_oct {
            for k in 0..BPO {
                freqs.push(2f64.powi(o as i32) * 2f64.powf(k as f64 / BPO as f64));
            }
        }
        freqs.truncate(nb);
        freqs.iter_mut().for_each(|f| *f *= FMIN);
        let lf: Vec<f64> = freqs.iter().map(|f| f.log2()).collect();
        let mut bp = vec![0f64; nb];
        bp[0] = 1.0 / (lf[1] - lf[0]);
        bp[nb - 1] = 1.0 / (lf[nb - 1] - lf[nb - 2]);
        for i in 1..nb - 1 {
            bp[i] = 2.0 / (lf[i + 1] - lf[i - 1]);
        }
        let alpha: Vec<f64> = bp.iter().map(|b| (2f64.powf(2.0 / b) - 1.0) / (2f64.powf(2.0 / b) + 1.0)).collect();
        let mut planner = FftPlanner::<f64>::new();
        let mut ffts = HashMap::new();
        let mut oct = Vec::new();
        let (mut my_sr, mut my_hop) = (SR, HOP);
        for i in 0..n_oct {
            let hi = if i == 0 { nb } else { nb - n_filt * i };
            let lo = nb.saturating_sub(n_filt * (i + 1));
            let lens: Vec<f64> = (lo..hi).map(|j| my_sr / (alpha[j] * freqs[j])).collect();
            let maxl = lens.iter().cloned().fold(0.0, f64::max);
            let nfft = 1usize << (maxl.log2().ceil() as usize);
            let half_n = nfft / 2;
            let fft = ffts.entry(nfft).or_insert_with(|| planner.plan_fft_forward(nfft)).clone();
            let (mut bi, mut br, mut bm) = (vec![], vec![], vec![]);
            for (q, &ilen) in lens.iter().enumerate() {
                let f = freqs[lo + q];
                let st = (-ilen / 2.0).floor() as i64;
                let lt = ((ilen / 2.0).floor() as i64 - st) as usize;
                let lp = (nfft - lt) / 2;
                let mut buf = vec![Complex::new(0f64, 0f64); nfft];
                let mut l1 = 0.0;
                for j in 0..lt {
                    let w = 0.5 - 0.5 * (2.0 * PI * j as f64 / lt as f64).cos();
                    let t = (st + j as i64) as f64 * 2.0 * PI * f / my_sr;
                    buf[lp + j] = Complex::new(t.cos() * w, t.sin() * w);
                    l1 += w;
                }
                let sc = ilen / nfft as f64 / l1;
                buf.iter_mut().for_each(|c| *c *= sc);
                fft.process(&mut buf);
                let mag: Vec<f64> = buf[..=half_n].iter().map(|c| c.norm()).collect();
                let tot: f64 = mag.iter().sum();
                let mut srt = mag.clone();
                srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let (mut cum, mut th) = (0.0, srt[0]);
                for v in &srt {
                    cum += v / tot;
                    if !(cum < 0.01) {
                        th = *v;
                        break;
                    }
                }
                let g = (SR / my_sr).sqrt();
                let (mut ix, mut rr, mut ii) = (vec![], vec![], vec![]);
                for k in 0..=half_n {
                    if mag[k] >= th {
                        ix.push(k as u32);
                        rr.push((buf[k].re * g) as f32);
                        ii.push((buf[k].im * g) as f32);
                    }
                }
                bi.push(ix);
                br.push(rr);
                bm.push(ii);
            }
            let down = my_hop % 2 == 0 && i + 1 < n_oct;
            oct.push(Octave { nfft, hop: my_hop, idx: bi, re: br, im: bm, down });
            if down {
                my_hop /= 2;
                my_sr /= 2.0;
            }
        }
        let norm = (0..nb).map(|i| (SR / (alpha[i] * freqs[i])).sqrt()).collect();
        Plan { oct, norm, ffts }
    }

    /// log1p(|CQT|) with shape [NBINS][T] (row-major), last two frames dropped.
    pub fn spec(&self, y: &[f32]) -> (Vec<f32>, usize) {
        let mut resp: Vec<(Vec<f32>, Vec<f32>, usize, usize)> = Vec::new();
        let mut yy = y.to_vec();
        for o in &self.oct {
            resp.push(self.octave(&yy, o));
            if o.down {
                yy = half(&yy, std::f32::consts::SQRT_2);
            }
        }
        let t_len = resp.iter().map(|r| r.2).min().unwrap() - 2;
        let mut out = vec![0f32; NBINS * t_len];
        let mut end = NBINS;
        for (r_, i_, nf, n) in &resp {
            let take = (*n).min(end);
            let r0 = n - take;
            let o0 = end - take;
            for f in 0..take {
                for t in 0..t_len {
                    let k = o0 + f;
                    let a = r_[(r0 + f) * nf + t] as f64;
                    let b = i_[(r0 + f) * nf + t] as f64;
                    out[k * t_len + t] = ((a.hypot(b) / self.norm[k]).ln_1p()) as f32;
                }
            }
            end -= n;
        }
        (out, t_len)
    }

    fn octave(&self, y: &[f32], o: &Octave) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let nb = o.idx.len();
        let nf = 1 + y.len() / o.hop;
        let half = (o.nfft / 2) as i64;
        let fft = self.ffts[&o.nfft].clone();
        let mut re = vec![0f32; nb * nf];
        let mut im = vec![0f32; nb * nf];
        let mut buf = vec![Complex::new(0f64, 0f64); o.nfft];
        for t in 0..nf {
            let s0 = (t * o.hop) as i64 - half;
            for (i, b) in buf.iter_mut().enumerate() {
                let p = s0 + i as i64;
                *b = Complex::new(if p >= 0 && (p as usize) < y.len() { y[p as usize] as f64 } else { 0.0 }, 0.0);
            }
            fft.process(&mut buf);
            for f in 0..nb {
                let (mut sr, mut si) = (0f64, 0f64);
                for q in 0..o.idx[f].len() {
                    let c = buf[o.idx[f][q] as usize];
                    let (br, bi) = (o.re[f][q] as f64, o.im[f][q] as f64);
                    sr += br * c.re - bi * c.im;
                    si += br * c.im + bi * c.re;
                }
                re[f * nf + t] = sr as f32;
                im[f * nf + t] = si as f32;
            }
        }
        (re, im, nf, nb)
    }
}
