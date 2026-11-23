//! Sanity test: a synthetic G-major progression must be detected as G major or a close relative.
//! Skipped when the model file is not present (run scripts/fetch-models first).
use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    ["../src-tauri/models/keynet.onnx", "../models/keynet.onnx"].iter().map(PathBuf::from).find(|p| p.exists())
}

fn synth() -> Vec<f32> {
    let sr = 44100usize;
    let n = sr * 40;
    let mut y = vec![0f32; n];
    let chords: [[f64; 3]; 4] = [[196.0, 246.9, 293.7], [130.8, 164.8, 196.0], [146.8, 185.0, 220.0], [196.0, 246.9, 293.7]];
    for bar in 0..80 {
        let ch = chords[bar % 4];
        let t0 = (bar as f64 * 0.5 * sr as f64) as usize;
        for f in ch {
            for i in 0..(sr / 2) {
                let t = i as f64 / sr as f64;
                let env = (-3.0 * t).exp() * (1.0 - (-60.0 * t).exp());
                let mut v = 0.0;
                for h in 1..7 {
                    if f * h as f64 > 8000.0 { break; }
                    v += 0.3 / (h as f64).powf(1.2) * (2.0 * std::f64::consts::PI * f * h as f64 * t).sin();
                }
                if t0 + i < n { y[t0 + i] += (v * env) as f32; }
            }
        }
    }
    let pk = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    y.iter_mut().for_each(|v| *v *= 0.95 / pk);
    y
}

#[test]
fn detects_g_major_family() {
    let Some(m) = model() else { eprintln!("model not found, skipping"); return };
    let r = satcore::key::detect(&m, &synth()).unwrap();
    println!("{r:?}");
    assert!(["9B", "9A", "10B", "8B"].contains(&r.camelot.as_str()), "unexpected key {:?}", r);
}

#[test]
fn resampler_keeps_length_and_tone() {
    let y: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
    let z = satcore::audio::resample(&y, 48000, 44100);
    assert_eq!(z.len(), 44100);
    let rms = (z[4000..40000].iter().map(|v| v * v).sum::<f32>() / 36000.0).sqrt();
    assert!((rms - 0.7071).abs() < 0.03, "rms {rms}");
}

fn model_named(n: &str) -> Option<PathBuf> {
    [format!("../src-tauri/models/{n}"), format!("../models/{n}")].iter().map(PathBuf::from).find(|p| p.exists())
}

#[test]
fn tempo_of_120_bpm_chord_pulse() {
    let Some(m) = model_named("deeprhythm.onnx") else { eprintln!("deeprhythm.onnx not found, skipping"); return };
    let y = satcore::dsp::half(&[], 1.0); // exercise the empty-input path without panicking
    assert!(y.is_empty());
    let r = satcore::tempo::detect(&m, &synth()).unwrap();
    println!("{r:?}");
    assert!((r.bpm - 120.0).abs() < 1.0 || (r.bpm - 60.0).abs() < 0.6 || (r.bpm - 240.0).abs() < 1.5, "unexpected tempo {:?}", r);
}

fn wav_bytes() -> Vec<u8> {
    let sr = 22050u32;
    let n = sr as usize * 3;
    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + 2 * n as u32).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&sr.to_le_bytes());
    v.extend_from_slice(&(sr * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(2 * n as u32).to_le_bytes());
    for i in 0..n {
        v.extend_from_slice(&(((i as f32 * 0.05).sin() * 8000.0) as i16).to_le_bytes());
    }
    v
}

#[test]
fn tags_roundtrip_backup_restore_and_scan() {
    let dir = std::env::temp_dir().join(format!("sat_tags_{}", std::process::id()));
    let sub = dir.join("album");
    std::fs::create_dir_all(&sub).unwrap();
    let f = sub.join("t.wav");
    std::fs::write(&f, wav_bytes()).unwrap();
    assert_eq!(satcore::tags::read_genre(&f).unwrap(), "");
    let g = satcore::tags::write_tags(&f, Some("Techno"), Some(128.4), Some("Am"), true).unwrap();
    assert_eq!(g, "Techno");
    assert!(satcore::tags::has_backup(&f));
    let found = satcore::scan::scan_dir(&dir).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].genre, "Techno");
    assert!(found[0].rel.ends_with("album/t.wav"));
    assert!(satcore::tags::restore_backup(&f).unwrap());
    assert_eq!(satcore::tags::read_genre(&f).unwrap(), "");
    // decode path of the analysis entry point (3 s file is too short for key/tempo but must decode)
    let a = satcore::analyze_bytes(&satcore::Models::default(), &satcore::Req::default(), std::fs::read(&f).unwrap(), Some("wav"));
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn maest_plumbing_with_tiny_stand_in_model() {
    // tests/data/tiny_maest.onnx is a 96->400 linear stand-in with MAEST's input/output shapes
    let m = PathBuf::from("tests/data/tiny_maest.onnx");
    let y16 = satcore::audio::resample(&synth(), 44100, 16000);
    let r = satcore::maest::classify(&m, &y16, 1).unwrap();
    assert_eq!(r.len(), 10);
    assert!(r.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(r.iter().all(|s| s.label.contains("---")));
    let models = satcore::Models { maest: Some(m), ..Default::default() };
    let req = satcore::Req { genre: true, feats: true, nwin: 3, ..Default::default() };
    let a = satcore::analyze_bytes(&models, &req, wav_bytes(), Some("wav"));
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    assert!(a.genre.is_some() && a.feats.is_some());
}
