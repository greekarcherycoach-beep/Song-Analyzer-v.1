// Dumps 16 kHz audio + feature vectors for parity checks against the JavaScript implementations.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1])?;
    let d = satcore::audio::decode_full(bytes, Some("wav"))?;
    let y44 = satcore::audio::resample(&d.y, d.sr, 44100);
    let y16 = satcore::audio::resample(&y44, 44100, 16000);
    let w = |p: &str, v: &[f32]| std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    w("/tmp/y16.f32", &y16)?;
    let f = satcore::features::analyze(&y16, 16000.0, d.sr, d.ch, y16.len() as f64 / 16000.0);
    println!("RUST_FEATS bpm={} key={} cam={} energy={} dance={} valence={} lufs={} peak={} dr={} cent={} bright={} conf={}", f.bpm, f.key, f.cam, f.energy, f.dance, f.valence, f.lufs, f.peak, f.dr, f.cent, f.bright, f.conf);
    let seg = satcore::maest::segments(&y16, 1);
    w("/tmp/maest_rust.f32", &satcore::maest::features(seg[0]))?;
    Ok(())
}
