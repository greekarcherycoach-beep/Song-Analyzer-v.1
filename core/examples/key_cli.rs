fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[2])?;
    let ext = std::path::Path::new(&a[2]).extension().and_then(|e| e.to_str());
    let t = std::time::Instant::now();
    let r = satcore::key::detect_bytes(std::path::Path::new(&a[1]), bytes, ext)?;
    println!("{:?}  ({} ms)", r, t.elapsed().as_millis());
    Ok(())
}
