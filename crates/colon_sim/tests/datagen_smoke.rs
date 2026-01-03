use std::process::Command;

#[test]
fn datagen_headless_smoke() {
    if std::env::var("RUN_DATAGEN_SMOKE").is_err() {
        eprintln!("skipping datagen smoke (set RUN_DATAGEN_SMOKE=1 to enable)");
        return;
    }

    // Run the datagen binary with a tiny frame cap and a temp output root.
    let tmp = tempfile::tempdir().unwrap();
    let status = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "datagen_headless",
            "--",
            "--max-frames",
            "20",
            "--output-root",
            tmp.path().to_str().unwrap(),
        ])
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("failed to spawn datagen")
        .wait()
        .expect("failed to wait on datagen");
    assert!(status.success(), "datagen should exit cleanly");

    // Basic assertions: run dir exists and has expected subfolders.
    let entries: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        !entries.is_empty(),
        "expected at least one run directory in {:?}",
        tmp.path()
    );
    let run_dir = entries[0].path();
    assert!(run_dir.join("images").is_dir());
    assert!(run_dir.join("labels").is_dir());
    assert!(run_dir.join("overlays").is_dir());
    assert!(run_dir.join("run_manifest.json").is_file());
}
