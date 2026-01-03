use capture_utils::generate_overlays;
use image::{Rgba, RgbaImage};
use serde_json::json;
use std::fs;

#[test]
fn overlay_generation_respects_schema_and_writes_files() {
    let tmp = tempfile::tempdir().unwrap();
    let run_dir = tmp.path();

    // Write a tiny test image.
    let img_path = run_dir.join("images");
    fs::create_dir_all(&img_path).unwrap();
    let img_file = img_path.join("frame_00000.png");
    let mut img = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 255]));
    img.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
    img.save(&img_file).unwrap();

    // Write a label JSON following the documented schema subset.
    let labels_dir = run_dir.join("labels");
    fs::create_dir_all(&labels_dir).unwrap();
    let meta = json!({
        "frame_id": 0,
        "sim_time": 0.0,
        "unix_time": 0.0,
        "image": "images/frame_00000.png",
        "image_present": true,
        "camera_active": true,
        "polyp_seed": 1,
        "polyp_labels": [
            { "center_world": [0.0, 0.0, 0.0], "bbox_px": [0.0, 0.0, 2.0, 2.0], "bbox_norm": [0.0, 0.0, 0.0, 0.0] }
        ]
    });
    fs::write(
        labels_dir.join("frame_00000.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    // Generate overlays and confirm output exists.
    generate_overlays(run_dir).unwrap();
    let overlays_dir = run_dir.join("overlays");
    let overlay_file = overlays_dir.join("frame_00000.png");
    assert!(overlay_file.exists(), "overlay file should be created");
}
