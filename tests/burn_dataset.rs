use colon_sim::tools::burn_dataset::load_run_dataset;
use image::{Rgba, RgbaImage};
use serde_json::json;
use std::fs;

#[test]
fn load_run_dataset_normalizes_images_and_boxes() {
    let tmp = tempfile::tempdir().unwrap();
    let run_dir = tmp.path();

    // Write a small test image.
    let img_dir = run_dir.join("images");
    fs::create_dir_all(&img_dir).unwrap();
    let img_path = img_dir.join("frame_00000.png");
    let mut img = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255])); // top-left red
    img.put_pixel(1, 0, Rgba([0, 255, 0, 255])); // top-right green
    img.put_pixel(0, 1, Rgba([0, 0, 255, 255])); // bottom-left blue
    img.put_pixel(1, 1, Rgba([255, 255, 255, 255])); // bottom-right white
    img.save(&img_path).unwrap();

    // Write a label file with a pixel bbox.
    let labels_dir = run_dir.join("labels");
    fs::create_dir_all(&labels_dir).unwrap();
    let label = json!({
        "frame_id": 0,
        "image": "images/frame_00000.png",
        "image_present": true,
        "polyp_labels": [
            { "bbox_px": [0.0, 0.0, 2.0, 2.0] }
        ]
    });
    fs::write(
        labels_dir.join("frame_00000.json"),
        serde_json::to_string_pretty(&label).unwrap(),
    )
    .unwrap();

    let samples = load_run_dataset(run_dir).expect("dataset should load");
    assert_eq!(samples.len(), 1);
    let sample = &samples[0];
    assert_eq!(sample.frame_id, 0);
    assert_eq!(sample.width, 2);
    assert_eq!(sample.height, 2);
    assert_eq!(sample.image_chw.len(), 12); // 3 * 2 * 2

    // First channel (R) first pixel should be 1.0, others normalized.
    assert!((sample.image_chw[0] - 1.0).abs() < 1e-6);
    assert!((sample.image_chw[1] - 0.0).abs() < 1e-6);

    // Bounding box normalized to full image extent.
    assert_eq!(sample.boxes.len(), 1);
    let bbox = sample.boxes[0];
    assert!((bbox[0] - 0.0).abs() < 1e-6);
    assert!((bbox[1] - 0.0).abs() < 1e-6);
    assert!((bbox[2] - 1.0).abs() < 1e-6);
    assert!((bbox[3] - 1.0).abs() < 1e-6);
}
