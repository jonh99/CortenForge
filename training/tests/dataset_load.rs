use std::fs;
use std::path::PathBuf;

use data_contracts::capture::{CaptureMetadata, PolypLabel};
use image::{Rgb, RgbImage};
use training::{collate, DatasetConfig};

#[test]
fn load_and_collate_synthetic() {
    let temp = tempfile::tempdir().unwrap();
    let labels_dir = temp.path().join("labels");
    fs::create_dir_all(&labels_dir).unwrap();
    let meta = CaptureMetadata {
        frame_id: 1,
        sim_time: 0.0,
        unix_time: 0.0,
        image: "frame_00001.png".into(),
        image_present: true,
        camera_active: true,
        polyp_seed: 42,
        polyp_labels: vec![PolypLabel {
            center_world: [0.0, 0.0, 0.0],
            bbox_px: Some([0.0, 0.0, 10.0, 10.0]),
            bbox_norm: Some([0.1, 0.1, 0.2, 0.2]),
        }],
    };
    let json = serde_json::to_vec(&meta).unwrap();
    fs::write(labels_dir.join("frame_00001.json"), json).unwrap();

    // Write a tiny 2x2 RGB image.
    let mut img = RgbImage::new(2, 2);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 0, 0]);
    }
    let img_path = temp.path().join("frame_00001.png");
    img.save(&img_path).unwrap();

    let cfg = DatasetConfig {
        root: PathBuf::from(temp.path()),
        labels_subdir: "labels".into(),
        images_subdir: ".".into(),
    };
    let samples = cfg.load().unwrap();
    assert_eq!(samples.len(), 1);

    // Collate with NdArray backend
    let batch = collate::<burn_ndarray::NdArray<f32>>(&samples, 4).unwrap();
    assert_eq!(batch.images.dims(), [1, 3, 2, 2]);
    assert_eq!(batch.boxes.dims(), [1, 4, 4]);
    assert_eq!(batch.box_mask.dims(), [1, 4]);
    assert_eq!(batch.features.dims(), [1, 8]);
    let mask: Vec<f32> = batch
        .box_mask
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    assert_eq!(mask, vec![1.0, 0.0, 0.0, 0.0]);

    // Features: mean/std RGB, aspect ratio, box count.
    let feats: Vec<f32> = batch
        .features
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    // Red image -> mean near 1.0 for R, 0 for G/B; std near 0; aspect 1.0; box_count 1.
    assert_eq!(feats.len(), 8);
    assert!((feats[0] - 1.0).abs() < 1e-5);
    assert!((feats[1] - 0.0).abs() < 1e-5);
    assert!((feats[2] - 0.0).abs() < 1e-5);
    assert!((feats[3]).abs() < 1e-5);
    assert!((feats[4]).abs() < 1e-5);
    assert!((feats[5]).abs() < 1e-5);
    assert!((feats[6] - 1.0).abs() < 1e-5); // aspect ratio
    assert!((feats[7] - 1.0).abs() < 1e-5); // box count
}
