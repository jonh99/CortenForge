#![cfg(feature = "burn_runtime")]

use colon_sim::tools::burn_dataset::{
    CacheableTransformConfig, DatasetSummary, Endianness, RunSummary, ShardDType, ShardMetadata,
    ValidationThresholds, WarehouseLoaders, WarehouseManifest,
};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

fn write_minimal_shard(dir: &PathBuf) {
    let fname = dir.join("shard_00000.bin");
    let mut f = File::create(&fname).unwrap();
    // Header constants
    // magic
    f.write_all(b"TWH1").unwrap();
    // shard_version
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // dtype f32
    f.write_all(&0u32.to_le_bytes()).unwrap();
    // endianness little
    f.write_all(&0u32.to_le_bytes()).unwrap();
    // width, height, channels
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    // max_boxes
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // samples
    f.write_all(&1u64.to_le_bytes()).unwrap();
    // offsets
    let image_offset = 80u64;
    let boxes_offset = image_offset + 3 * 4;
    let mask_offset = boxes_offset + 4 * 4;
    f.write_all(&image_offset.to_le_bytes()).unwrap();
    f.write_all(&boxes_offset.to_le_bytes()).unwrap();
    f.write_all(&mask_offset.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap(); // meta_offset
    f.write_all(&0u64.to_le_bytes()).unwrap(); // checksum_offset
    // payload: images (3 floats)
    for v in [0.0f32, 1.0, 2.0] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    // boxes (4 floats)
    for v in [0.1f32, 0.2, 0.3, 0.4] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    // mask (1 float)
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
}

fn make_manifest(dir: &PathBuf) -> PathBuf {
    let shard_meta = ShardMetadata {
        id: "00000".into(),
        relative_path: "shard_00000.bin".into(),
        shard_version: 1,
        samples: 1,
        width: 1,
        height: 1,
        channels: 3,
        max_boxes: 1,
        checksum_sha256: None,
        dtype: ShardDType::F32,
        endianness: Endianness::Little,
    };
    let summary = DatasetSummary {
        runs: vec![RunSummary {
            run_dir: PathBuf::from("run"),
            total: 1,
            non_empty: 1,
            empty: 0,
            missing_image: 0,
            missing_file: 0,
            invalid: 0,
        }],
        totals: RunSummary {
            run_dir: PathBuf::new(),
            total: 1,
            non_empty: 1,
            empty: 0,
            missing_image: 0,
            missing_file: 0,
            invalid: 0,
        },
    };
    let thresholds = ValidationThresholds::default();
    let transform = CacheableTransformConfig {
        target_size: Some((1, 1)),
        resize_mode: colon_sim::tools::burn_dataset::ResizeMode::Letterbox,
        max_boxes: 1,
    };
    let version = WarehouseManifest::compute_version(dir, &transform, true, "test");
    let manifest = WarehouseManifest::new(
        dir.clone(),
        transform,
        version.clone(),
        "test".into(),
        "test".into(),
        vec![shard_meta],
        summary,
        thresholds,
    );
    let path = dir.join("manifest.json");
    manifest.save(&path).unwrap();
    path
}

#[test]
fn store_modes_len_match() {
    let tempdir = std::env::temp_dir().join("warehouse_store_test");
    let _ = fs::remove_dir_all(&tempdir);
    fs::create_dir_all(&tempdir).unwrap();
    write_minimal_shard(&tempdir);
    let manifest_path = make_manifest(&tempdir);

    let modes = ["memory", "mmap", "stream"];
    let mut lengths = Vec::new();
    for m in modes.iter() {
        std::env::set_var("WAREHOUSE_STORE", m);
        let loaders =
            WarehouseLoaders::from_manifest_path(manifest_path.as_path(), 0.0, None, false)
                .unwrap();
        lengths.push((loaders.train_len(), loaders.val_len()));
    }
    // all modes should see the same lengths
    assert!(lengths.windows(2).all(|w| w[0] == w[1]));
}
