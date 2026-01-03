#![cfg(feature = "burn_runtime")]

use burn::backend::ndarray::NdArray;
use burn_dataset::{
    CacheableTransformConfig, DatasetSummary, Endianness, RunSummary, ShardDType, ShardMetadata,
    ValidationThresholds, WarehouseLoaders, WarehouseManifest,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::tempdir;

fn write_minimal_shard(dir: &Path, samples: usize) {
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
    let max_boxes = 1u32;
    f.write_all(&max_boxes.to_le_bytes()).unwrap();
    // samples
    f.write_all(&(samples as u64).to_le_bytes()).unwrap();
    // offsets
    let image_offset = 80u64;
    let boxes_offset = image_offset + (3 * 4 * samples as u64);
    let mask_offset = boxes_offset + (4 * 4 * samples as u64); // 4 coords * f32 per sample (max_boxes=1)
    f.write_all(&image_offset.to_le_bytes()).unwrap();
    f.write_all(&boxes_offset.to_le_bytes()).unwrap();
    f.write_all(&mask_offset.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap(); // meta_offset
    f.write_all(&0u64.to_le_bytes()).unwrap(); // checksum_offset
    // payload: images (3 floats)
    for s in 0..samples {
        for v in [0.0f32 + s as f32, 1.0 + s as f32, 2.0 + s as f32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    // boxes (4 floats)
    for s in 0..samples {
        for v in [0.1f32 + s as f32 * 0.01, 0.2, 0.3, 0.4] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    // mask (1 float)
    for _ in 0..samples {
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
    }
}

fn make_manifest(dir: &Path, samples: usize) -> PathBuf {
    let shard_meta = ShardMetadata {
        id: "00000".into(),
        relative_path: "shard_00000.bin".into(),
        shard_version: 1,
        samples,
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
            total: samples,
            non_empty: samples,
            empty: 0,
            missing_image: 0,
            missing_file: 0,
            invalid: 0,
        }],
        totals: RunSummary {
            run_dir: PathBuf::new(),
            total: samples,
            non_empty: samples,
            empty: 0,
            missing_image: 0,
            missing_file: 0,
            invalid: 0,
        },
    };
    let thresholds = ValidationThresholds::default();
    let transform = CacheableTransformConfig {
        target_size: Some((1, 1)),
        resize_mode: burn_dataset::ResizeMode::Letterbox,
        max_boxes: 1,
    };
    let version = WarehouseManifest::compute_version(dir, &transform, true, "test");
    let manifest = WarehouseManifest::new(
        dir.to_path_buf(),
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
    let tempdir = tempdir().unwrap();
    let base = tempdir.path();
    write_minimal_shard(base, 1);
    let manifest_path = make_manifest(base, 1);

    let modes = ["memory", "mmap", "stream"];
    let mut lengths = Vec::new();
    for m in modes.iter() {
        unsafe {
            std::env::set_var("WAREHOUSE_STORE", m);
        }
        let loaders =
            WarehouseLoaders::from_manifest_path(manifest_path.as_path(), 0.0, None, false)
                .unwrap();
        lengths.push((loaders.train_len(), loaders.val_len()));
    }
    // all modes should see the same lengths
    assert!(lengths.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn streaming_vs_ram_throughput_smoke() {
    let tempdir = tempdir().unwrap();
    let base = tempdir.path();
    let samples = 16usize;
    write_minimal_shard(base, samples);
    let manifest_path = make_manifest(base, samples);
    let modes = ["memory", "stream", "mmap"];
    for m in modes.iter() {
        unsafe {
            std::env::set_var("WAREHOUSE_STORE", m);
        }
        let loaders =
            WarehouseLoaders::from_manifest_path(manifest_path.as_path(), 0.0, None, false)
                .unwrap();
        let start = Instant::now();
        let device = <NdArray<f32> as burn::tensor::backend::Backend>::Device::default();
        let mut iter = loaders.train_iter();
        let mut seen = 0usize;
        while let Some(batch) = iter.next_batch::<NdArray<f32>>(4, &device).unwrap() {
            let batch_size = batch.images.dims()[0];
            seen += batch_size;
        }
        let elapsed = start.elapsed().as_millis();
        assert_eq!(seen, samples);
        eprintln!(
            "[warehouse][{}] samples={} elapsed_ms={}",
            m, samples, elapsed
        );
    }
}

#[test]
fn streaming_bench_optional() {
    if std::env::var("STREAM_BENCH").ok().as_deref() != Some("1") {
        return;
    }
    let sizes = [16usize, 256usize];
    let modes = ["memory", "stream", "mmap"];
    for &samples in sizes.iter() {
        let tempdir = tempdir().unwrap();
        let base = tempdir.path();
        write_minimal_shard(base, samples);
        let manifest_path = make_manifest(base, samples);
        for m in modes.iter() {
            unsafe {
                std::env::set_var("WAREHOUSE_STORE", m);
            }
            let loaders =
                WarehouseLoaders::from_manifest_path(manifest_path.as_path(), 0.0, None, false)
                    .unwrap();
            let start = Instant::now();
            let device = <NdArray<f32> as burn::tensor::backend::Backend>::Device::default();
            let mut iter = loaders.train_iter();
            let mut seen = 0usize;
            while let Some(batch) = iter.next_batch::<NdArray<f32>>(32, &device).unwrap() {
                let batch_size = batch.images.dims()[0];
                seen += batch_size;
            }
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(seen, samples);
            let img_per_s = samples as f64 / elapsed.max(1e-6);
            eprintln!(
                "[warehouse][bench][{}] samples={} elapsed_s={:.4} img_per_s={:.2}",
                m, samples, elapsed, img_per_s
            );
        }
    }
}
