use anyhow::Context;
use burn_dataset::{
    index_runs, load_sample_for_etl, summarize_root_with_thresholds, CacheableTransformConfig,
    DatasetConfig, DatasetSample, Endianness, ResizeMode, ShardDType, ShardMetadata,
    ValidationThresholds, WarehouseManifest,
};
use clap::Parser;
use rayon::prelude::*;
use sha2::Digest;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "warehouse_etl",
    about = "Build tensor warehouse shards + manifest"
)]
struct Args {
    /// Input root containing capture runs.
    #[arg(long, default_value = "assets/datasets/captures_filtered")]
    input_root: PathBuf,
    /// Output root for the warehouse artifacts.
    #[command(flatten)]
    output: cli_support::common::WarehouseOutputArgs,
    /// Target size WxH (e.g., 256x256).
    #[arg(long, value_parser = parse_target_size, default_value = "384x384")]
    target_size: (u32, u32),
    /// Resize mode: force or letterbox.
    #[arg(long, value_parser = ["force", "letterbox"], default_value = "letterbox")]
    resize_mode: String,
    /// Maximum boxes per sample.
    #[arg(long, default_value_t = 16)]
    max_boxes: usize,
    /// Shard size in samples (approx; final shard may be smaller).
    #[arg(long, default_value_t = 1024)]
    shard_samples: usize,
    /// Skip samples with no boxes.
    #[arg(long, default_value_t = true)]
    skip_empty: bool,
    /// Dtype for shards: f32 (default). f16 not yet supported.
    #[arg(long, value_parser = ["f32"], default_value = "f32")]
    dtype: String,
}

fn parse_target_size(s: &str) -> Result<(u32, u32), String> {
    let parts: Vec<_> = s.split('x').collect();
    if parts.len() != 2 {
        return Err("expected WxH".into());
    }
    let w = parts[0]
        .parse::<u32>()
        .map_err(|_| "width must be u32".to_string())?;
    let h = parts[1]
        .parse::<u32>()
        .map_err(|_| "height must be u32".to_string())?;
    Ok((w, h))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.dtype != "f32" {
        anyhow::bail!("only f32 shards are supported for now");
    }

    let input_root = args.input_root.as_path();
    let output_root = args.output.output_root.as_path();
    fs::create_dir_all(output_root)
        .with_context(|| format!("creating output root {}", output_root.display()))?;

    // Validation pass (summary + thresholds).
    let thresholds = ValidationThresholds::from_env();
    let report = summarize_root_with_thresholds(input_root, &thresholds)?;
    println!(
        "Validation outcome: {} (runs={} total={} non_empty={} empty={} missing_image={} missing_file={} invalid={})",
        report.outcome.as_str(),
        report.summary.runs.len(),
        report.summary.totals.total,
        report.summary.totals.non_empty,
        report.summary.totals.empty,
        report.summary.totals.missing_image,
        report.summary.totals.missing_file,
        report.summary.totals.invalid
    );
    for run in report.summary.runs.iter() {
        println!(
            " - {}: total={} non_empty={} empty={} missing_image={} missing_file={} invalid={}",
            run.run_dir.display(),
            run.total,
            run.non_empty,
            run.empty,
            run.missing_image,
            run.missing_file,
            run.invalid
        );
    }
    if report.outcome == burn_dataset::ValidationOutcome::Fail {
        anyhow::bail!("Validation failed; see above.");
    }

    let resize_mode = match args.resize_mode.as_str() {
        "force" => ResizeMode::Force,
        _ => ResizeMode::Letterbox,
    };
    let cfg = DatasetConfig {
        target_size: Some(args.target_size),
        resize_mode,
        max_boxes: args.max_boxes,
        skip_empty_labels: args.skip_empty,
        flip_horizontal_prob: 0.0,
        color_jitter_prob: 0.0,
        color_jitter_strength: 0.0,
        scale_jitter_prob: 0.0,
        noise_prob: 0.0,
        blur_prob: 0.0,
        drop_last: false,
        shuffle: false,
        ..Default::default()
    };
    let pipeline = cfg
        .transform
        .clone()
        .unwrap_or_else(|| burn_dataset::TransformPipeline::from_config(&cfg));

    let code_version = WarehouseManifest::resolve_code_version();
    let version = WarehouseManifest::compute_version(
        input_root,
        &CacheableTransformConfig {
            target_size: Some(args.target_size),
            resize_mode,
            max_boxes: args.max_boxes,
        },
        args.skip_empty,
        &code_version,
    );
    let version_root = output_root.join(format!("v{}", version));
    let clear_requested = env::var("WAREHOUSE_CLEAR")
        .ok()
        .map(|v| v != "0" && !v.trim().is_empty())
        .unwrap_or(false);
    if clear_requested && version_root.exists() {
        println!(
            "WAREHOUSE_CLEAR set; removing existing version root {}",
            version_root.display()
        );
        fs::remove_dir_all(&version_root)
            .with_context(|| format!("clearing version root {}", version_root.display()))?;
    }
    fs::create_dir_all(&version_root)
        .with_context(|| format!("creating versioned root {}", version_root.display()))?;
    let manifest_path = version_root.join("manifest.json");
    let skip_if_exists = env::var("WAREHOUSE_SKIP_IF_EXISTS")
        .ok()
        .map(|v| v != "0" && !v.trim().is_empty())
        .unwrap_or(false);
    if skip_if_exists && manifest_path.exists() {
        println!(
            "Manifest already exists at {}; WAREHOUSE_SKIP_IF_EXISTS set, exiting without rebuild.",
            manifest_path.display()
        );
        return Ok(());
    }
    println!(
        "Warehouse version {} (code_version={}); writing shards under {}",
        version,
        code_version,
        version_root.display()
    );

    let indices = index_runs(input_root)?;
    if indices.is_empty() {
        anyhow::bail!("No label files found under {}", input_root.display());
    }

    let mut shards = Vec::<ShardMetadata>::new();
    let mut shard_counter = 0usize;
    let trace_path = std::env::var("WAREHOUSE_TRACE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| Path::new(&s).to_path_buf());
    let mut trace_file = trace_path.as_ref().and_then(|p| {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        File::create(p).ok().map(BufWriter::new)
    });

    for chunk in indices.chunks(args.shard_samples) {
        let decode_start = Instant::now();
        let loaded: Vec<_> = chunk
            .par_iter()
            .filter_map(|idx| match load_sample_for_etl(idx, &pipeline) {
                Ok(s) => {
                    if args.skip_empty && s.boxes.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                }
                Err(e) => {
                    eprintln!("Warning: skipping {}: {e:?}", idx.label_path.display());
                    None
                }
            })
            .collect();
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        if loaded.is_empty() {
            continue;
        }
        let size = (loaded[0].width, loaded[0].height);
        if loaded.iter().any(|s| (s.width, s.height) != size) {
            anyhow::bail!("varying image sizes found within shard chunk");
        }

        let write_start = Instant::now();
        let meta = write_shard(
            &loaded,
            shard_counter,
            &version_root,
            size,
            args.max_boxes,
            ShardDType::F32,
            Endianness::Little,
        )?;
        let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;

        if let Some(tf) = trace_file.as_mut() {
            let rec = serde_json::json!({
                "shard": meta.id,
                "samples": meta.samples,
                "decode_ms": decode_ms,
                "write_ms": write_ms
            });
            let _ = writeln!(tf, "{}", rec);
        }

        eprintln!(
            "[warehouse] shard {} samples={} decode_ms={:.1} write_ms={:.1}",
            meta.id, meta.samples, decode_ms, write_ms
        );
        shards.push(meta);
        shard_counter += 1;
    }

    let manifest = WarehouseManifest::new(
        input_root.to_path_buf(),
        CacheableTransformConfig {
            target_size: Some(args.target_size),
            resize_mode,
            max_boxes: args.max_boxes,
        },
        version,
        "sha256(dataset_root + cacheable_transform + max_boxes + skip_empty + code_version)"
            .to_string(),
        code_version,
        shards,
        report.summary,
        thresholds,
    );
    manifest.save(&manifest_path)?;
    println!(
        "Wrote manifest {} with {} shards",
        manifest_path.display(),
        manifest.shards.len()
    );

    Ok(())
}

fn write_bytes<W: Write>(mut w: W, hasher: &mut sha2::Sha256, bytes: &[u8]) -> anyhow::Result<()> {
    hasher.update(bytes);
    w.write_all(bytes)?;
    Ok(())
}

fn write_u32<W: Write>(w: &mut W, hasher: &mut sha2::Sha256, v: u32) -> anyhow::Result<()> {
    write_bytes(w, hasher, &v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, hasher: &mut sha2::Sha256, v: u64) -> anyhow::Result<()> {
    write_bytes(w, hasher, &v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, hasher: &mut sha2::Sha256, v: f32) -> anyhow::Result<()> {
    write_bytes(w, hasher, &v.to_le_bytes())
}

fn write_shard(
    samples: &[DatasetSample],
    shard_counter: usize,
    output_root: &Path,
    expected_size: (u32, u32),
    max_boxes: usize,
    dtype: ShardDType,
    endianness: Endianness,
) -> anyhow::Result<ShardMetadata> {
    let width = expected_size.0;
    let height = expected_size.1;
    let channels = 3u32;
    let samples_len = samples.len();
    let fname = format!("shard_{:05}.bin", shard_counter);
    let out_path = output_root.join(&fname);
    let mut file = BufWriter::new(
        File::create(&out_path)
            .with_context(|| format!("creating shard {}", out_path.display()))?,
    );
    let mut hasher = sha2::Sha256::new();

    // Header
    write_bytes(&mut file, &mut hasher, b"TWH1")?;
    write_u32(&mut file, &mut hasher, 1)?; // shard_version
    write_u32(
        &mut file,
        &mut hasher,
        match dtype {
            ShardDType::F32 => 0,
            ShardDType::F16 => 1,
        },
    )?;
    write_u32(
        &mut file,
        &mut hasher,
        match endianness {
            Endianness::Little => 0,
            Endianness::Big => 1,
        },
    )?;
    write_u32(&mut file, &mut hasher, width)?;
    write_u32(&mut file, &mut hasher, height)?;
    write_u32(&mut file, &mut hasher, channels)?;
    write_u32(&mut file, &mut hasher, max_boxes as u32)?;
    write_u64(&mut file, &mut hasher, samples_len as u64)?;

    let header_size: u64 = 4 + 4 * 7 + 8 + 5 * 8; // magic + seven u32 + samples u64 + five u64
    let image_elems = samples_len * 3 * width as usize * height as usize;
    let image_bytes = image_elems * std::mem::size_of::<f32>();
    let box_elems = samples_len * max_boxes * 4;
    let box_bytes = box_elems * std::mem::size_of::<f32>();
    let mask_elems = samples_len * max_boxes;
    let _mask_bytes = mask_elems * std::mem::size_of::<f32>();

    let image_offset = header_size;
    let boxes_offset = image_offset + image_bytes as u64;
    let mask_offset = boxes_offset + box_bytes as u64;
    let meta_offset = 0u64;
    let checksum_offset = 0u64;

    write_u64(&mut file, &mut hasher, image_offset)?;
    write_u64(&mut file, &mut hasher, boxes_offset)?;
    write_u64(&mut file, &mut hasher, mask_offset)?;
    write_u64(&mut file, &mut hasher, meta_offset)?;
    write_u64(&mut file, &mut hasher, checksum_offset)?;

    // Payload: images
    for sample in samples.iter() {
        for v in sample.image_chw.iter() {
            write_f32(&mut file, &mut hasher, *v)?;
        }
    }
    // Payload: boxes
    for sample in samples.iter() {
        for i in 0..max_boxes {
            if let Some(b) = sample.boxes.get(i) {
                write_f32(&mut file, &mut hasher, b[0])?;
                write_f32(&mut file, &mut hasher, b[1])?;
                write_f32(&mut file, &mut hasher, b[2])?;
                write_f32(&mut file, &mut hasher, b[3])?;
            } else {
                write_f32(&mut file, &mut hasher, 0.0)?;
                write_f32(&mut file, &mut hasher, 0.0)?;
                write_f32(&mut file, &mut hasher, 0.0)?;
                write_f32(&mut file, &mut hasher, 0.0)?;
            }
        }
    }
    // Payload: mask
    for sample in samples.iter() {
        for i in 0..max_boxes {
            let m = if i < sample.boxes.len() { 1.0 } else { 0.0 };
            write_f32(&mut file, &mut hasher, m)?;
        }
    }
    file.flush()?;
    let checksum = hasher.finalize();
    let checksum_hex = format!("{:x}", checksum);

    let meta = ShardMetadata {
        id: format!("{:05}", shard_counter),
        relative_path: fname,
        shard_version: 1,
        samples: samples_len,
        width,
        height,
        channels,
        max_boxes,
        checksum_sha256: Some(checksum_hex),
        dtype,
        endianness,
    };
    Ok(meta)
}
