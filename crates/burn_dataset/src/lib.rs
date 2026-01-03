#[cfg(feature = "burn_runtime")]
use crossbeam_channel::{bounded, Receiver};
use image::imageops::FilterType;
#[cfg(feature = "burn_runtime")]
use memmap2::MmapOptions;
use rand::{seq::SliceRandom, Rng, SeedableRng};
#[cfg(feature = "burn_runtime")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::fs;
#[cfg(feature = "burn_runtime")]
use std::fs::File;
#[cfg(feature = "burn_runtime")]
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(feature = "burn_runtime")]
use std::thread;
#[cfg(feature = "burn_runtime")]
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "burn_runtime")]
const DEFAULT_LOG_EVERY_SAMPLES: usize = 1000;

pub type DatasetResult<T> = Result<T, BurnDatasetError>;

#[derive(Debug, Error)]
pub enum BurnDatasetError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("label validation failed at {path}: {msg}")]
    Validation { path: PathBuf, msg: String },
    #[error("image missing for label {path}")]
    MissingImage { path: PathBuf },
    #[error("image file missing for label {path}: {image}")]
    MissingImageFile { path: PathBuf, image: PathBuf },
    #[error("image decode error at {path}: {source}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub frame_id: u64,
    /// Image in CHW layout, normalized to [0, 1].
    pub image_chw: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// Normalized bounding boxes: [x_min, y_min, x_max, y_max] in 0..1.
    pub boxes: Vec<[f32; 4]>,
}

/// Split run indices into train/val sets by ratio. Uses run directory grouping to avoid leakage.
pub fn split_runs(
    indices: Vec<SampleIndex>,
    val_ratio: f32,
) -> (Vec<SampleIndex>, Vec<SampleIndex>) {
    let mut by_run: std::collections::BTreeMap<PathBuf, Vec<SampleIndex>> =
        std::collections::BTreeMap::new();
    for idx in indices {
        by_run.entry(idx.run_dir.clone()).or_default().push(idx);
    }
    let mut runs: Vec<_> = by_run.into_iter().collect();
    if val_ratio > 0.0 {
        let mut rng = rand::thread_rng();
        runs.shuffle(&mut rng);
    }
    let total = runs.len().max(1);
    let val_count = ((val_ratio.clamp(0.0, 1.0) * total as f32).round() as usize).min(total);
    let (val_runs, train_runs) = runs.split_at(val_count);

    let mut train = Vec::new();
    let mut val = Vec::new();
    for (_, v) in train_runs {
        train.extend(v.clone());
    }
    for (_, v) in val_runs {
        val.extend(v.clone());
    }
    (train, val)
}

/// Stratified split by box count buckets (0,1,2+ boxes), seeded shuffle. Does not group by run.
pub fn split_runs_stratified(
    indices: Vec<SampleIndex>,
    val_ratio: f32,
    seed: Option<u64>,
) -> (Vec<SampleIndex>, Vec<SampleIndex>) {
    let mut buckets: [Vec<SampleIndex>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for idx in indices {
        let count = count_boxes(&idx).unwrap_or(0);
        let bucket_idx = if count == 0 {
            0
        } else if count == 1 {
            1
        } else {
            2
        };
        buckets[bucket_idx].push(idx);
    }
    let mut rng: Box<dyn rand::RngCore> = match seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };
    let mut train = Vec::new();
    let mut val = Vec::new();
    for bucket in buckets.iter_mut() {
        bucket.shuffle(&mut rng);
        let total = bucket.len();
        if total == 0 {
            continue;
        }
        let val_count = ((val_ratio.clamp(0.0, 1.0) * total as f32).round() as usize).min(total);
        let (val_bucket, train_bucket) = bucket.split_at(val_count);
        val.extend_from_slice(val_bucket);
        train.extend_from_slice(train_bucket);
    }
    (train, val)
}

pub fn count_boxes(idx: &SampleIndex) -> DatasetResult<usize> {
    let raw = fs::read(&idx.label_path).map_err(|e| BurnDatasetError::Io {
        path: idx.label_path.clone(),
        source: e,
    })?;
    let meta: LabelEntry = serde_json::from_slice(&raw).map_err(|e| BurnDatasetError::Json {
        path: idx.label_path.clone(),
        source: e,
    })?;
    Ok(meta.polyp_labels.len())
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Resize all images to this (width, height). If None, images must already share shape.
    pub target_size: Option<(u32, u32)>,
    /// How to resize images when target_size is set.
    pub resize_mode: ResizeMode,
    /// Probability of applying a horizontal flip augmentation.
    pub flip_horizontal_prob: f32,
    /// Probability of applying a light color jitter (brightness/contrast).
    pub color_jitter_prob: f32,
    /// Max jitter scale for brightness/contrast.
    pub color_jitter_strength: f32,
    /// Probability of applying a scale jitter (zoom in/out with bbox-safe padding/cropping).
    pub scale_jitter_prob: f32,
    /// Min scale factor for scale jitter.
    pub scale_jitter_min: f32,
    /// Max scale factor for scale jitter.
    pub scale_jitter_max: f32,
    /// Probability of adding uniform noise per channel.
    pub noise_prob: f32,
    /// Max absolute noise added (0-1 range).
    pub noise_strength: f32,
    /// Probability of applying a blur.
    pub blur_prob: f32,
    /// Blur sigma (passed to image::imageops::blur).
    pub blur_sigma: f32,
    /// Cap on boxes per image; extras are dropped, padding uses zeros with mask.
    pub max_boxes: usize,
    /// Shuffle samples before iteration.
    pub shuffle: bool,
    /// Seed for reproducible shuffling.
    pub seed: Option<u64>,
    /// Skip frames with no bounding boxes.
    pub skip_empty_labels: bool,
    /// Drop the last partial batch (training stability for small batches).
    pub drop_last: bool,
    /// Optional transform pipeline override; if None, built from other fields.
    pub transform: Option<TransformPipeline>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            target_size: Some((512, 512)),
            resize_mode: ResizeMode::Letterbox,
            flip_horizontal_prob: 0.0,
            color_jitter_prob: 0.0,
            color_jitter_strength: 0.1,
            scale_jitter_prob: 0.0,
            scale_jitter_min: 0.8,
            scale_jitter_max: 1.2,
            noise_prob: 0.0,
            noise_strength: 0.02,
            blur_prob: 0.0,
            blur_sigma: 1.0,
            max_boxes: 16,
            shuffle: true,
            seed: None,
            skip_empty_labels: true,
            drop_last: false,
            transform: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResizeMode {
    /// Stretch to fill the target dimensions (may distort boxes).
    Force,
    /// Preserve aspect ratio; pad to target with zeros.
    Letterbox,
}

#[derive(Debug, Clone)]
pub struct SampleIndex {
    pub run_dir: PathBuf,
    pub label_path: PathBuf,
}

#[derive(Deserialize)]
struct LabelEntry {
    frame_id: u64,
    image: String,
    image_present: bool,
    polyp_labels: Vec<PolypLabel>,
}

#[derive(Deserialize)]
struct PolypLabel {
    bbox_px: Option<[f32; 4]>,
    bbox_norm: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheableTransformConfig {
    pub target_size: Option<(u32, u32)>,
    pub resize_mode: ResizeMode,
    pub max_boxes: usize,
}

#[derive(Debug, Clone)]
pub struct TransformPipeline {
    pub cacheable: CacheableTransformConfig,
    pub target_size: Option<(u32, u32)>,
    pub resize_mode: ResizeMode,
    pub flip_horizontal_prob: f32,
    pub color_jitter_prob: f32,
    pub color_jitter_strength: f32,
    pub scale_jitter_prob: f32,
    pub scale_jitter_min: f32,
    pub scale_jitter_max: f32,
    pub noise_prob: f32,
    pub noise_strength: f32,
    pub blur_prob: f32,
    pub blur_sigma: f32,
    pub max_boxes: usize,
    pub seed: Option<u64>,
}

impl TransformPipeline {
    pub fn from_config(cfg: &DatasetConfig) -> Self {
        Self {
            cacheable: CacheableTransformConfig {
                target_size: cfg.target_size,
                resize_mode: cfg.resize_mode,
                max_boxes: cfg.max_boxes,
            },
            target_size: cfg.target_size,
            resize_mode: cfg.resize_mode,
            flip_horizontal_prob: cfg.flip_horizontal_prob,
            color_jitter_prob: cfg.color_jitter_prob,
            color_jitter_strength: cfg.color_jitter_strength,
            scale_jitter_prob: cfg.scale_jitter_prob,
            scale_jitter_min: cfg.scale_jitter_min,
            scale_jitter_max: cfg.scale_jitter_max,
            noise_prob: cfg.noise_prob,
            noise_strength: cfg.noise_strength,
            blur_prob: cfg.blur_prob,
            blur_sigma: cfg.blur_sigma,
            max_boxes: cfg.max_boxes,
            seed: cfg.seed,
        }
    }

    pub fn describe(&self) -> String {
        let ts = self
            .target_size
            .map(|(w, h)| format!("{}x{}", w, h))
            .unwrap_or_else(|| "none".to_string());
        format!(
            "target_size={} resize={:?} flip_p={:.2} color_jitter_p={:.2} strength={:.2} scale_jitter_p={:.2} range=[{:.2},{:.2}] noise_p={:.2} strength={:.3} blur_p={:.2} sigma={:.2} max_boxes={} seed={}",
            ts,
            self.resize_mode,
            self.flip_horizontal_prob,
            self.color_jitter_prob,
            self.color_jitter_strength,
            self.scale_jitter_prob,
            self.scale_jitter_min,
            self.scale_jitter_max,
            self.noise_prob,
            self.noise_strength,
            self.blur_prob,
            self.blur_sigma,
            self.max_boxes,
            self.seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "none".to_string())
        )
    }

    fn apply(&self, img: image::RgbImage, meta: &LabelEntry) -> DatasetResult<DatasetSample> {
        let (mut width, mut height) = img.dimensions();
        // Choose RNG: seeded if provided (per-frame deterministic), else thread-local.
        let mut rng_local;
        let mut seeded_rng;
        let rng: &mut dyn rand::RngCore = if let Some(seed) = self.seed {
            let mixed = seed ^ meta.frame_id;
            seeded_rng = rand::rngs::StdRng::seed_from_u64(mixed);
            &mut seeded_rng
        } else {
            rng_local = rand::thread_rng();
            &mut rng_local
        };

        if let Some((w, h)) = self.target_size {
            match self.resize_mode {
                ResizeMode::Force => {
                    width = w;
                    height = h;
                    let mut resized = image::imageops::resize(&img, w, h, FilterType::Triangle);
                    let mut boxes = normalize_boxes(&meta.polyp_labels, w, h);
                    maybe_hflip(&mut resized, &mut boxes, self.flip_horizontal_prob, rng);
                    return build_sample_from_image(
                        resized,
                        width,
                        height,
                        boxes,
                        meta.frame_id,
                        self.max_boxes,
                    );
                }
                ResizeMode::Letterbox => {
                    let (mut resized_img, pad_w, pad_h) = letterbox_resize(&img, w, h)?;
                    let scale_x = resized_img.width() as f32 / img.width() as f32;
                    let scale_y = resized_img.height() as f32 / img.height() as f32;

                    let (norm_boxes, px_boxes) =
                        normalize_boxes_with_px(&meta.polyp_labels, img.width(), img.height());

                    let mut boxes = norm_boxes
                        .into_iter()
                        .zip(px_boxes.into_iter())
                        .map(|(_norm, px)| {
                            let scaled = [
                                px[0] * scale_x + pad_w as f32,
                                px[1] * scale_y + pad_h as f32,
                                px[2] * scale_x + pad_w as f32,
                                px[3] * scale_y + pad_h as f32,
                            ];
                            [
                                scaled[0] / w as f32,
                                scaled[1] / h as f32,
                                scaled[2] / w as f32,
                                scaled[3] / h as f32,
                            ]
                        })
                        .map(|mut b| {
                            for v in b.iter_mut() {
                                *v = v.clamp(0.0, 1.0);
                            }
                            b
                        })
                        .collect::<Vec<_>>();

                    maybe_hflip(&mut resized_img, &mut boxes, self.flip_horizontal_prob, rng);
                    maybe_jitter(
                        &mut resized_img,
                        self.color_jitter_prob,
                        self.color_jitter_strength,
                        rng,
                    );
                    maybe_scale_jitter(
                        &mut resized_img,
                        &mut boxes,
                        self.scale_jitter_prob,
                        self.scale_jitter_min,
                        self.scale_jitter_max,
                        rng,
                    );
                    maybe_noise(&mut resized_img, self.noise_prob, self.noise_strength, rng);
                    maybe_blur(&mut resized_img, self.blur_prob, self.blur_sigma, rng);

                    if boxes.len() > self.max_boxes {
                        boxes.truncate(self.max_boxes);
                    }

                    return build_sample_from_image(
                        resized_img,
                        w,
                        h,
                        boxes,
                        meta.frame_id,
                        self.max_boxes,
                    );
                }
            }
        }

        let mut boxes = normalize_boxes(&meta.polyp_labels, width, height);
        let mut img = img;
        maybe_hflip(&mut img, &mut boxes, self.flip_horizontal_prob, rng);
        maybe_jitter(
            &mut img,
            self.color_jitter_prob,
            self.color_jitter_strength,
            rng,
        );
        maybe_scale_jitter(
            &mut img,
            &mut boxes,
            self.scale_jitter_prob,
            self.scale_jitter_min,
            self.scale_jitter_max,
            rng,
        );
        maybe_noise(&mut img, self.noise_prob, self.noise_strength, rng);
        maybe_blur(&mut img, self.blur_prob, self.blur_sigma, rng);
        let sample =
            build_sample_from_image(img, width, height, boxes, meta.frame_id, self.max_boxes)?;
        Ok(sample)
    }
}

#[derive(Debug, Clone)]
pub struct TransformPipelineBuilder {
    inner: TransformPipeline,
}

impl TransformPipelineBuilder {
    pub fn new() -> Self {
        Self {
            inner: TransformPipeline::from_config(&DatasetConfig::default()),
        }
    }
    pub fn target_size(mut self, size: Option<(u32, u32)>) -> Self {
        self.inner.target_size = size;
        self.inner.cacheable.target_size = size;
        self
    }
    pub fn resize_mode(mut self, mode: ResizeMode) -> Self {
        self.inner.resize_mode = mode;
        self.inner.cacheable.resize_mode = mode;
        self
    }
    pub fn flip_horizontal_prob(mut self, p: f32) -> Self {
        self.inner.flip_horizontal_prob = p;
        self
    }
    pub fn color_jitter(mut self, prob: f32, strength: f32) -> Self {
        self.inner.color_jitter_prob = prob;
        self.inner.color_jitter_strength = strength;
        self
    }
    pub fn scale_jitter(mut self, prob: f32, min: f32, max: f32) -> Self {
        self.inner.scale_jitter_prob = prob;
        self.inner.scale_jitter_min = min;
        self.inner.scale_jitter_max = max;
        self
    }
    pub fn noise(mut self, prob: f32, strength: f32) -> Self {
        self.inner.noise_prob = prob;
        self.inner.noise_strength = strength;
        self
    }
    pub fn blur(mut self, prob: f32, sigma: f32) -> Self {
        self.inner.blur_prob = prob;
        self.inner.blur_sigma = sigma;
        self
    }
    pub fn max_boxes(mut self, max_boxes: usize) -> Self {
        self.inner.max_boxes = max_boxes;
        self.inner.cacheable.max_boxes = max_boxes;
        self
    }
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.inner.seed = seed;
        self
    }
    pub fn build(self) -> TransformPipeline {
        self.inner
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_dir: PathBuf,
    pub total: usize,
    pub non_empty: usize,
    pub empty: usize,
    pub missing_image: usize,
    pub missing_file: usize,
    pub invalid: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub runs: Vec<RunSummary>,
    pub totals: RunSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationOutcome {
    Pass,
    Warn,
    Fail,
}

impl ValidationOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationOutcome::Pass => "pass",
            ValidationOutcome::Warn => "warn",
            ValidationOutcome::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationThresholds {
    pub max_invalid: Option<usize>,
    pub max_missing: Option<usize>,
    pub max_empty: Option<usize>,
    pub max_invalid_ratio: Option<f32>,
    pub max_missing_ratio: Option<f32>,
    pub max_empty_ratio: Option<f32>,
}

impl ValidationThresholds {
    pub fn from_env() -> Self {
        fn parse_usize(key: &str) -> Option<usize> {
            std::env::var(key).ok()?.parse().ok()
        }
        fn parse_ratio(key: &str) -> Option<f32> {
            std::env::var(key).ok()?.parse().ok()
        }
        ValidationThresholds {
            max_invalid: parse_usize("BURN_DATASET_MAX_INVALID"),
            max_missing: parse_usize("BURN_DATASET_MAX_MISSING"),
            max_empty: parse_usize("BURN_DATASET_MAX_EMPTY"),
            max_invalid_ratio: parse_ratio("BURN_DATASET_MAX_INVALID_RATIO"),
            max_missing_ratio: parse_ratio("BURN_DATASET_MAX_MISSING_RATIO"),
            max_empty_ratio: parse_ratio("BURN_DATASET_MAX_EMPTY_RATIO"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub outcome: ValidationOutcome,
    pub reasons: Vec<String>,
    pub summary: DatasetSummary,
}

fn apply_thresholds(
    label: &str,
    count: usize,
    ratio: f32,
    max_count: Option<usize>,
    max_ratio: Option<f32>,
    outcome: &mut ValidationOutcome,
    reasons: &mut Vec<String>,
) {
    if let Some(max) = max_count {
        if count > max {
            *outcome = ValidationOutcome::Fail;
            reasons.push(format!("{label}: {count} exceeds max {max}"));
        }
    }
    if let Some(max_r) = max_ratio {
        if ratio > max_r {
            *outcome = ValidationOutcome::Fail;
            reasons.push(format!(
                "{label}: ratio {:.3} exceeds max {:.3}",
                ratio, max_r
            ));
        }
    }
    if count > 0 {
        if *outcome == ValidationOutcome::Pass {
            *outcome = ValidationOutcome::Warn;
        }
        reasons.push(format!("{label}: {count} observed"));
    }
}

pub fn validate_summary(
    summary: DatasetSummary,
    thresholds: &ValidationThresholds,
) -> ValidationReport {
    let totals = &summary.totals;
    let checked = totals.total + totals.missing_file + totals.missing_image + totals.invalid;
    let denom = checked.max(1) as f32;
    let missing = totals.missing_file + totals.missing_image;

    let mut outcome = ValidationOutcome::Pass;
    let mut reasons = Vec::new();

    apply_thresholds(
        "missing (image/file)",
        missing,
        missing as f32 / denom,
        thresholds.max_missing,
        thresholds.max_missing_ratio,
        &mut outcome,
        &mut reasons,
    );
    apply_thresholds(
        "invalid labels",
        totals.invalid,
        totals.invalid as f32 / denom,
        thresholds.max_invalid,
        thresholds.max_invalid_ratio,
        &mut outcome,
        &mut reasons,
    );
    apply_thresholds(
        "empty labels",
        totals.empty,
        totals.empty as f32 / denom,
        thresholds.max_empty,
        thresholds.max_empty_ratio,
        &mut outcome,
        &mut reasons,
    );

    ValidationReport {
        outcome,
        reasons,
        summary,
    }
}

pub fn summarize_with_thresholds(
    indices: &[SampleIndex],
    thresholds: &ValidationThresholds,
) -> DatasetResult<ValidationReport> {
    let summary = summarize_runs(indices)?;
    Ok(validate_summary(summary, thresholds))
}

pub fn summarize_root_with_thresholds(
    root: &Path,
    thresholds: &ValidationThresholds,
) -> DatasetResult<ValidationReport> {
    let indices = index_runs(root)?;
    summarize_with_thresholds(&indices, thresholds)
}

/// Public entrypoint for deterministic sample loading (used by the warehouse ETL).
pub fn load_sample_for_etl(
    idx: &SampleIndex,
    pipeline: &TransformPipeline,
) -> DatasetResult<DatasetSample> {
    load_sample(idx, pipeline)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub id: String,
    /// Path to the shard file, relative to the warehouse root (UTF-8).
    pub relative_path: String,
    /// Shard format version (for binary header layout).
    pub shard_version: u32,
    pub samples: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub max_boxes: usize,
    /// Hex-encoded SHA256 of the shard contents (optional until populated).
    pub checksum_sha256: Option<String>,
    pub dtype: ShardDType,
    pub endianness: Endianness,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ShardDType {
    F32,
    F16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Endianness {
    Little,
    Big,
}

/// Storage backend for warehouse shards.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WarehouseStoreMode {
    InMemory,
    Mmap,
    Streaming,
}

impl WarehouseStoreMode {
    pub fn from_env() -> Self {
        match std::env::var("WAREHOUSE_STORE")
            .unwrap_or_else(|_| "memory".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "mmap" | "mapped" | "mmaped" => WarehouseStoreMode::Mmap,
            "stream" | "streaming" => WarehouseStoreMode::Streaming,
            _ => WarehouseStoreMode::InMemory,
        }
    }

    pub fn prefetch_from_env() -> usize {
        std::env::var("WAREHOUSE_PREFETCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarehouseManifest {
    /// Source dataset root as a UTF-8 string.
    pub dataset_root: String,
    pub transform: CacheableTransformConfig,
    /// Warehouse version key (hex-encoded SHA256 of source + config tuple).
    pub version: String,
    /// Version recipe: sha256(dataset_root + cacheable_transform + max_boxes + skip_empty + code_version).
    pub version_recipe: String,
    /// Code version used in the key (crate version or VCS hash).
    pub code_version: String,
    /// Default shard dtype for this manifest.
    pub default_dtype: ShardDType,
    /// Default shard format version.
    pub default_shard_version: u32,
    pub created_at_ms: u64,
    pub shards: Vec<ShardMetadata>,
    pub summary: DatasetSummary,
    pub thresholds: ValidationThresholds,
}

impl WarehouseManifest {
    /// Default code version string (crate version).
    pub fn default_code_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Resolve code version with optional override (e.g., git hash).
    pub fn resolve_code_version() -> String {
        if let Ok(val) = std::env::var("CODE_VERSION") {
            if !val.trim().is_empty() {
                return val;
            }
        }
        Self::default_code_version()
    }

    /// Compute a canonical warehouse version (SHA256 hex) from inputs.
    pub fn compute_version(
        dataset_root: &Path,
        transform: &CacheableTransformConfig,
        skip_empty: bool,
        code_version: &str,
    ) -> String {
        #[derive(Serialize)]
        struct VersionTuple<'a> {
            dataset_root: &'a str,
            target_size: Option<(u32, u32)>,
            resize_mode: &'a ResizeMode,
            max_boxes: usize,
            skip_empty: bool,
            code_version: &'a str,
        }
        let tuple = VersionTuple {
            dataset_root: &dataset_root.display().to_string(),
            target_size: transform.target_size,
            resize_mode: &transform.resize_mode,
            max_boxes: transform.max_boxes,
            skip_empty,
            code_version,
        };
        let bytes = serde_json::to_vec(&tuple).unwrap_or_default();
        use sha2::Digest;
        let hash = sha2::Sha256::digest(bytes);
        format!("{:x}", hash)
    }

    pub fn save(&self, path: &Path) -> DatasetResult<()> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| BurnDatasetError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let data =
            serde_json::to_vec_pretty(self).map_err(|e| BurnDatasetError::Other(e.to_string()))?;
        fs::write(path, data).map_err(|e| BurnDatasetError::Io {
            path: path.to_path_buf(),
            source: e,
        })
    }

    pub fn load(path: &Path) -> DatasetResult<Self> {
        let raw = fs::read(path).map_err(|e| BurnDatasetError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        serde_json::from_slice(&raw).map_err(|e| BurnDatasetError::Json {
            path: path.to_path_buf(),
            source: e,
        })
    }

    pub fn new(
        dataset_root: PathBuf,
        transform: CacheableTransformConfig,
        version: String,
        version_recipe: String,
        code_version: String,
        shards: Vec<ShardMetadata>,
        summary: DatasetSummary,
        thresholds: ValidationThresholds,
    ) -> Self {
        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_default();
        Self {
            dataset_root: dataset_root.display().to_string(),
            transform,
            version,
            version_recipe,
            code_version,
            default_dtype: ShardDType::F32,
            default_shard_version: 1,
            created_at_ms,
            shards,
            summary,
            thresholds,
        }
    }
}

#[cfg(feature = "burn_runtime")]
pub fn build_train_val_iters(
    root: &Path,
    val_ratio: f32,
    train_cfg: DatasetConfig,
    val_cfg: Option<DatasetConfig>,
) -> DatasetResult<(BatchIter, BatchIter)> {
    let indices = index_runs(root)?;
    let (train_idx, val_idx) = split_runs(indices, val_ratio);
    // For val, default to no shuffle/aug.
    let val_cfg = val_cfg.unwrap_or_else(|| DatasetConfig {
        shuffle: false,
        drop_last: false,
        flip_horizontal_prob: 0.0,
        color_jitter_prob: 0.0,
        color_jitter_strength: 0.0,
        scale_jitter_prob: 0.0,
        noise_prob: 0.0,
        blur_prob: 0.0,
        ..train_cfg.clone()
    });
    // Train typically shuffles; keep whatever caller set.
    let train_iter = BatchIter::from_indices(train_idx, train_cfg)?;
    let val_iter = BatchIter::from_indices(val_idx, val_cfg)?;
    Ok((train_iter, val_iter))
}

fn validate_label_entry(meta: &LabelEntry, path: &Path) -> DatasetResult<()> {
    if meta.image.trim().is_empty() {
        return Err(BurnDatasetError::Validation {
            path: path.to_path_buf(),
            msg: "missing image filename".to_string(),
        });
    }
    if meta.polyp_labels.is_empty() {
        return Err(BurnDatasetError::Validation {
            path: path.to_path_buf(),
            msg: "no polyp_labels entries".to_string(),
        });
    }
    for (i, lbl) in meta.polyp_labels.iter().enumerate() {
        match (lbl.bbox_norm, lbl.bbox_px) {
            (None, None) => {
                return Err(BurnDatasetError::Validation {
                    path: path.to_path_buf(),
                    msg: format!("polyp_labels[{i}] has no bbox_norm or bbox_px"),
                });
            }
            (Some(b), _) => {
                if b.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0) {
                    return Err(BurnDatasetError::Validation {
                        path: path.to_path_buf(),
                        msg: format!("polyp_labels[{i}] bbox_norm out of [0,1]"),
                    });
                }
                if b[0] >= b[2] || b[1] >= b[3] {
                    return Err(BurnDatasetError::Validation {
                        path: path.to_path_buf(),
                        msg: format!("polyp_labels[{i}] bbox_norm min>=max ({b:?})"),
                    });
                }
            }
            (_, Some(b)) => {
                if b.iter().any(|v| !v.is_finite()) {
                    return Err(BurnDatasetError::Validation {
                        path: path.to_path_buf(),
                        msg: format!("polyp_labels[{i}] bbox_px contains non-finite values"),
                    });
                }
                if b[0] >= b[2] || b[1] >= b[3] {
                    return Err(BurnDatasetError::Validation {
                        path: path.to_path_buf(),
                        msg: format!("polyp_labels[{i}] bbox_px min>=max ({b:?})"),
                    });
                }
            }
        }
    }
    Ok(())
}

fn label_has_box(meta: &LabelEntry) -> bool {
    meta.polyp_labels.iter().any(|lbl| {
        lbl.bbox_norm
            .is_some_and(|b| b.iter().all(|v| v.is_finite()) && b[0] < b[2] && b[1] < b[3])
            || lbl
                .bbox_px
                .is_some_and(|b| b.iter().all(|v| v.is_finite()) && b[0] < b[2] && b[1] < b[3])
    })
}

#[cfg(all(test, feature = "burn_runtime"))]
mod streaming_tests {
    use super::*;

    #[test]
    fn streamed_matches_owned_for_small_shard() {
        let tmp = tempfile::tempdir().unwrap();
        let shard_path = tmp.path().join("shard.bin");
        let width = 2u32;
        let height = 1u32;
        let channels = 3u32;
        let max_boxes = 1usize;
        let samples = 2usize;
        let header_len = 64usize;
        let img_bytes = samples * (width as usize) * (height as usize) * channels as usize * 4;
        let box_bytes = samples * max_boxes * 4 * 4;
        let mask_bytes = samples * max_boxes * 4;
        let image_offset = header_len;
        let boxes_offset = image_offset + img_bytes;
        let mask_offset = boxes_offset + box_bytes;
        let mut data = vec![0u8; header_len + img_bytes + box_bytes + mask_bytes];
        data[0..4].copy_from_slice(b"TWH1");
        data[4..8].copy_from_slice(&(1u32).to_le_bytes()); // shard_version
        data[8..12].copy_from_slice(&(0u32).to_le_bytes()); // dtype f32
        data[12..16].copy_from_slice(&(0u32).to_le_bytes()); // endianness little
        data[16..20].copy_from_slice(&width.to_le_bytes());
        data[20..24].copy_from_slice(&height.to_le_bytes());
        data[24..28].copy_from_slice(&channels.to_le_bytes());
        data[28..32].copy_from_slice(&(max_boxes as u32).to_le_bytes());
        data[32..40].copy_from_slice(&(samples as u64).to_le_bytes());
        data[40..48].copy_from_slice(&(image_offset as u64).to_le_bytes());
        data[48..56].copy_from_slice(&(boxes_offset as u64).to_le_bytes());
        data[56..64].copy_from_slice(&(mask_offset as u64).to_le_bytes());

        // Fill images.
        let mut cursor = image_offset;
        for sample in 0..samples {
            for _ in 0..(width * height * channels) {
                let val = (sample + 1) as f32;
                data[cursor..cursor + 4].copy_from_slice(&val.to_le_bytes());
                cursor += 4;
            }
        }
        // Fill boxes.
        let mut cursor = boxes_offset;
        let boxes = [[0.0f32, 0.0, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]];
        for b in &boxes {
            for v in b {
                data[cursor..cursor + 4].copy_from_slice(&v.to_le_bytes());
                cursor += 4;
            }
        }
        // Fill masks.
        let mut cursor = mask_offset;
        for m in [1.0f32, 0.0f32] {
            data[cursor..cursor + 4].copy_from_slice(&m.to_le_bytes());
            cursor += 4;
        }

        std::fs::write(&shard_path, data).unwrap();

        let meta = ShardMetadata {
            id: "test".into(),
            relative_path: shard_path
                .strip_prefix(tmp.path())
                .unwrap()
                .to_string_lossy()
                .to_string(),
            shard_version: 1,
            samples,
            width,
            height,
            channels,
            max_boxes,
            checksum_sha256: None,
            dtype: ShardDType::F32,
            endianness: Endianness::Little,
        };

        let owned = load_shard_owned(tmp.path(), &meta).expect("owned load");
        let streamed = load_shard_streamed(tmp.path(), &meta).expect("streamed load");
        assert_eq!(owned.samples, streamed.samples);
        assert_eq!(owned.width, streamed.width);
        assert_eq!(owned.height, streamed.height);
        assert_eq!(owned.max_boxes, streamed.max_boxes);

        for idx in 0..samples {
            let mut o_img = Vec::new();
            let mut o_box = Vec::new();
            let mut o_mask = Vec::new();
            owned
                .copy_sample(idx, &mut o_img, &mut o_box, &mut o_mask)
                .unwrap();

            let mut s_img = Vec::new();
            let mut s_box = Vec::new();
            let mut s_mask = Vec::new();
            streamed
                .copy_sample(idx, &mut s_img, &mut s_box, &mut s_mask)
                .unwrap();

            assert_eq!(o_img, s_img);
            assert_eq!(o_box, s_box);
            assert_eq!(o_mask, s_mask);
        }
    }
}
pub fn summarize_runs(indices: &[SampleIndex]) -> DatasetResult<DatasetSummary> {
    let mut by_run: std::collections::BTreeMap<PathBuf, RunSummary> =
        std::collections::BTreeMap::new();
    for idx in indices {
        let entry = by_run
            .entry(idx.run_dir.clone())
            .or_insert_with(|| RunSummary {
                run_dir: idx.run_dir.clone(),
                ..Default::default()
            });
        match fs::read(&idx.label_path)
            .ok()
            .and_then(|raw| serde_json::from_slice::<LabelEntry>(&raw).ok())
        {
            None => {
                entry.invalid += 1;
            }
            Some(meta) => {
                if meta.image.trim().is_empty() {
                    entry.missing_image += 1;
                    continue;
                }
                let img_path = idx.run_dir.join(&meta.image);
                if !meta.image_present || !img_path.exists() {
                    entry.missing_file += 1;
                    continue;
                }
                entry.total += 1;
                if label_has_box(&meta) {
                    entry.non_empty += 1;
                } else {
                    entry.empty += 1;
                }
            }
        }
    }
    let mut totals = RunSummary::default();
    for summary in by_run.values() {
        totals.total += summary.total;
        totals.non_empty += summary.non_empty;
        totals.empty += summary.empty;
        totals.missing_image += summary.missing_image;
        totals.missing_file += summary.missing_file;
        totals.invalid += summary.invalid;
    }
    Ok(DatasetSummary {
        runs: by_run.into_values().collect(),
        totals,
    })
}

/// Scan a captures root (e.g., `assets/datasets/captures`) and index all label files.
pub fn index_runs(root: &Path) -> DatasetResult<Vec<SampleIndex>> {
    let mut indices = Vec::new();
    let entries = fs::read_dir(root).map_err(|e| BurnDatasetError::Io {
        path: root.to_path_buf(),
        source: e,
    })?;
    for entry in entries {
        let Ok(run) = entry else { continue };
        let run_path = run.path();
        if !run_path.is_dir() {
            continue;
        }
        let labels_dir = run_path.join("labels");
        if !labels_dir.exists() {
            continue;
        }
        let labels_iter = fs::read_dir(&labels_dir).map_err(|e| BurnDatasetError::Io {
            path: labels_dir.clone(),
            source: e,
        })?;
        for label in labels_iter {
            let Ok(label_entry) = label else { continue };
            let label_path = label_entry.path();
            if label_path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            indices.push(SampleIndex {
                run_dir: run_path.clone(),
                label_path,
            });
        }
    }
    indices.sort_by(|a, b| a.label_path.cmp(&b.label_path));
    Ok(indices)
}

/// Load a capture run into an in-memory vector (eager). Prefer `BatchIter` for large sets.
pub fn load_run_dataset(run_dir: &Path) -> DatasetResult<Vec<DatasetSample>> {
    let labels_dir = run_dir.join("labels");
    if !labels_dir.exists() {
        return Err(BurnDatasetError::Io {
            path: labels_dir.clone(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "labels directory missing"),
        });
    }

    let mut label_paths: Vec<PathBuf> = fs::read_dir(&labels_dir)
        .map_err(|e| BurnDatasetError::Io {
            path: labels_dir.clone(),
            source: e,
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    label_paths.sort();

    let cfg = DatasetConfig {
        target_size: None,
        resize_mode: ResizeMode::Force,
        flip_horizontal_prob: 0.0,
        color_jitter_prob: 0.0,
        color_jitter_strength: 0.0,
        scale_jitter_prob: 0.0,
        scale_jitter_min: 1.0,
        scale_jitter_max: 1.0,
        noise_prob: 0.0,
        noise_strength: 0.0,
        blur_prob: 0.0,
        blur_sigma: 0.0,
        max_boxes: usize::MAX,
        shuffle: false,
        seed: None,
        skip_empty_labels: true,
        drop_last: false,
        transform: None,
    };
    let pipeline = cfg
        .transform
        .clone()
        .unwrap_or_else(|| TransformPipeline::from_config(&cfg));
    let mut samples = Vec::new();
    for label_path in label_paths {
        let sample = load_sample(
            &SampleIndex {
                run_dir: run_dir.to_path_buf(),
                label_path,
            },
            &pipeline,
        )?;
        samples.push(sample);
    }

    Ok(samples)
}

fn load_sample(idx: &SampleIndex, pipeline: &TransformPipeline) -> DatasetResult<DatasetSample> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let raw = fs::read(&idx.label_path).map_err(|e| BurnDatasetError::Io {
        path: idx.label_path.clone(),
        source: e,
    })?;
    let meta: LabelEntry = serde_json::from_slice(&raw).map_err(|e| BurnDatasetError::Json {
        path: idx.label_path.clone(),
        source: e,
    })?;
    validate_label_entry(&meta, &idx.label_path)?;
    if !meta.image_present {
        return Err(BurnDatasetError::MissingImage {
            path: idx.label_path.clone(),
        });
    }

    let img_path = idx.run_dir.join(&meta.image);
    if !img_path.exists() {
        return Err(BurnDatasetError::MissingImageFile {
            path: idx.label_path.clone(),
            image: img_path,
        });
    }
    let img = image::open(&img_path)
        .map_err(|e| BurnDatasetError::Image {
            path: img_path.clone(),
            source: e,
        })?
        .to_rgb8();
    let sample = pipeline.apply(img, &meta)?;
    ONCE.call_once(|| {
        if sample.boxes.is_empty() {
            eprintln!(
                "Debug: first sample {} has 0 boxes (image {}x{})",
                idx.label_path.display(),
                sample.width,
                sample.height
            );
        } else {
            eprintln!(
                "Debug: first sample {} boxes {:?}",
                idx.label_path.display(),
                sample.boxes
            );
        }
    });
    Ok(sample)
}

fn build_sample_from_image(
    img: image::RgbImage,
    width: u32,
    height: u32,
    mut boxes: Vec<[f32; 4]>,
    frame_id: u64,
    max_boxes: usize,
) -> DatasetResult<DatasetSample> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let mut image_chw = vec![0.0f32; (width * height * 3) as usize];
    for (y, x, pixel) in img.enumerate_pixels() {
        let base = (y * width + x) as usize;
        image_chw[base] = pixel[0] as f32 / 255.0;
        image_chw[(width * height) as usize + base] = pixel[1] as f32 / 255.0;
        image_chw[2 * (width * height) as usize + base] = pixel[2] as f32 / 255.0;
    }

    if boxes.len() > max_boxes {
        boxes.truncate(max_boxes);
    }

    ONCE.call_once(|| {
        if boxes.is_empty() {
            eprintln!(
                "Debug: first sample had 0 boxes (image {}x{})",
                width, height
            );
        } else {
            eprintln!("Debug: first sample boxes {:?}", boxes);
        }
    });

    Ok(DatasetSample {
        frame_id,
        image_chw,
        width,
        height,
        boxes,
    })
}

fn letterbox_resize(
    img: &image::RgbImage,
    target_w: u32,
    target_h: u32,
) -> DatasetResult<(image::RgbImage, u32, u32)> {
    let (w, h) = img.dimensions();
    let scale = f32::min(target_w as f32 / w as f32, target_h as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);

    let pad_w = (target_w - new_w) / 2;
    let pad_h = (target_h - new_h) / 2;

    let mut canvas = image::RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut canvas, &resized, pad_w.into(), pad_h.into());

    Ok((canvas, pad_w, pad_h))
}

fn normalize_boxes(labels: &[PolypLabel], w: u32, h: u32) -> Vec<[f32; 4]> {
    labels
        .iter()
        .filter_map(|l| {
            if let Some(norm) = l.bbox_norm {
                Some(norm)
            } else {
                l.bbox_px.map(|px| {
                    [
                        px[0] / w as f32,
                        px[1] / h as f32,
                        px[2] / w as f32,
                        px[3] / h as f32,
                    ]
                })
            }
        })
        .map(|mut b| {
            for v in b.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
            b
        })
        .collect()
}

fn normalize_boxes_with_px(
    labels: &[PolypLabel],
    w: u32,
    h: u32,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut norm = Vec::new();
    let mut pxs = Vec::new();
    for l in labels {
        if let Some(px) = l.bbox_px {
            norm.push([
                px[0] / w as f32,
                px[1] / h as f32,
                px[2] / w as f32,
                px[3] / h as f32,
            ]);
            pxs.push(px);
        } else if let Some(n) = l.bbox_norm {
            norm.push(n);
            pxs.push([
                n[0] * w as f32,
                n[1] * h as f32,
                n[2] * w as f32,
                n[3] * h as f32,
            ]);
        }
    }
    for b in norm.iter_mut() {
        for v in b.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    }
    (norm, pxs)
}

pub(crate) fn maybe_hflip(
    img: &mut image::RgbImage,
    boxes: &mut Vec<[f32; 4]>,
    prob: f32,
    rng: &mut dyn rand::RngCore,
) {
    if prob <= 0.0 {
        return;
    }
    if rng.gen_range(0.0..1.0) < prob {
        image::imageops::flip_horizontal_in_place(img);
        for b in boxes.iter_mut() {
            let x0 = b[0];
            let x1 = b[2];
            b[0] = (1.0 - x1).clamp(0.0, 1.0);
            b[2] = (1.0 - x0).clamp(0.0, 1.0);
        }
    }
}

pub(crate) fn maybe_jitter(
    img: &mut image::RgbImage,
    prob: f32,
    strength: f32,
    rng: &mut dyn rand::RngCore,
) {
    if prob <= 0.0 || strength <= 0.0 {
        return;
    }
    if rng.gen_range(0.0..1.0) >= prob {
        return;
    }
    let bright = 1.0 + rng.gen_range(-strength..strength);
    let contrast = 1.0 + rng.gen_range(-strength..strength);
    for pixel in img.pixels_mut() {
        for c in 0..3 {
            let v = pixel[c] as f32 / 255.0;
            let mut v = (v - 0.5) * contrast + 0.5;
            v *= bright;
            pixel[c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
}

pub(crate) fn maybe_noise(
    img: &mut image::RgbImage,
    prob: f32,
    strength: f32,
    rng: &mut dyn rand::RngCore,
) {
    if prob <= 0.0 || strength <= 0.0 {
        return;
    }
    if rng.gen_range(0.0..1.0) >= prob {
        return;
    }
    for pixel in img.pixels_mut() {
        for c in 0..3 {
            let noise = rng.gen_range(-strength..strength);
            let v = (pixel[c] as f32 / 255.0 + noise).clamp(0.0, 1.0);
            pixel[c] = (v * 255.0) as u8;
        }
    }
}

pub(crate) fn maybe_scale_jitter(
    img: &mut image::RgbImage,
    boxes: &mut Vec<[f32; 4]>,
    prob: f32,
    min_scale: f32,
    max_scale: f32,
    rng: &mut dyn rand::RngCore,
) {
    if prob <= 0.0 || min_scale <= 0.0 || max_scale <= 0.0 {
        return;
    }
    if rng.gen_range(0.0..1.0) >= prob {
        return;
    }
    let scale = rng.gen_range(min_scale..max_scale);
    let (w, h) = img.dimensions();
    let new_w = max(1, (w as f32 * scale).round() as u32);
    let new_h = max(1, (h as f32 * scale).round() as u32);

    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);
    let mut canvas = image::RgbImage::new(w, h);

    if new_w >= w && new_h >= h {
        // crop center
        let x0 = ((new_w - w) / 2) as i64;
        let y0 = ((new_h - h) / 2) as i64;
        image::imageops::replace(&mut canvas, &resized, -x0, -y0);
        let sx = scale;
        let sy = scale;
        for b in boxes.iter_mut() {
            let mut px0 = b[0] * w as f32 * sx - x0 as f32;
            let mut py0 = b[1] * h as f32 * sy - y0 as f32;
            let mut px1 = b[2] * w as f32 * sx - x0 as f32;
            let mut py1 = b[3] * h as f32 * sy - y0 as f32;
            px0 = px0.clamp(0.0, w as f32);
            py0 = py0.clamp(0.0, h as f32);
            px1 = px1.clamp(px0, w as f32);
            py1 = py1.clamp(py0, h as f32);
            b[0] = px0 / w as f32;
            b[1] = py0 / h as f32;
            b[2] = px1 / w as f32;
            b[3] = py1 / h as f32;
        }
    } else {
        // pad center
        let x0 = ((w - new_w) / 2) as i64;
        let y0 = ((h - new_h) / 2) as i64;
        image::imageops::replace(&mut canvas, &resized, x0, y0);
        let sx = scale;
        let sy = scale;
        for b in boxes.iter_mut() {
            let mut px0 = b[0] * w as f32 * sx + x0 as f32;
            let mut py0 = b[1] * h as f32 * sy + y0 as f32;
            let mut px1 = b[2] * w as f32 * sx + x0 as f32;
            let mut py1 = b[3] * h as f32 * sy + y0 as f32;
            px0 = px0.clamp(0.0, w as f32);
            py0 = py0.clamp(0.0, h as f32);
            px1 = px1.clamp(px0, w as f32);
            py1 = py1.clamp(py0, h as f32);
            b[0] = px0 / w as f32;
            b[1] = py0 / h as f32;
            b[2] = px1 / w as f32;
            b[3] = py1 / h as f32;
        }
    }

    *img = canvas;
}

pub(crate) fn maybe_blur(
    img: &mut image::RgbImage,
    prob: f32,
    sigma: f32,
    rng: &mut dyn rand::RngCore,
) {
    if prob <= 0.0 || sigma <= 0.0 {
        return;
    }
    if rng.gen_range(0.0..1.0) >= prob {
        return;
    }
    let blurred = image::imageops::blur(img, sigma);
    *img = blurred;
}

#[cfg(test)]
mod aug_tests {
    use super::maybe_hflip;
    use rand::thread_rng;

    #[test]
    fn hflip_boxes_are_inverted() {
        let mut img = image::RgbImage::new(2, 2);
        let mut boxes = vec![[0.25, 0.0, 0.75, 1.0]];
        let mut rng = thread_rng();
        maybe_hflip(&mut img, &mut boxes, 1.0, &mut rng);
        let flipped = boxes[0];
        assert!((flipped[0] - 0.25).abs() < 1e-6);
        assert!((flipped[2] - 0.75).abs() < 1e-6);
        assert!(flipped[0] < flipped[2]);
    }
}

#[cfg(feature = "burn_runtime")]
pub struct BurnBatch<B: burn::tensor::backend::Backend> {
    pub images: burn::tensor::Tensor<B, 4>,
    pub boxes: burn::tensor::Tensor<B, 3>,
    pub box_mask: burn::tensor::Tensor<B, 2>,
    pub frame_ids: burn::tensor::Tensor<B, 1>,
}

#[cfg(feature = "burn_runtime")]
pub struct BatchIter {
    indices: Vec<SampleIndex>,
    cursor: usize,
    cfg: DatasetConfig,
    processed_samples: usize,
    processed_batches: usize,
    skipped_total: usize,
    skipped_empty: usize,
    skipped_missing: usize,
    skipped_errors: usize,
    warn_once: bool,
    warned_counts: bool,
    started: Instant,
    total_load_time: Duration,
    total_assemble_time: Duration,
    last_log: Instant,
    last_logged_samples: usize,
    log_every_samples: Option<usize>,
    permissive_errors: bool,
    images_buf: Vec<f32>,
    boxes_buf: Vec<f32>,
    mask_buf: Vec<f32>,
    frame_ids_buf: Vec<f32>,
    trace_path: Option<PathBuf>,
    trace_file: Option<std::fs::File>,
    pipeline: TransformPipeline,
}

#[cfg(feature = "burn_runtime")]
impl BatchIter {
    pub fn from_root(root: &Path, cfg: DatasetConfig) -> DatasetResult<Self> {
        let indices = index_runs(root)?;
        Self::from_indices(indices, cfg)
    }

    pub fn from_indices(mut indices: Vec<SampleIndex>, cfg: DatasetConfig) -> DatasetResult<Self> {
        use rand::seq::SliceRandom;
        let mut rng = match cfg.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        if cfg.shuffle {
            indices.shuffle(&mut rng);
        }
        let log_every_samples = match std::env::var("BURN_DATASET_LOG_EVERY") {
            Ok(val) => {
                if val.eq_ignore_ascii_case("off") || val.trim() == "0" {
                    None
                } else {
                    val.parse::<usize>().ok().filter(|v| *v > 0)
                }
            }
            Err(_) => Some(DEFAULT_LOG_EVERY_SAMPLES),
        };
        let permissive_errors = std::env::var("BURN_DATASET_PERMISSIVE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .map(|v| v == "0" || v == "false" || v == "off")
            .map(|strict| !strict)
            .unwrap_or(true);
        let warn_once = std::env::var("BURN_DATASET_WARN_ONCE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .map(|v| v == "1" || v == "true" || v == "on")
            .unwrap_or(false);
        let trace_path = std::env::var("BURN_DATASET_TRACE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from);
        let now = Instant::now();
        let pipeline = cfg
            .transform
            .clone()
            .unwrap_or_else(|| TransformPipeline::from_config(&cfg));
        Ok(Self {
            indices,
            cursor: 0,
            cfg,
            processed_samples: 0,
            processed_batches: 0,
            skipped_total: 0,
            skipped_empty: 0,
            skipped_missing: 0,
            skipped_errors: 0,
            warn_once,
            warned_counts: false,
            started: now,
            total_load_time: Duration::ZERO,
            total_assemble_time: Duration::ZERO,
            last_log: now,
            last_logged_samples: 0,
            log_every_samples,
            permissive_errors,
            images_buf: Vec::new(),
            boxes_buf: Vec::new(),
            mask_buf: Vec::new(),
            frame_ids_buf: Vec::new(),
            trace_path,
            trace_file: None,
            pipeline,
        })
    }

    pub fn next_batch<B: burn::tensor::backend::Backend>(
        &mut self,
        batch_size: usize,
        device: &B::Device,
    ) -> DatasetResult<Option<BurnBatch<B>>> {
        loop {
            if self.cursor >= self.indices.len() {
                return Ok(None);
            }
            let end = (self.cursor + batch_size).min(self.indices.len());
            let slice = &self.indices[self.cursor..end];
            self.cursor = end;

            self.images_buf.clear();
            self.boxes_buf.clear();
            self.mask_buf.clear();
            self.frame_ids_buf.clear();

            let mut expected_size: Option<(u32, u32)> = None;
            let mut skipped_empty = 0usize;
            let skipped_missing = 0usize;

            let t_load = Instant::now();
            let mut loaded: Vec<_> = slice
                .par_iter()
                .enumerate()
                .map(|(i, idx)| (i, idx, load_sample(idx, &self.pipeline)))
                .collect();
            loaded.sort_by_key(|(i, _, _)| *i);
            let load_elapsed = t_load.elapsed();

            for (_i, idx, res) in loaded {
                let sample = match res {
                    Ok(s) => s,
                    Err(e) => {
                        if self.permissive_errors {
                            if !self.warn_once {
                                eprintln!(
                                    "Warning: skipping label {}: {e}",
                                    idx.label_path.display()
                                );
                            }
                            self.skipped_errors += 1;
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                };
                if self.cfg.skip_empty_labels && sample.boxes.is_empty() {
                    if !self.warn_once {
                        eprintln!(
                            "Warning: no boxes found in {} (skipping sample)",
                            idx.label_path.display()
                        );
                    }
                    skipped_empty += 1;
                    continue;
                }

                let size = (sample.width, sample.height);
                match expected_size {
                    None => expected_size = Some(size),
                    Some(sz) if sz != size => {
                        return Err(BurnDatasetError::Other(
                            "batch contains varying image sizes; set a target_size to force consistency"
                                .to_string(),
                        ));
                    }
                    _ => {}
                }

                // Ensure backing buffers are large enough after the first sample sets size.
                if let Some((w, h)) = expected_size {
                    let elems = batch_size * 3 * w as usize * h as usize;
                    if self.images_buf.capacity() < elems {
                        self.images_buf.reserve(elems - self.images_buf.capacity());
                    }
                    let box_elems = batch_size * self.cfg.max_boxes * 4;
                    if self.boxes_buf.capacity() < box_elems {
                        self.boxes_buf
                            .reserve(box_elems - self.boxes_buf.capacity());
                        self.mask_buf
                            .reserve(box_elems / 4 - self.mask_buf.capacity());
                    }
                    if self.frame_ids_buf.capacity() < batch_size {
                        self.frame_ids_buf
                            .reserve(batch_size - self.frame_ids_buf.capacity());
                    }
                }

                self.frame_ids_buf.push(sample.frame_id as f32);
                self.images_buf.extend_from_slice(&sample.image_chw);

                let mut padded = vec![0.0f32; self.cfg.max_boxes * 4];
                let mut mask = vec![0.0f32; self.cfg.max_boxes];
                for (i, b) in sample.boxes.iter().take(self.cfg.max_boxes).enumerate() {
                    padded[i * 4] = b[0];
                    padded[i * 4 + 1] = b[1];
                    padded[i * 4 + 2] = b[2];
                    padded[i * 4 + 3] = b[3];
                    mask[i] = 1.0;
                }
                self.boxes_buf.extend_from_slice(&padded);
                self.mask_buf.extend_from_slice(&mask);
            }

            if self.images_buf.is_empty() {
                if skipped_empty > 0 || skipped_missing > 0 {
                    self.skipped_total += skipped_empty + skipped_missing;
                    self.skipped_empty += skipped_empty;
                    self.skipped_missing += skipped_missing;
                    continue;
                } else {
                    return Ok(None);
                }
            }

            let t_assemble = Instant::now();
            let (width, height) = expected_size.expect("batch size > 0 ensures size is set");
            let batch_len = self.frame_ids_buf.len();
            if self.cfg.drop_last && batch_len < batch_size {
                if self.cursor >= self.indices.len() {
                    return Ok(None);
                } else {
                    continue;
                }
            }
            let image_shape = [batch_len, 3, height as usize, width as usize];
            let boxes_shape = [batch_len, self.cfg.max_boxes, 4];
            let mask_shape = [batch_len, self.cfg.max_boxes];

            let images =
                burn::tensor::Tensor::<B, 1>::from_floats(self.images_buf.as_slice(), device)
                    .reshape(image_shape);
            let boxes =
                burn::tensor::Tensor::<B, 1>::from_floats(self.boxes_buf.as_slice(), device)
                    .reshape(boxes_shape);
            let box_mask =
                burn::tensor::Tensor::<B, 1>::from_floats(self.mask_buf.as_slice(), device)
                    .reshape(mask_shape);
            let frame_ids =
                burn::tensor::Tensor::<B, 1>::from_floats(self.frame_ids_buf.as_slice(), device)
                    .reshape([batch_len]);
            let assemble_elapsed = t_assemble.elapsed();

            self.processed_samples += batch_len;
            self.processed_batches += 1;
            self.skipped_total += skipped_empty + skipped_missing;
            self.skipped_empty += skipped_empty;
            self.skipped_missing += skipped_missing;
            self.total_load_time += load_elapsed;
            self.total_assemble_time += assemble_elapsed;
            self.maybe_trace(
                batch_len,
                width as usize,
                height as usize,
                load_elapsed,
                assemble_elapsed,
            );
            self.maybe_log_progress();

            return Ok(Some(BurnBatch {
                images,
                boxes,
                box_mask,
                frame_ids,
            }));
        }
    }

    fn maybe_log_progress(&mut self) {
        let Some(threshold) = self.log_every_samples else {
            return;
        };
        let processed_since = self
            .processed_samples
            .saturating_sub(self.last_logged_samples);
        let elapsed = self.started.elapsed();
        let since_last = self.last_log.elapsed();
        let should_log = processed_since >= threshold || since_last >= Duration::from_secs(30);
        if !should_log {
            return;
        }
        let secs = elapsed.as_secs_f32().max(0.001);
        let rate = self.processed_samples as f32 / secs;
        let avg_load_ms = if self.processed_batches > 0 {
            (self.total_load_time.as_secs_f64() * 1000.0) / self.processed_batches as f64
        } else {
            0.0
        };
        let avg_assemble_ms = if self.processed_batches > 0 {
            (self.total_assemble_time.as_secs_f64() * 1000.0) / self.processed_batches as f64
        } else {
            0.0
        };
        if !self.warn_once || !self.warned_counts {
            eprintln!(
                "[dataset] batches={} samples={} skipped_empty={} skipped_missing={} skipped_errors={} elapsed={:.1}s rate={:.1} img/s avg_load_ms={:.2} avg_assemble_ms={:.2}",
                self.processed_batches,
                self.processed_samples,
                self.skipped_empty,
                self.skipped_missing,
                self.skipped_errors,
                secs,
                rate,
                avg_load_ms,
                avg_assemble_ms
            );
        }
        self.last_logged_samples = self.processed_samples;
        self.last_log = Instant::now();
        if self.warn_once {
            self.warned_counts = true;
        }
    }

    fn maybe_trace(
        &mut self,
        batch_len: usize,
        width: usize,
        height: usize,
        load_elapsed: Duration,
        assemble_elapsed: Duration,
    ) {
        let Some(path) = &self.trace_path else {
            return;
        };
        if self.trace_file.is_none() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(f) => self.trace_file = Some(f),
                Err(e) => {
                    eprintln!("Failed to open trace file {}: {e}", path.display());
                    self.trace_path = None;
                    return;
                }
            }
        }
        let Some(file) = self.trace_file.as_mut() else {
            return;
        };
        let record = serde_json::json!({
            "batch": self.processed_batches,
            "samples": batch_len,
            "width": width,
            "height": height,
            "max_boxes": self.cfg.max_boxes,
            "skipped_empty_total": self.skipped_empty,
            "skipped_missing_total": self.skipped_missing,
            "skipped_errors_total": self.skipped_errors,
            "load_ms": load_elapsed.as_secs_f64() * 1000.0,
            "assemble_ms": assemble_elapsed.as_secs_f64() * 1000.0,
            "timestamp_ms": self.started.elapsed().as_millis() as u64
        });
        if let Err(e) = writeln!(file, "{}", record) {
            eprintln!("Failed to write trace record: {e}");
            self.trace_path = None;
            self.trace_file = None;
        }
    }
}

#[cfg(feature = "burn_runtime")]
struct ShardBuffer {
    samples: usize,
    width: u32,
    height: u32,
    max_boxes: usize,
    backing: ShardBacking,
}

#[cfg(feature = "burn_runtime")]
enum ShardBacking {
    Owned {
        images: Vec<f32>,
        boxes: Vec<f32>,
        masks: Vec<f32>,
    },
    Mmap {
        mmap: std::sync::Arc<memmap2::Mmap>,
        image_offset: usize,
        boxes_offset: usize,
        mask_offset: usize,
    },
    #[allow(dead_code)]
    Streamed {
        path: PathBuf,
        image_offset: usize,
        boxes_offset: usize,
        mask_offset: usize,
        samples: usize,
    }, // placeholder for future extensions
}

#[cfg(feature = "burn_runtime")]
impl ShardBuffer {
    fn copy_sample(
        &self,
        sample_idx: usize,
        out_images: &mut Vec<f32>,
        out_boxes: &mut Vec<f32>,
        out_masks: &mut Vec<f32>,
    ) -> DatasetResult<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        let img_elems = 3 * w * h;
        let box_elems = self.max_boxes * 4;
        let mask_elems = self.max_boxes;
        match &self.backing {
            ShardBacking::Owned {
                images,
                boxes,
                masks,
            } => {
                let img_offset = sample_idx
                    .checked_mul(img_elems)
                    .ok_or_else(|| BurnDatasetError::Other("image offset overflow".into()))?;
                let box_offset = sample_idx
                    .checked_mul(box_elems)
                    .ok_or_else(|| BurnDatasetError::Other("box offset overflow".into()))?;
                let mask_offset = sample_idx
                    .checked_mul(mask_elems)
                    .ok_or_else(|| BurnDatasetError::Other("mask offset overflow".into()))?;
                out_images.extend_from_slice(&images[img_offset..img_offset + img_elems]);
                out_boxes.extend_from_slice(&boxes[box_offset..box_offset + box_elems]);
                out_masks.extend_from_slice(&masks[mask_offset..mask_offset + mask_elems]);
                Ok(())
            }
            ShardBacking::Mmap {
                mmap,
                image_offset,
                boxes_offset,
                mask_offset,
            } => {
                let img_bytes = img_elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| BurnDatasetError::Other("image byte size overflow".into()))?;
                let box_bytes = box_elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| BurnDatasetError::Other("box byte size overflow".into()))?;
                let mask_bytes = mask_elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| BurnDatasetError::Other("mask byte size overflow".into()))?;

                let img_start = image_offset
                    .checked_add(sample_idx * img_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("image offset overflow".into()))?;
                let box_start = boxes_offset
                    .checked_add(sample_idx * box_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("box offset overflow".into()))?;
                let mask_start = mask_offset
                    .checked_add(sample_idx * mask_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("mask offset overflow".into()))?;

                if img_start + img_bytes > mmap.len()
                    || box_start + box_bytes > mmap.len()
                    || mask_start + mask_bytes > mmap.len()
                {
                    return Err(BurnDatasetError::Other(
                        "shard mmap truncated for requested sample".into(),
                    ));
                }

                for chunk in mmap[img_start..img_start + img_bytes].chunks_exact(4) {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    out_images.push(f32::from_le_bytes(arr));
                }
                for chunk in mmap[box_start..box_start + box_bytes].chunks_exact(4) {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    out_boxes.push(f32::from_le_bytes(arr));
                }
                for chunk in mmap[mask_start..mask_start + mask_bytes].chunks_exact(4) {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    out_masks.push(f32::from_le_bytes(arr));
                }
                Ok(())
            }
            ShardBacking::Streamed {
                path,
                image_offset,
                boxes_offset,
                mask_offset,
                samples,
            } => {
                if sample_idx >= *samples {
                    return Err(BurnDatasetError::Other(format!(
                        "sample {} out of range for {}",
                        sample_idx,
                        path.display()
                    )));
                }
                let img_elems = 3 * w * h;
                let box_elems = self.max_boxes * 4;
                let mask_elems = self.max_boxes;
                let img_bytes = img_elems * std::mem::size_of::<f32>();
                let box_bytes = box_elems * std::mem::size_of::<f32>();
                let mask_bytes = mask_elems * std::mem::size_of::<f32>();

                let img_start = image_offset
                    .checked_add(sample_idx * img_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("image offset overflow".into()))?;
                let boxes_start = boxes_offset
                    .checked_add(sample_idx * box_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("box offset overflow".into()))?;
                let mask_start = mask_offset
                    .checked_add(sample_idx * mask_bytes)
                    .ok_or_else(|| BurnDatasetError::Other("mask offset overflow".into()))?;

                let mut file =
                    BufReader::new(File::open(path).map_err(|e| BurnDatasetError::Io {
                        path: path.clone(),
                        source: e,
                    })?);

                fn read_f32s<R: Read + Seek>(
                    file: &mut R,
                    offset: usize,
                    bytes: usize,
                    out: &mut Vec<f32>,
                    path: &Path,
                ) -> DatasetResult<()> {
                    file.seek(SeekFrom::Start(offset as u64)).map_err(|e| {
                        BurnDatasetError::Io {
                            path: path.to_path_buf(),
                            source: e,
                        }
                    })?;
                    let mut buf = vec![0u8; bytes];
                    file.read_exact(&mut buf)
                        .map_err(|e| BurnDatasetError::Io {
                            path: path.to_path_buf(),
                            source: e,
                        })?;
                    for chunk in buf.chunks_exact(4) {
                        let mut arr = [0u8; 4];
                        arr.copy_from_slice(chunk);
                        out.push(f32::from_le_bytes(arr));
                    }
                    Ok(())
                }

                read_f32s(&mut file, img_start, img_bytes, out_images, path)?;
                read_f32s(&mut file, boxes_start, box_bytes, out_boxes, path)?;
                read_f32s(&mut file, mask_start, mask_bytes, out_masks, path)?;
                Ok(())
            }
        }
    }
}

#[cfg(feature = "burn_runtime")]
pub struct WarehouseBatchIter {
    inner: WarehouseBatchIterKind,
    width: u32,
    height: u32,
    max_boxes: usize,
}

#[cfg(feature = "burn_runtime")]
enum WarehouseBatchIterKind {
    Direct {
        order: Vec<(usize, usize)>,
        shards: std::sync::Arc<Vec<ShardBuffer>>,
        cursor: usize,
        drop_last: bool,
    },
    Stream {
        rx: Receiver<Option<StreamedSample>>,
        remaining: usize,
        drop_last: bool,
        ended: bool,
    },
}

#[cfg(feature = "burn_runtime")]
struct StreamedSample {
    images: Vec<f32>,
    boxes: Vec<f32>,
    masks: Vec<f32>,
}

#[cfg(feature = "burn_runtime")]
struct StreamingStore {
    shards: std::sync::Arc<Vec<ShardBuffer>>,
    train_order: Vec<(usize, usize)>,
    val_order: Vec<(usize, usize)>,
    drop_last: bool,
    width: u32,
    height: u32,
    max_boxes: usize,
    prefetch: usize,
}

#[cfg(feature = "burn_runtime")]
impl StreamingStore {
    pub fn from_manifest_path(
        manifest_path: &Path,
        val_ratio: f32,
        seed: Option<u64>,
        drop_last: bool,
        prefetch: usize,
    ) -> DatasetResult<Self> {
        let manifest = WarehouseManifest::load(manifest_path)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let shards_vec = manifest
            .shards
            .iter()
            .enumerate()
            .map(|(i, meta)| {
                let t0 = Instant::now();
                let shard = load_shard_streamed(root, meta)?;
                let ms = t0.elapsed().as_millis();
                println!(
                    "[warehouse] stream shard {} (id={}, samples={}, size={}x{}, max_boxes={}) in {} ms",
                    i,
                    meta.id,
                    shard.samples,
                    shard.width,
                    shard.height,
                    shard.max_boxes,
                    ms
                );
                Ok(shard)
            })
            .collect::<DatasetResult<Vec<_>>>()?;
        let shards = std::sync::Arc::new(shards_vec);
        let total_samples: usize = shards.iter().map(|s| s.samples).sum();
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(total_samples);
        for (si, shard) in shards.iter().enumerate() {
            for i in 0..shard.samples {
                order.push((si, i));
            }
        }
        if let Some(s) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            order.shuffle(&mut rng);
        }
        let val_count =
            ((val_ratio.clamp(0.0, 1.0) * order.len() as f32).round() as usize).min(order.len());
        let (val_order, train_order) = order.split_at(val_count);
        let width = shards.get(0).map(|s| s.width).unwrap_or(0);
        let height = shards.get(0).map(|s| s.height).unwrap_or(0);
        let max_boxes = shards.get(0).map(|s| s.max_boxes).unwrap_or(0);
        Ok(StreamingStore {
            shards,
            train_order: train_order.to_vec(),
            val_order: val_order.to_vec(),
            drop_last,
            width,
            height,
            max_boxes,
            prefetch: prefetch.max(1),
        })
    }

    fn spawn_iter(&self, order: &[(usize, usize)], drop_last: bool) -> WarehouseBatchIter {
        let (tx, rx) = bounded(self.prefetch);
        let shards = self.shards.clone();
        let order_vec: Vec<(usize, usize)> = order.to_vec();
        let width = self.width;
        let height = self.height;
        let max_boxes = self.max_boxes;
        thread::spawn(move || {
            for (shard_idx, sample_idx) in order_vec.into_iter() {
                let shard = match shards.get(shard_idx) {
                    Some(s) => s,
                    None => break,
                };
                let mut images = Vec::new();
                let mut boxes = Vec::new();
                let mut masks = Vec::new();
                if let Err(e) = shard.copy_sample(sample_idx, &mut images, &mut boxes, &mut masks) {
                    eprintln!("[warehouse] streaming copy error: {:?}", e);
                    break;
                }
                if tx
                    .send(Some(StreamedSample {
                        images,
                        boxes,
                        masks,
                    }))
                    .is_err()
                {
                    break;
                }
            }
            let _ = tx.send(None);
        });

        WarehouseBatchIter {
            inner: WarehouseBatchIterKind::Stream {
                rx,
                remaining: order.len(),
                drop_last,
                ended: false,
            },
            width,
            height,
            max_boxes,
        }
    }
}

#[cfg(feature = "burn_runtime")]
impl WarehouseShardStore for StreamingStore {
    fn train_iter(&self) -> WarehouseBatchIter {
        self.spawn_iter(&self.train_order, self.drop_last)
    }

    fn val_iter(&self) -> WarehouseBatchIter {
        self.spawn_iter(&self.val_order, false)
    }

    fn train_len(&self) -> usize {
        self.train_order.len()
    }

    fn val_len(&self) -> usize {
        self.val_order.len()
    }

    fn total_shards(&self) -> usize {
        self.shards.len()
    }

    fn mode(&self) -> WarehouseStoreMode {
        WarehouseStoreMode::Streaming
    }
}

#[cfg(feature = "burn_runtime")]
pub trait WarehouseShardStore: Send + Sync {
    fn train_iter(&self) -> WarehouseBatchIter;
    fn val_iter(&self) -> WarehouseBatchIter;
    fn train_len(&self) -> usize;
    fn val_len(&self) -> usize;
    fn total_shards(&self) -> usize;
    #[allow(dead_code)]
    fn mode(&self) -> WarehouseStoreMode {
        WarehouseStoreMode::InMemory
    }
}

#[cfg(feature = "burn_runtime")]
pub struct WarehouseLoaders {
    store: Box<dyn WarehouseShardStore>,
}

#[cfg(feature = "burn_runtime")]
struct InMemoryStore {
    shards: std::sync::Arc<Vec<ShardBuffer>>,
    train_order: Vec<(usize, usize)>,
    val_order: Vec<(usize, usize)>,
    drop_last: bool,
    width: u32,
    height: u32,
    max_boxes: usize,
}

#[cfg(feature = "burn_runtime")]
impl InMemoryStore {
    pub fn from_manifest_path(
        manifest_path: &Path,
        val_ratio: f32,
        seed: Option<u64>,
        drop_last: bool,
    ) -> DatasetResult<Self> {
        let manifest = WarehouseManifest::load(manifest_path)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let shards_vec = manifest
            .shards
            .iter()
            .enumerate()
            .map(|(i, meta)| {
                let t0 = Instant::now();
                let shard = load_shard_owned(root, meta)?;
                let ms = t0.elapsed().as_millis();
                println!(
                    "[warehouse] loaded shard {} (id={}, samples={}, size={}x{}, max_boxes={}) in {} ms",
                    i,
                    meta.id,
                    shard.samples,
                    shard.width,
                    shard.height,
                    shard.max_boxes,
                    ms
                );
                Ok(shard)
            })
            .collect::<DatasetResult<Vec<_>>>()?;
        let shards = std::sync::Arc::new(shards_vec);
        let total_samples: usize = shards.iter().map(|s| s.samples).sum();
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(total_samples);
        for (si, shard) in shards.iter().enumerate() {
            for i in 0..shard.samples {
                order.push((si, i));
            }
        }
        if let Some(s) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            order.shuffle(&mut rng);
        }
        let val_count =
            ((val_ratio.clamp(0.0, 1.0) * order.len() as f32).round() as usize).min(order.len());
        let (val_order, train_order) = order.split_at(val_count);
        let width = shards.get(0).map(|s| s.width).unwrap_or(0);
        let height = shards.get(0).map(|s| s.height).unwrap_or(0);
        let max_boxes = shards.get(0).map(|s| s.max_boxes).unwrap_or(0);
        Ok(InMemoryStore {
            shards,
            train_order: train_order.to_vec(),
            val_order: val_order.to_vec(),
            drop_last,
            width,
            height,
            max_boxes,
        })
    }
}

#[cfg(feature = "burn_runtime")]
impl WarehouseShardStore for InMemoryStore {
    fn train_iter(&self) -> WarehouseBatchIter {
        WarehouseBatchIter {
            inner: WarehouseBatchIterKind::Direct {
                order: self.train_order.clone(),
                shards: self.shards.clone(),
                cursor: 0,
                drop_last: self.drop_last,
            },
            width: self.width,
            height: self.height,
            max_boxes: self.max_boxes,
        }
    }

    fn val_iter(&self) -> WarehouseBatchIter {
        WarehouseBatchIter {
            inner: WarehouseBatchIterKind::Direct {
                order: self.val_order.clone(),
                shards: self.shards.clone(),
                cursor: 0,
                drop_last: false,
            },
            width: self.width,
            height: self.height,
            max_boxes: self.max_boxes,
        }
    }

    fn train_len(&self) -> usize {
        self.train_order.len()
    }

    fn val_len(&self) -> usize {
        self.val_order.len()
    }

    fn total_shards(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(feature = "burn_runtime")]
struct MmapStore {
    shards: std::sync::Arc<Vec<ShardBuffer>>,
    train_order: Vec<(usize, usize)>,
    val_order: Vec<(usize, usize)>,
    drop_last: bool,
    width: u32,
    height: u32,
    max_boxes: usize,
}

#[cfg(feature = "burn_runtime")]
impl MmapStore {
    pub fn from_manifest_path(
        manifest_path: &Path,
        val_ratio: f32,
        seed: Option<u64>,
        drop_last: bool,
    ) -> DatasetResult<Self> {
        let manifest = WarehouseManifest::load(manifest_path)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let shards_vec = manifest
            .shards
            .iter()
            .enumerate()
            .map(|(i, meta)| {
                let t0 = Instant::now();
                let shard = load_shard_mmap(root, meta)?;
                let ms = t0.elapsed().as_millis();
                println!(
                    "[warehouse] mmap shard {} (id={}, samples={}, size={}x{}, max_boxes={}) in {} ms",
                    i,
                    meta.id,
                    shard.samples,
                    shard.width,
                    shard.height,
                    shard.max_boxes,
                    ms
                );
                Ok(shard)
            })
            .collect::<DatasetResult<Vec<_>>>()?;
        let shards = std::sync::Arc::new(shards_vec);
        let total_samples: usize = shards.iter().map(|s| s.samples).sum();
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(total_samples);
        for (si, shard) in shards.iter().enumerate() {
            for i in 0..shard.samples {
                order.push((si, i));
            }
        }
        if let Some(s) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            order.shuffle(&mut rng);
        }
        let val_count =
            ((val_ratio.clamp(0.0, 1.0) * order.len() as f32).round() as usize).min(order.len());
        let (val_order, train_order) = order.split_at(val_count);
        let width = shards.get(0).map(|s| s.width).unwrap_or(0);
        let height = shards.get(0).map(|s| s.height).unwrap_or(0);
        let max_boxes = shards.get(0).map(|s| s.max_boxes).unwrap_or(0);
        Ok(MmapStore {
            shards,
            train_order: train_order.to_vec(),
            val_order: val_order.to_vec(),
            drop_last,
            width,
            height,
            max_boxes,
        })
    }
}

#[cfg(feature = "burn_runtime")]
impl WarehouseShardStore for MmapStore {
    fn train_iter(&self) -> WarehouseBatchIter {
        WarehouseBatchIter {
            inner: WarehouseBatchIterKind::Direct {
                order: self.train_order.clone(),
                shards: self.shards.clone(),
                cursor: 0,
                drop_last: self.drop_last,
            },
            width: self.width,
            height: self.height,
            max_boxes: self.max_boxes,
        }
    }

    fn val_iter(&self) -> WarehouseBatchIter {
        WarehouseBatchIter {
            inner: WarehouseBatchIterKind::Direct {
                order: self.val_order.clone(),
                shards: self.shards.clone(),
                cursor: 0,
                drop_last: false,
            },
            width: self.width,
            height: self.height,
            max_boxes: self.max_boxes,
        }
    }

    fn train_len(&self) -> usize {
        self.train_order.len()
    }

    fn val_len(&self) -> usize {
        self.val_order.len()
    }

    fn total_shards(&self) -> usize {
        self.shards.len()
    }

    fn mode(&self) -> WarehouseStoreMode {
        WarehouseStoreMode::Mmap
    }
}
#[cfg(feature = "burn_runtime")]
impl WarehouseBatchIter {
    pub fn len(&self) -> usize {
        match &self.inner {
            WarehouseBatchIterKind::Direct { order, .. } => order.len(),
            WarehouseBatchIterKind::Stream { remaining, .. } => *remaining,
        }
    }

    pub fn next_batch<B: burn::tensor::backend::Backend>(
        &mut self,
        batch_size: usize,
        device: &B::Device,
    ) -> DatasetResult<Option<BurnBatch<B>>> {
        match &mut self.inner {
            WarehouseBatchIterKind::Direct {
                order,
                shards,
                cursor,
                drop_last,
            } => {
                if *cursor >= order.len() {
                    return Ok(None);
                }
                let end = (*cursor + batch_size).min(order.len());
                let slice = &order[*cursor..end];
                *cursor = end;
                if *drop_last && slice.len() < batch_size {
                    return Ok(None);
                }
                let mut images = Vec::new();
                let mut boxes = Vec::new();
                let mut masks = Vec::new();
                let mut frame_ids = Vec::new();
                for (global_idx, (shard_idx, sample_idx)) in slice.iter().enumerate() {
                    let shard = &shards[*shard_idx];
                    shard.copy_sample(*sample_idx, &mut images, &mut boxes, &mut masks)?;
                    frame_ids.push(global_idx as f32);
                }
                let image_shape = [slice.len(), 3, self.height as usize, self.width as usize];
                let boxes_shape = [slice.len(), self.max_boxes, 4];
                let mask_shape = [slice.len(), self.max_boxes];

                let images = burn::tensor::Tensor::<B, 1>::from_floats(images.as_slice(), device)
                    .reshape(image_shape);
                let boxes = burn::tensor::Tensor::<B, 1>::from_floats(boxes.as_slice(), device)
                    .reshape(boxes_shape);
                let box_mask = burn::tensor::Tensor::<B, 1>::from_floats(masks.as_slice(), device)
                    .reshape(mask_shape);
                let frame_ids =
                    burn::tensor::Tensor::<B, 1>::from_floats(frame_ids.as_slice(), device)
                        .reshape([slice.len()]);

                Ok(Some(BurnBatch {
                    images,
                    boxes,
                    box_mask,
                    frame_ids,
                }))
            }
            WarehouseBatchIterKind::Stream {
                rx,
                remaining,
                drop_last,
                ended,
            } => {
                if *ended || *remaining == 0 {
                    return Ok(None);
                }
                let mut images = Vec::new();
                let mut boxes = Vec::new();
                let mut masks = Vec::new();
                let mut frame_ids = Vec::new();
                let mut pulled = 0usize;
                while pulled < batch_size {
                    match rx.recv() {
                        Ok(Some(sample)) => {
                            images.extend_from_slice(&sample.images);
                            boxes.extend_from_slice(&sample.boxes);
                            masks.extend_from_slice(&sample.masks);
                            frame_ids.push(pulled as f32);
                            pulled += 1;
                        }
                        Ok(None) => {
                            *ended = true;
                            break;
                        }
                        Err(_) => {
                            *ended = true;
                            break;
                        }
                    }
                }
                if pulled == 0 || (*drop_last && pulled < batch_size) {
                    return Ok(None);
                }
                *remaining = remaining.saturating_sub(pulled);
                let image_shape = [pulled, 3, self.height as usize, self.width as usize];
                let boxes_shape = [pulled, self.max_boxes, 4];
                let mask_shape = [pulled, self.max_boxes];
                let images = burn::tensor::Tensor::<B, 1>::from_floats(images.as_slice(), device)
                    .reshape(image_shape);
                let boxes = burn::tensor::Tensor::<B, 1>::from_floats(boxes.as_slice(), device)
                    .reshape(boxes_shape);
                let box_mask = burn::tensor::Tensor::<B, 1>::from_floats(masks.as_slice(), device)
                    .reshape(mask_shape);
                let frame_ids =
                    burn::tensor::Tensor::<B, 1>::from_floats(frame_ids.as_slice(), device)
                        .reshape([pulled]);
                Ok(Some(BurnBatch {
                    images,
                    boxes,
                    box_mask,
                    frame_ids,
                }))
            }
        }
    }
}

#[cfg(feature = "burn_runtime")]
impl WarehouseLoaders {
    pub fn store_len(&self) -> usize {
        self.store.total_shards()
    }
    pub fn from_manifest_path(
        manifest_path: &Path,
        val_ratio: f32,
        seed: Option<u64>,
        drop_last: bool,
    ) -> DatasetResult<Self> {
        let mode = WarehouseStoreMode::from_env();
        println!("[warehouse] store mode: {:?}", mode);
        match mode {
            WarehouseStoreMode::InMemory => {
                let store =
                    InMemoryStore::from_manifest_path(manifest_path, val_ratio, seed, drop_last)?;
                Ok(WarehouseLoaders {
                    store: Box::new(store),
                })
            }
            WarehouseStoreMode::Mmap => {
                let store =
                    MmapStore::from_manifest_path(manifest_path, val_ratio, seed, drop_last)?;
                Ok(WarehouseLoaders {
                    store: Box::new(store),
                })
            }
            WarehouseStoreMode::Streaming => {
                let prefetch = WarehouseStoreMode::prefetch_from_env();
                println!("[warehouse] streaming prefetch depth: {}", prefetch);
                let store = StreamingStore::from_manifest_path(
                    manifest_path,
                    val_ratio,
                    seed,
                    drop_last,
                    prefetch,
                )?;
                Ok(WarehouseLoaders {
                    store: Box::new(store),
                })
            }
        }
    }

    pub fn train_iter(&self) -> WarehouseBatchIter {
        self.store.train_iter()
    }

    pub fn val_iter(&self) -> WarehouseBatchIter {
        self.store.val_iter()
    }

    pub fn train_len(&self) -> usize {
        self.store.train_len()
    }

    pub fn val_len(&self) -> usize {
        self.store.val_len()
    }
}

#[cfg(feature = "burn_runtime")]
fn read_u32_le(data: &[u8]) -> u32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(data);
    u32::from_le_bytes(arr)
}

#[cfg(feature = "burn_runtime")]
fn read_u64_le(data: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(data);
    u64::from_le_bytes(arr)
}

#[cfg(feature = "burn_runtime")]
fn load_shard_owned(root: &Path, meta: &ShardMetadata) -> DatasetResult<ShardBuffer> {
    let path = root.join(&meta.relative_path);
    let data = fs::read(&path).map_err(|e| BurnDatasetError::Io {
        path: path.clone(),
        source: e,
    })?;
    if data.len() < 4 {
        return Err(BurnDatasetError::Other(format!(
            "shard {} too small",
            path.display()
        )));
    }
    if &data[0..4] != b"TWH1" {
        return Err(BurnDatasetError::Other(format!(
            "bad magic in shard {}",
            path.display()
        )));
    }
    let shard_version = read_u32_le(&data[4..8]);
    if shard_version != meta.shard_version {
        return Err(BurnDatasetError::Other(format!(
            "shard version mismatch {} vs {}",
            shard_version, meta.shard_version
        )));
    }
    let dtype = read_u32_le(&data[8..12]);
    if dtype != 0 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported dtype {} in {}",
            dtype,
            path.display()
        )));
    }
    let width = read_u32_le(&data[16..20]);
    let height = read_u32_le(&data[20..24]);
    let channels = read_u32_le(&data[24..28]);
    if channels != 3 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported channels {} in {}",
            channels,
            path.display()
        )));
    }
    let max_boxes = read_u32_le(&data[28..32]) as usize;
    let samples = read_u64_le(&data[32..40]) as usize;
    let image_offset = read_u64_le(&data[40..48]) as usize;
    let boxes_offset = read_u64_le(&data[48..56]) as usize;
    let mask_offset = read_u64_le(&data[56..64]) as usize;

    let image_elems = samples
        .checked_mul(3)
        .and_then(|v| v.checked_mul(width as usize))
        .and_then(|v| v.checked_mul(height as usize))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing image elems".into()))?;
    let box_elems = samples
        .checked_mul(max_boxes)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing box elems".into()))?;
    let mask_elems = samples
        .checked_mul(max_boxes)
        .ok_or_else(|| BurnDatasetError::Other("overflow computing mask elems".into()))?;

    let image_bytes = image_elems * std::mem::size_of::<f32>();
    let box_bytes = box_elems * std::mem::size_of::<f32>();
    let mask_bytes = mask_elems * std::mem::size_of::<f32>();

    if image_offset + image_bytes > data.len()
        || boxes_offset + box_bytes > data.len()
        || mask_offset + mask_bytes > data.len()
    {
        return Err(BurnDatasetError::Other(format!(
            "shard {} truncated",
            path.display()
        )));
    }

    let images = data[image_offset..image_offset + image_bytes]
        .chunks_exact(4)
        .map(|c| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(c);
            f32::from_le_bytes(arr)
        })
        .collect();
    let boxes = data[boxes_offset..boxes_offset + box_bytes]
        .chunks_exact(4)
        .map(|c| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(c);
            f32::from_le_bytes(arr)
        })
        .collect();
    let masks = data[mask_offset..mask_offset + mask_bytes]
        .chunks_exact(4)
        .map(|c| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(c);
            f32::from_le_bytes(arr)
        })
        .collect();

    Ok(ShardBuffer {
        samples,
        width,
        height,
        max_boxes,
        backing: ShardBacking::Owned {
            images,
            boxes,
            masks,
        },
    })
}

#[cfg(feature = "burn_runtime")]
fn load_shard_mmap(root: &Path, meta: &ShardMetadata) -> DatasetResult<ShardBuffer> {
    let path = root.join(&meta.relative_path);
    let file = File::open(&path).map_err(|e| BurnDatasetError::Io {
        path: path.clone(),
        source: e,
    })?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| BurnDatasetError::Io {
                path: path.clone(),
                source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            })?
    };
    let data = &mmap[..];
    if data.len() < 4 {
        return Err(BurnDatasetError::Other(format!(
            "shard {} too small",
            path.display()
        )));
    }
    if &data[0..4] != b"TWH1" {
        return Err(BurnDatasetError::Other(format!(
            "bad magic in shard {}",
            path.display()
        )));
    }
    let shard_version = read_u32_le(&data[4..8]);
    if shard_version != meta.shard_version {
        return Err(BurnDatasetError::Other(format!(
            "shard version mismatch {} vs {}",
            shard_version, meta.shard_version
        )));
    }
    let dtype = read_u32_le(&data[8..12]);
    if dtype != 0 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported dtype {} in {}",
            dtype,
            path.display()
        )));
    }
    let width = read_u32_le(&data[16..20]);
    let height = read_u32_le(&data[20..24]);
    let channels = read_u32_le(&data[24..28]);
    if channels != 3 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported channels {} in {}",
            channels,
            path.display()
        )));
    }
    let max_boxes = read_u32_le(&data[28..32]) as usize;
    let samples = read_u64_le(&data[32..40]) as usize;
    let image_offset = read_u64_le(&data[40..48]) as usize;
    let boxes_offset = read_u64_le(&data[48..56]) as usize;
    let mask_offset = read_u64_le(&data[56..64]) as usize;

    let image_elems = samples
        .checked_mul(3)
        .and_then(|v| v.checked_mul(width as usize))
        .and_then(|v| v.checked_mul(height as usize))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing image elems".into()))?;
    let box_elems = samples
        .checked_mul(max_boxes)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing box elems".into()))?;
    let mask_elems = samples
        .checked_mul(max_boxes)
        .ok_or_else(|| BurnDatasetError::Other("overflow computing mask elems".into()))?;

    let image_bytes = image_elems * std::mem::size_of::<f32>();
    let box_bytes = box_elems * std::mem::size_of::<f32>();
    let mask_bytes = mask_elems * std::mem::size_of::<f32>();

    if image_offset + image_bytes > data.len()
        || boxes_offset + box_bytes > data.len()
        || mask_offset + mask_bytes > data.len()
    {
        return Err(BurnDatasetError::Other(format!(
            "shard {} truncated",
            path.display()
        )));
    }

    Ok(ShardBuffer {
        samples,
        width,
        height,
        max_boxes,
        backing: ShardBacking::Mmap {
            mmap: std::sync::Arc::new(mmap),
            image_offset,
            boxes_offset,
            mask_offset,
        },
    })
}

#[cfg(feature = "burn_runtime")]
fn load_shard_streamed(root: &Path, meta: &ShardMetadata) -> DatasetResult<ShardBuffer> {
    let path = root.join(&meta.relative_path);
    let mut file = File::open(&path).map_err(|e| BurnDatasetError::Io {
        path: path.clone(),
        source: e,
    })?;
    let mut header = vec![0u8; 64];
    let read = file.read(&mut header).map_err(|e| BurnDatasetError::Io {
        path: path.clone(),
        source: e,
    })?;
    if read < 64 {
        return Err(BurnDatasetError::Other(format!(
            "shard {} too small",
            path.display()
        )));
    }
    if &header[0..4] != b"TWH1" {
        return Err(BurnDatasetError::Other(format!(
            "bad magic in shard {}",
            path.display()
        )));
    }
    let shard_version = read_u32_le(&header[4..8]);
    if shard_version != meta.shard_version {
        return Err(BurnDatasetError::Other(format!(
            "shard version mismatch {} vs {}",
            shard_version, meta.shard_version
        )));
    }
    let dtype = read_u32_le(&header[8..12]);
    if dtype != 0 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported dtype {} in {}",
            dtype,
            path.display()
        )));
    }
    let width = read_u32_le(&header[16..20]);
    let height = read_u32_le(&header[20..24]);
    let channels = read_u32_le(&header[24..28]);
    if channels != 3 {
        return Err(BurnDatasetError::Other(format!(
            "unsupported channels {} in {}",
            channels,
            path.display()
        )));
    }
    let max_boxes = read_u32_le(&header[28..32]) as usize;
    let samples = read_u64_le(&header[32..40]) as usize;
    let image_offset = read_u64_le(&header[40..48]) as usize;
    let boxes_offset = read_u64_le(&header[48..56]) as usize;
    let mask_offset = read_u64_le(&header[56..64]) as usize;

    let img_elems = samples
        .checked_mul(3)
        .and_then(|v| v.checked_mul(width as usize))
        .and_then(|v| v.checked_mul(height as usize))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing image elems".into()))?;
    let box_elems = samples
        .checked_mul(max_boxes)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| BurnDatasetError::Other("overflow computing box elems".into()))?;
    let mask_elems = samples
        .checked_mul(max_boxes)
        .ok_or_else(|| BurnDatasetError::Other("overflow computing mask elems".into()))?;

    let image_bytes = img_elems * std::mem::size_of::<f32>();
    let box_bytes = box_elems * std::mem::size_of::<f32>();
    let mask_bytes = mask_elems * std::mem::size_of::<f32>();

    let file_len = file
        .metadata()
        .map_err(|e| BurnDatasetError::Io {
            path: path.clone(),
            source: e,
        })?
        .len() as usize;

    if image_offset
        .checked_add(image_bytes)
        .map(|v| v > file_len)
        .unwrap_or(true)
        || boxes_offset
            .checked_add(box_bytes)
            .map(|v| v > file_len)
            .unwrap_or(true)
        || mask_offset
            .checked_add(mask_bytes)
            .map(|v| v > file_len)
            .unwrap_or(true)
    {
        return Err(BurnDatasetError::Other(format!(
            "shard {} truncated",
            path.display()
        )));
    }

    Ok(ShardBuffer {
        samples,
        width,
        height,
        max_boxes,
        backing: ShardBacking::Streamed {
            path,
            image_offset,
            boxes_offset,
            mask_offset,
            samples,
        },
    })
}
