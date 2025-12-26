use image::imageops::FilterType;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use serde::Deserialize;
use std::cmp::max;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "burn_runtime")]
use rayon::prelude::*;
#[cfg(feature = "burn_runtime")]
use std::io::Write;
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
    let raw = fs::read(&idx.label_path)
        .map_err(|e| BurnDatasetError::Io { path: idx.label_path.clone(), source: e })?;
    let meta: LabelEntry = serde_json::from_slice(&raw)
        .map_err(|e| BurnDatasetError::Json { path: idx.label_path.clone(), source: e })?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
pub struct TransformPipeline {
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
            self.seed.map(|s| s.to_string()).unwrap_or_else(|| "none".to_string())
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
        maybe_jitter(&mut img, self.color_jitter_prob, self.color_jitter_strength, rng);
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
        let sample = build_sample_from_image(
            img,
            width,
            height,
            boxes,
            meta.frame_id,
            self.max_boxes,
        )?;
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
        self
    }
    pub fn resize_mode(mut self, mode: ResizeMode) -> Self {
        self.inner.resize_mode = mode;
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

#[derive(Debug, Clone, Default)]
pub struct RunSummary {
    pub run_dir: PathBuf,
    pub total: usize,
    pub non_empty: usize,
    pub empty: usize,
    pub missing_image: usize,
    pub missing_file: usize,
    pub invalid: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DatasetSummary {
    pub runs: Vec<RunSummary>,
    pub totals: RunSummary,
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
                })
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

    #[test]
    fn hflip_boxes_are_inverted() {
        let mut img = image::RgbImage::new(2, 2);
        let mut boxes = vec![[0.25, 0.0, 0.75, 1.0]];
        maybe_hflip(&mut img, &mut boxes, 1.0);
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

    pub fn from_indices(
        mut indices: Vec<SampleIndex>,
        cfg: DatasetConfig,
    ) -> DatasetResult<Self> {
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
                                eprintln!("Warning: skipping label {}: {e}", idx.label_path.display());
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

            let images = burn::tensor::Tensor::<B, 1>::from_floats(
                self.images_buf.as_slice(),
                device,
            )
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
            self.maybe_trace(batch_len, width as usize, height as usize, load_elapsed, assemble_elapsed);
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
        let processed_since = self.processed_samples.saturating_sub(self.last_logged_samples);
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
