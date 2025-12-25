use image::imageops::FilterType;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use serde::Deserialize;
use std::cmp::max;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

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

pub fn count_boxes(idx: &SampleIndex) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let raw = fs::read(&idx.label_path)?;
    let meta: LabelEntry = serde_json::from_slice(&raw)?;
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

/// Scan a captures root (e.g., `assets/datasets/captures`) and index all label files.
pub fn index_runs(root: &Path) -> Result<Vec<SampleIndex>, Box<dyn Error + Send + Sync>> {
    let mut indices = Vec::new();
    for entry in fs::read_dir(root)? {
        let Ok(run) = entry else { continue };
        let run_path = run.path();
        if !run_path.is_dir() {
            continue;
        }
        let labels_dir = run_path.join("labels");
        if !labels_dir.exists() {
            continue;
        }
        for label in fs::read_dir(labels_dir)? {
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
pub fn load_run_dataset(
    run_dir: &Path,
) -> Result<Vec<DatasetSample>, Box<dyn Error + Send + Sync>> {
    let labels_dir = run_dir.join("labels");
    if !labels_dir.exists() {
        return Err(format!("labels directory not found at {}", labels_dir.display()).into());
    }

    let mut label_paths: Vec<PathBuf> = fs::read_dir(&labels_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    label_paths.sort();

    let mut samples = Vec::new();
    for label_path in label_paths {
        let sample = load_sample(
            &SampleIndex {
                run_dir: run_dir.to_path_buf(),
                label_path,
            },
            &DatasetConfig {
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
            },
        )?;
        samples.push(sample);
    }

    Ok(samples)
}

fn load_sample(
    idx: &SampleIndex,
    cfg: &DatasetConfig,
) -> Result<DatasetSample, Box<dyn Error + Send + Sync>> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let raw = fs::read(&idx.label_path)?;
    let meta: LabelEntry = serde_json::from_slice(&raw)?;
    if !meta.image_present {
        return Err(format!("image not present for {}", idx.label_path.display()).into());
    }

    let img_path = idx.run_dir.join(&meta.image);
    if !img_path.exists() {
        return Err(format!("image file missing: {}", img_path.display()).into());
    }
    let img = image::open(&img_path)?.to_rgb8();
    let (mut width, mut height) = img.dimensions();

    if let Some((w, h)) = cfg.target_size {
        match cfg.resize_mode {
            ResizeMode::Force => {
                width = w;
                height = h;
                let mut resized = image::imageops::resize(&img, w, h, FilterType::Triangle);
                let mut boxes = normalize_boxes(&meta.polyp_labels, w, h);
                maybe_hflip(&mut resized, &mut boxes, cfg.flip_horizontal_prob);
                return build_sample_from_image(
                    resized,
                    width,
                    height,
                    boxes,
                    meta.frame_id,
                    cfg.max_boxes,
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

                maybe_hflip(&mut resized_img, &mut boxes, cfg.flip_horizontal_prob);
                maybe_jitter(
                    &mut resized_img,
                    cfg.color_jitter_prob,
                    cfg.color_jitter_strength,
                );
                maybe_scale_jitter(
                    &mut resized_img,
                    &mut boxes,
                    cfg.scale_jitter_prob,
                    cfg.scale_jitter_min,
                    cfg.scale_jitter_max,
                );
                maybe_noise(&mut resized_img, cfg.noise_prob, cfg.noise_strength);
                maybe_blur(&mut resized_img, cfg.blur_prob, cfg.blur_sigma);

                if boxes.len() > cfg.max_boxes {
                    boxes.truncate(cfg.max_boxes);
                }

                return build_sample_from_image(
                    resized_img,
                    w,
                    h,
                    boxes,
                    meta.frame_id,
                    cfg.max_boxes,
                );
            }
        }
    }

    let mut boxes = normalize_boxes(&meta.polyp_labels, width, height);
    let mut img = img;
    maybe_hflip(&mut img, &mut boxes, cfg.flip_horizontal_prob);
    maybe_jitter(&mut img, cfg.color_jitter_prob, cfg.color_jitter_strength);
    maybe_scale_jitter(
        &mut img,
        &mut boxes,
        cfg.scale_jitter_prob,
        cfg.scale_jitter_min,
        cfg.scale_jitter_max,
    );
    maybe_noise(&mut img, cfg.noise_prob, cfg.noise_strength);
    maybe_blur(&mut img, cfg.blur_prob, cfg.blur_sigma);
    let sample = build_sample_from_image(img, width, height, boxes, meta.frame_id, cfg.max_boxes)?;
    ONCE.call_once(|| {
        if sample.boxes.is_empty() {
            eprintln!(
                "Debug: first sample {} has 0 boxes (image {}x{})",
                idx.label_path.display(),
                width,
                height
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
) -> Result<DatasetSample, Box<dyn Error + Send + Sync>> {
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
) -> Result<(image::RgbImage, u32, u32), Box<dyn Error + Send + Sync>> {
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

pub(crate) fn maybe_hflip(img: &mut image::RgbImage, boxes: &mut Vec<[f32; 4]>, prob: f32) {
    if prob <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
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

pub(crate) fn maybe_jitter(img: &mut image::RgbImage, prob: f32, strength: f32) {
    if prob <= 0.0 || strength <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
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

pub(crate) fn maybe_noise(img: &mut image::RgbImage, prob: f32, strength: f32) {
    if prob <= 0.0 || strength <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
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

pub(crate) fn maybe_blur(img: &mut image::RgbImage, prob: f32, sigma: f32) {
    if prob <= 0.0 || sigma <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
    if rng.gen_range(0.0..1.0) >= prob {
        return;
    }
    let blurred = image::imageops::blur(img, sigma);
    *img = blurred;
}

pub(crate) fn maybe_scale_jitter(
    img: &mut image::RgbImage,
    boxes: &mut Vec<[f32; 4]>,
    prob: f32,
    min_scale: f32,
    max_scale: f32,
) {
    if prob <= 0.0 || min_scale <= 0.0 || max_scale <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
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
}

#[cfg(feature = "burn_runtime")]
impl BatchIter {
    pub fn from_root(
        root: &Path,
        cfg: DatasetConfig,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let indices = index_runs(root)?;
        Self::from_indices(indices, cfg)
    }

    pub fn from_indices(
        mut indices: Vec<SampleIndex>,
        cfg: DatasetConfig,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        use rand::seq::SliceRandom;
        let mut rng = match cfg.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        if cfg.shuffle {
            indices.shuffle(&mut rng);
        }
        Ok(Self {
            indices,
            cursor: 0,
            cfg,
        })
    }

    pub fn next_batch<B: burn::tensor::backend::Backend>(
        &mut self,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Option<BurnBatch<B>>, Box<dyn Error + Send + Sync>> {
        loop {
            if self.cursor >= self.indices.len() {
                return Ok(None);
            }
            let end = (self.cursor + batch_size).min(self.indices.len());
            let slice = &self.indices[self.cursor..end];
            self.cursor = end;

            let mut images = Vec::new();
            let mut boxes = Vec::new();
            let mut box_mask = Vec::new();
            let mut frame_ids = Vec::new();

            let mut expected_size: Option<(u32, u32)> = None;
            let mut skipped = 0usize;

            for idx in slice {
                let sample = load_sample(idx, &self.cfg)?;
                if self.cfg.skip_empty_labels && sample.boxes.is_empty() {
                    eprintln!(
                        "Warning: no boxes found in {} (skipping sample)",
                        idx.label_path.display()
                    );
                    skipped += 1;
                    continue;
                }

                let size = (sample.width, sample.height);
                match expected_size {
                    None => expected_size = Some(size),
                    Some(sz) if sz != size => {
                        return Err("batch contains varying image sizes; set a target_size to force consistency".into());
                    }
                    _ => {}
                }

                frame_ids.push(sample.frame_id as f32);
                images.extend_from_slice(&sample.image_chw);

                let mut padded = vec![0.0f32; self.cfg.max_boxes * 4];
                let mut mask = vec![0.0f32; self.cfg.max_boxes];
                for (i, b) in sample.boxes.iter().take(self.cfg.max_boxes).enumerate() {
                    padded[i * 4] = b[0];
                    padded[i * 4 + 1] = b[1];
                    padded[i * 4 + 2] = b[2];
                    padded[i * 4 + 3] = b[3];
                    mask[i] = 1.0;
                }
                boxes.extend_from_slice(&padded);
                box_mask.extend_from_slice(&mask);
            }

            if images.is_empty() {
                if skipped > 0 {
                    continue;
                } else {
                    return Ok(None);
                }
            }

            let (width, height) = expected_size.expect("batch size > 0 ensures size is set");
            let batch_len = frame_ids.len();
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

            let images = burn::tensor::Tensor::<B, 1>::from_floats(images.as_slice(), device)
                .reshape(image_shape);
            let boxes = burn::tensor::Tensor::<B, 1>::from_floats(boxes.as_slice(), device)
                .reshape(boxes_shape);
            let box_mask = burn::tensor::Tensor::<B, 1>::from_floats(box_mask.as_slice(), device)
                .reshape(mask_shape);
            let frame_ids = burn::tensor::Tensor::<B, 1>::from_floats(frame_ids.as_slice(), device)
                .reshape([batch_len]);

            return Ok(Some(BurnBatch {
                images,
                boxes,
                box_mask,
                frame_ids,
            }));
        }
    }
}
