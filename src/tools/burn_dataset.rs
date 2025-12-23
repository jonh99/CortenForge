use image::imageops::FilterType;
use serde::Deserialize;
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
        use rand::seq::SliceRandom;
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

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Resize all images to this (width, height). If None, images must already share shape.
    pub target_size: Option<(u32, u32)>,
    /// How to resize images when target_size is set.
    pub resize_mode: ResizeMode,
    /// Cap on boxes per image; extras are dropped, padding uses zeros with mask.
    pub max_boxes: usize,
    /// Shuffle samples before iteration.
    pub shuffle: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            target_size: Some((512, 512)),
            resize_mode: ResizeMode::Letterbox,
            max_boxes: 16,
            shuffle: true,
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
                max_boxes: usize::MAX,
                shuffle: false,
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
                let resized = image::imageops::resize(&img, w, h, FilterType::Triangle);
                let boxes = normalize_boxes(&meta.polyp_labels, w, h);
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
                let (resized_img, pad_w, pad_h) = letterbox_resize(&img, w, h)?;
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

    build_sample_from_image(
        img,
        width,
        height,
        normalize_boxes(&meta.polyp_labels, width, height),
        meta.frame_id,
        cfg.max_boxes,
    )
}

fn build_sample_from_image(
    img: image::RgbImage,
    width: u32,
    height: u32,
    mut boxes: Vec<[f32; 4]>,
    frame_id: u64,
    max_boxes: usize,
) -> Result<DatasetSample, Box<dyn Error + Send + Sync>> {
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
        if cfg.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
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
    ) -> Result<Option<BurnBatch<B>>, Box<dyn Error + Send + Sync>> {
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

        for idx in slice {
            let sample = load_sample(idx, &self.cfg)?;

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

        let (width, height) = expected_size.expect("batch size > 0 ensures size is set");
        let image_shape = [slice.len(), 3, height as usize, width as usize];
        let boxes_shape = [slice.len(), self.cfg.max_boxes, 4];
        let mask_shape = [slice.len(), self.cfg.max_boxes];

        let images = burn::tensor::Tensor::<B, 4>::from_floats(images, image_shape);
        let boxes = burn::tensor::Tensor::<B, 3>::from_floats(boxes, boxes_shape);
        let box_mask = burn::tensor::Tensor::<B, 2>::from_floats(box_mask, mask_shape);
        let frame_ids = burn::tensor::Tensor::<B, 1>::from_floats(frame_ids, [slice.len()]);

        Ok(Some(BurnBatch {
            images,
            boxes,
            box_mask,
            frame_ids,
        }))
    }
}
