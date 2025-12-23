//! Minimal anchor-free detection head for Burn integration.
//!
//! Shapes (normalized coords):
//! - Input images: `[B, 3, H, W]`
//! - Objectness logits: `[B, 1, H, W]`
//! - Box offsets: `[B, 4, H, W]` (x_min, y_min, x_max, y_max in 0..1)

#![allow(unused)] // Gated behind `burn_runtime`.

use burn::module::{Ignored, Module};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::Reduction;
use burn::tensor::{Tensor, activation::sigmoid, backend::Backend};

/// Configuration for the detection head.
#[derive(Debug, Clone, Copy)]
pub struct TinyDetConfig {
    pub max_boxes: usize,
}

impl Default for TinyDetConfig {
    fn default() -> Self {
        Self { max_boxes: 16 }
    }
}

#[derive(Module, Debug)]
pub struct TinyDet<B: Backend> {
    stem: Conv2d<B>,
    head_obj: Conv2d<B>,
    head_box: Conv2d<B>,
    pub config: Ignored<TinyDetConfig>,
}

impl<B: Backend> TinyDet<B> {
    pub fn new(config: TinyDetConfig, device: &B::Device) -> Self {
        let stem = Conv2dConfig::new([3, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let head_obj = Conv2dConfig::new([32, 1], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        let head_box = Conv2dConfig::new([32, 4], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        Self {
            stem,
            head_obj,
            head_box,
            config: Ignored(config),
        }
    }

    /// Forward pass returning (objectness_logits, box_offsets).
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = burn::tensor::activation::relu(self.stem.forward(input));
        let obj_logits = self.head_obj.forward(x.clone());
        let box_logits = self.head_box.forward(x);
        (obj_logits, box_logits)
    }

    /// Compute detection loss (objectness BCE + Huber boxes, masked).
    pub fn loss(
        &self,
        obj_logits: Tensor<B, 4>,
        box_preds: Tensor<B, 4>,
        target_obj: Tensor<B, 4>,
        target_boxes: Tensor<B, 4>,
        box_mask: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        // Objectness: BCE with logits.
        let one = Tensor::from_floats([1.0], device);
        let eps = Tensor::from_floats([1e-6], device);
        let prob = sigmoid(obj_logits.clone());
        let obj_loss = (target_obj.clone() * (prob.clone() + eps.clone()).log()
            + (one.clone() - target_obj.clone()) * (one.clone() - prob + eps.clone()).log())
        .neg()
        .mean();

        // Box regression: Huber masked by presence.
        let diff = (box_preds.clone() - target_boxes.clone()).abs();
        let delta = Tensor::from_floats([1.0], device);
        let mask_small = diff.clone().lower(delta.clone());
        let pow_two = Tensor::from_floats([2.0], device);
        let small = Tensor::from_floats([0.5], device) * diff.clone().powf(pow_two);
        let large = diff - Tensor::from_floats([0.5], device);
        let mask_float = mask_small.clone().float();
        let per_elem =
            small * mask_float.clone() + large * (Tensor::from_floats([1.0], device) - mask_float);

        let masked = per_elem * box_mask.clone();
        let denom = box_mask.clone().sum() + Tensor::from_floats([1e-6], device);
        let box_loss = masked.sum() / denom;

        let ciou_loss = self.ciou_loss(box_preds, target_boxes, box_mask, device);

        obj_loss + box_loss + ciou_loss
    }

    fn ciou_loss(
        &self,
        pred: Tensor<B, 4>,
        target: Tensor<B, 4>,
        mask: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let eps = Tensor::from_floats([1e-6], device);
        let p = sigmoid(pred).clamp(0.0, 1.0);
        let t = target.clamp(0.0, 1.0);
        let obj_mask = mask.narrow(1, 0, 1);

        let p = p.clone().reshape([p.dims()[0], 4, p.dims()[2] * p.dims()[3]]);
        let t = t.clone().reshape([t.dims()[0], 4, t.dims()[2] * t.dims()[3]]);
        let m = obj_mask.clone().reshape([
            obj_mask.dims()[0],
            1,
            obj_mask.dims()[2] * obj_mask.dims()[3],
        ]);

        let px0 = p.clone().narrow(1, 0, 1);
        let py0 = p.clone().narrow(1, 1, 1);
        let px1 = p.clone().narrow(1, 2, 1);
        let py1 = p.clone().narrow(1, 3, 1);

        let tx0 = t.clone().narrow(1, 0, 1);
        let ty0 = t.clone().narrow(1, 1, 1);
        let tx1 = t.clone().narrow(1, 2, 1);
        let ty1 = t.clone().narrow(1, 3, 1);

        let inter_x0 = px0.clone().max_pair(tx0.clone());
        let inter_y0 = py0.clone().max_pair(ty0.clone());
        let inter_x1 = px1.clone().min_pair(tx1.clone());
        let inter_y1 = py1.clone().min_pair(ty1.clone());

        let inter_w = (inter_x1 - inter_x0).clamp_min(0.0);
        let inter_h = (inter_y1 - inter_y0).clamp_min(0.0);
        let inter = inter_w.clone() * inter_h.clone();

        let area_p = (px1.clone() - px0.clone()).clamp_min(0.0)
            * (py1.clone() - py0.clone()).clamp_min(0.0);
        let area_t = (tx1.clone() - tx0.clone()).clamp_min(0.0)
            * (ty1.clone() - ty0.clone()).clamp_min(0.0);
        let union = area_p + area_t - inter.clone() + eps.clone();
        let iou = inter / union;

        let half = Tensor::from_floats([0.5], device);
        let two = Tensor::from_floats([2.0], device);

        let cx_p = (px0.clone() + px1.clone()) * half.clone();
        let cy_p = (py0.clone() + py1.clone()) * half.clone();
        let cx_t = (tx0.clone() + tx1.clone()) * half.clone();
        let cy_t = (ty0.clone() + ty1.clone()) * half;

        let rho2 = (cx_p - cx_t).powf(two.clone()) + (cy_p - cy_t).powf(two.clone());

        let enc_x0 = px0.min_pair(tx0);
        let enc_y0 = py0.min_pair(ty0);
        let enc_x1 = px1.max_pair(tx1);
        let enc_y1 = py1.max_pair(ty1);
        let c2 = (enc_x1 - enc_x0)
            .clamp_min(1e-6)
            .powf(two.clone())
            + (enc_y1 - enc_y0).clamp_min(1e-6).powf(two);

        let diou = iou - rho2 / c2;
        let ciou_loss = (Tensor::from_floats([1.0], device) - diou) * m.clone();

        let denom = m.sum().clamp_min(1e-6);
        ciou_loss.sum() / denom
    }
}

/// Assign ground-truth boxes to a grid (H, W) by nearest center, producing per-cell targets.
pub fn assign_targets_to_grid(
    boxes: &[[f32; 4]],
    grid_h: usize,
    grid_w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut obj = vec![0.0f32; grid_h * grid_w];
    let mut tgt = vec![0.0f32; grid_h * grid_w * 4];
    let mut mask = vec![0.0f32; grid_h * grid_w * 4];
    let mut score = vec![0.0f32; grid_h * grid_w];

    for b in boxes {
        let cx = (b[0] + b[2]) * 0.5;
        let cy = (b[1] + b[3]) * 0.5;
        let gx = (cx * grid_w as f32).clamp(0.0, (grid_w - 1) as f32) as usize;
        let gy = (cy * grid_h as f32).clamp(0.0, (grid_h - 1) as f32) as usize;
        let idx = gy * grid_w + gx;
        // Score by IoU with the cell itself to pick the best-fitting box per cell.
        let cell = [
            gx as f32 / grid_w as f32,
            gy as f32 / grid_h as f32,
            (gx + 1) as f32 / grid_w as f32,
            (gy + 1) as f32 / grid_h as f32,
        ];
        let inter_x0 = b[0].max(cell[0]);
        let inter_y0 = b[1].max(cell[1]);
        let inter_x1 = b[2].min(cell[2]);
        let inter_y1 = b[3].min(cell[3]);
        let inter = (inter_x1 - inter_x0).max(0.0) * (inter_y1 - inter_y0).max(0.0);
        let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
        let area_c = (cell[2] - cell[0]) * (cell[3] - cell[1]);
        let iou = inter / (area_b + area_c - inter + 1e-6);

        if iou > score[idx] {
            score[idx] = iou;
            obj[idx] = 1.0;
            let base = idx * 4;
            tgt[base] = b[0];
            tgt[base + 1] = b[1];
            tgt[base + 2] = b[2];
            tgt[base + 3] = b[3];
            mask[base] = 1.0;
            mask[base + 1] = 1.0;
            mask[base + 2] = 1.0;
            mask[base + 3] = 1.0;
        }
    }

    (obj, tgt, mask)
}

/// Simple NMS over boxes `[x0,y0,x1,y1]` with scores.
pub fn nms(mut boxes: Vec<[f32; 4]>, mut scores: Vec<f32>, iou_thresh: f32) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..boxes.len()).collect();
    idxs.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    while let Some(i) = idxs.pop() {
        keep.push(i);
        idxs.retain(|&j| iou(&boxes[i], &boxes[j]) <= iou_thresh);
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter + 1e-6;
    inter / union
}
