use crate::{InferenceBackend, InferenceModel, InferenceModelConfig};
use burn::module::Module;
use burn::tensor::TensorData;
use std::path::Path;
use std::sync::{Arc, Mutex};
use vision_core::interfaces::{DetectionResult, Detector, Frame};

/// Thresholds for inference (objectness + IoU).
#[derive(Debug, Clone, Copy)]
pub struct InferenceThresholds {
    pub obj_thresh: f32,
    pub iou_thresh: f32,
}

impl Default for InferenceThresholds {
    fn default() -> Self {
        Self {
            obj_thresh: 0.3,
            iou_thresh: 0.5,
        }
    }
}

/// Heuristic detector placeholder; used when Burn weights are unavailable.
struct HeuristicDetector {
    obj_thresh: f32,
}

impl Detector for HeuristicDetector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult {
        let confidence = self.obj_thresh;
        DetectionResult {
            frame_id: frame.id,
            positive: confidence >= self.obj_thresh,
            confidence,
            boxes: Vec::new(),
            scores: Vec::new(),
        }
    }
}

struct BurnTinyDetDetector {
    model: Arc<Mutex<InferenceModel<InferenceBackend>>>,
    obj_thresh: f32,
    #[allow(dead_code)]
    iou_thresh: f32,
}

impl BurnTinyDetDetector {
    fn frame_to_tensor(&self, frame: &Frame) -> TensorData {
        let (w, h) = frame.size;
        if let Some(rgba) = &frame.rgba {
            let mut mean = [0f32; 3];
            let mut count = 0usize;
            for chunk in rgba.chunks_exact(4) {
                mean[0] += chunk[0] as f32;
                mean[1] += chunk[1] as f32;
                mean[2] += chunk[2] as f32;
                count += 1;
            }
            if count > 0 {
                mean[0] /= count as f32 * 255.0;
                mean[1] /= count as f32 * 255.0;
                mean[2] /= count as f32 * 255.0;
            }
            TensorData::new(vec![mean[0], mean[1], mean[2], w as f32 / h as f32], [1, 4])
        } else {
            TensorData::new(vec![0.0, 0.0, 0.0, w as f32 / h as f32], [1, 4])
        }
    }
}

impl Detector for BurnTinyDetDetector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult {
        let input = self.frame_to_tensor(frame);
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
        let model = self.model.lock().expect("model mutex poisoned");
        let logits = model.forward(burn::tensor::Tensor::<InferenceBackend, 2>::from_data(
            input, &device,
        ));
        let scores = logits.into_data().to_vec::<f32>().unwrap_or_default();
        let confidence = scores.first().copied().unwrap_or(0.0);
        DetectionResult {
            frame_id: frame.id,
            positive: confidence >= self.obj_thresh,
            confidence,
            boxes: Vec::new(),
            scores,
        }
    }
}

/// Factory that will load Burn checkpoints when available; currently returns heuristic.
pub struct InferenceFactory;

impl InferenceFactory {
    pub fn build(
        &self,
        thresh: InferenceThresholds,
        weights: Option<&Path>,
    ) -> Box<dyn vision_core::interfaces::Detector + Send + Sync> {
        if let Some(det) = self.try_load_burn_detector(thresh, weights) {
            return det;
        }
        eprintln!("InferenceFactory: no valid checkpoint provided; using heuristic detector.");
        Box::new(HeuristicDetector {
            obj_thresh: thresh.obj_thresh,
        })
    }

    fn try_load_burn_detector(
        &self,
        thresh: InferenceThresholds,
        weights: Option<&Path>,
    ) -> Option<Box<dyn vision_core::interfaces::Detector + Send + Sync>> {
        let path = weights?;
        if !path.exists() {
            return None;
        }
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        match InferenceModel::<InferenceBackend>::new(InferenceModelConfig::default(), &device)
            .load_file(path, &recorder, &device)
        {
            Ok(model) => Some(Box::new(BurnTinyDetDetector {
                model: Arc::new(Mutex::new(model)),
                obj_thresh: thresh.obj_thresh,
                iou_thresh: thresh.iou_thresh,
            })),
            Err(err) => {
                eprintln!(
                    "Failed to load detector checkpoint {:?}: {err}. Falling back to heuristic.",
                    path
                );
                None
            }
        }
    }
}
