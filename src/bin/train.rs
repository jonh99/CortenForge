#![cfg(feature = "burn_runtime")]

use anyhow::Result;
use burn_ndarray::NdArray;
use colon_sim::burn_model::{TinyDet, TinyDetConfig, assign_targets_to_grid};
use colon_sim::tools::burn_dataset::{BatchIter, DatasetConfig, split_runs};
use std::path::Path;

type Backend = NdArray<f32>;

fn main() -> Result<()> {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let cfg = DatasetConfig {
        target_size: Some((128, 128)),
        flip_horizontal_prob: 0.5,
        max_boxes: 8,
        ..Default::default()
    };

    let root = Path::new("assets/datasets/captures");
    let indices =
        colon_sim::tools::burn_dataset::index_runs(root).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let (train_idx, _) = split_runs(indices, 0.2);
    let mut train =
        BatchIter::from_indices(train_idx, cfg).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let model = TinyDet::<Backend>::new(TinyDetConfig::default(), &device);

    if let Some(batch) = train
        .next_batch::<Backend>(1, &device)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
    {
        let (obj_logits, box_logits) = model.forward(batch.images.clone());
        let (t_obj, t_boxes, t_mask) =
            build_targets(&batch, obj_logits.dims()[2], obj_logits.dims()[3], &device)?;
        let loss = model.loss(obj_logits, box_logits, t_obj, t_boxes, t_mask, &device);
        let loss_val = loss.to_data().to_vec::<f32>().unwrap_or_default();
        println!("dummy training step, loss={:?}", loss_val.first());
    } else {
        println!("No training batches found under {:?}", root);
    }

    Ok(())
}

fn build_targets(
    batch: &colon_sim::tools::burn_dataset::BurnBatch<Backend>,
    grid_h: usize,
    grid_w: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<(
    burn::tensor::Tensor<Backend, 4>,
    burn::tensor::Tensor<Backend, 4>,
    burn::tensor::Tensor<Backend, 4>,
)> {
    // Supports batch_size = 1 for now.
    let boxes_vec = batch
        .boxes
        .to_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let box_mask_vec = batch
        .box_mask
        .to_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let max_boxes = batch.boxes.dims()[1];
    let mut valid_boxes = Vec::new();
    for i in 0..max_boxes {
        let base = i * 4;
        if box_mask_vec[i] > 0.0 {
            valid_boxes.push([
                boxes_vec[base],
                boxes_vec[base + 1],
                boxes_vec[base + 2],
                boxes_vec[base + 3],
            ]);
        }
    }

    let (obj, tgt, mask) = assign_targets_to_grid(&valid_boxes, grid_h, grid_w);
    let obj_t = burn::tensor::Tensor::<Backend, 4>::from_floats(obj.as_slice(), device)
        .reshape([1, 1, grid_h, grid_w]);
    let boxes_t = burn::tensor::Tensor::<Backend, 4>::from_floats(tgt.as_slice(), device)
        .reshape([1, 4, grid_h, grid_w]);
    let mask_t = burn::tensor::Tensor::<Backend, 4>::from_floats(mask.as_slice(), device)
        .reshape([1, 4, grid_h, grid_w]);
    Ok((obj_t, boxes_t, mask_t))
}

#[cfg(not(feature = "burn_runtime"))]
fn main() {
    eprintln!("Enable --features burn_runtime to run the training harness.");
}
