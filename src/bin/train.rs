#[cfg(feature = "burn_runtime")]
mod real {
    use std::fs;
    use std::path::Path;

    use anyhow::Result;
    use burn::backend::{autodiff::Autodiff, ndarray::NdArray};
    use burn::lr_scheduler::{linear::LinearLrSchedulerConfig, LrScheduler};
    use burn::module::Module;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::AutodiffBackend;
    use clap::Parser;
    use colon_sim::burn_model::{nms, assign_targets_to_grid, TinyDet, TinyDetConfig};
    use colon_sim::tools::burn_dataset::{
        BatchIter, BurnBatch, DatasetConfig, split_runs,
    };

    #[derive(Parser, Debug)]
    #[command(name = "train", about = "TinyDet training harness")]
    struct TrainArgs {
        /// Training batch size.
        #[arg(long, default_value_t = 2)]
        batch_size: usize,
        /// Number of epochs to run.
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        /// Log every N steps.
        #[arg(long, default_value_t = 1)]
        log_every: usize,
        /// Starting learning rate.
        #[arg(long, default_value_t = 1e-3)]
        lr_start: f64,
        /// Ending learning rate.
        #[arg(long, default_value_t = 1e-4)]
        lr_end: f64,
        /// Validation ratio (0..1).
        #[arg(long, default_value_t = 0.2)]
        val_ratio: f32,
        /// Optional shuffle seed for deterministic splits/batching.
        #[arg(long)]
        seed: Option<u64>,
        /// Checkpoint directory.
        #[arg(long, default_value = "checkpoints")]
        ckpt_dir: String,
    }

    type Backend = NdArray<f32>;
    type ADBackend = Autodiff<Backend>;
    type Optim = OptimizerAdaptor<
        burn::optim::AdamW<<ADBackend as AutodiffBackend>::InnerBackend>,
        TinyDet<ADBackend>,
        ADBackend,
    >;
    type Scheduler = burn::lr_scheduler::linear::LinearLrScheduler;

    pub fn main_impl() -> Result<()> {
        let args = TrainArgs::parse();
        let device = <ADBackend as burn::tensor::backend::Backend>::Device::default();
        let cfg = DatasetConfig {
            target_size: Some((128, 128)),
            flip_horizontal_prob: 0.5,
            max_boxes: 8,
            seed: args.seed,
            ..Default::default()
        };

        let root = Path::new("assets/datasets/captures");
        let indices = colon_sim::tools::burn_dataset::index_runs(root)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let (train_idx, val_idx) = split_runs(indices, args.val_ratio);
        let val_cfg = DatasetConfig {
            flip_horizontal_prob: 0.0,
            shuffle: false,
            ..cfg
        };

        let mut model = TinyDet::<ADBackend>::new(TinyDetConfig::default(), &device);
        let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        let total_steps = {
            let per_epoch = (train_idx.len().max(1) + args.batch_size - 1) / args.batch_size;
            per_epoch.max(1) * args.epochs
        };
        let mut scheduler =
            LinearLrSchedulerConfig::new(args.lr_start, args.lr_end, total_steps.max(1)).init();
        load_checkpoint(
            &args.ckpt_dir,
            "tinydet",
            "tinydet_optim",
            "tinydet_sched",
            &device,
            &mut model,
            &mut optim,
            &mut scheduler,
        );

        for epoch in 0..args.epochs {
            println!("epoch {}", epoch + 1);
            let mut train = BatchIter::from_indices(train_idx.clone(), cfg.clone())
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let mut step = 0usize;

            while let Some(batch) = train
                .next_batch::<ADBackend>(args.batch_size, &device)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
            {
                step += 1;
                let (obj_logits, box_logits) = model.forward(batch.images.clone());
                let (t_obj, t_boxes, t_mask) =
                    build_targets(&batch, obj_logits.dims()[2], obj_logits.dims()[3], &device)?;
                let loss = model.loss(
                    obj_logits,
                    box_logits.clone(),
                    t_obj.clone(),
                    t_boxes.clone(),
                    t_mask.clone(),
                    &device,
                );
                let loss_scalar = loss
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                let lr = <Scheduler as LrScheduler<ADBackend>>::step(&mut scheduler);
                model = optim.step(lr, model, grads);

                if step % args.log_every == 0 {
                    let mean_iou = mean_iou_host(&box_logits, &t_boxes, &t_obj);
                    println!(
                        "step {step}: loss={:.4}, mean_iou={:.4}",
                        loss_scalar,
                        mean_iou
                    );
                }
            }

            let mut val = BatchIter::from_indices(val_idx.clone(), val_cfg.clone())
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let mut val_sum = 0.0f32;
            let mut val_batches = 0usize;
            while let Some(val_batch) = val
                .next_batch::<ADBackend>(args.batch_size, &device)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
            {
                let (v_obj, v_boxes) = model.forward(val_batch.images.clone());
                let val_iou = mean_iou_nms(&v_obj, &v_boxes, &val_batch, 0.3, 0.5);
                val_sum += val_iou;
                val_batches += 1;
            }
            if val_batches > 0 {
                println!("val mean IoU = {:.4}", val_sum / val_batches as f32);
            } else {
                println!("No val batches found under {:?}", root);
            }

            save_checkpoint(
                &args.ckpt_dir,
                "tinydet",
                "tinydet_optim",
                "tinydet_sched",
                &device,
                &model,
                &optim,
                &scheduler,
            );
        }

        Ok(())
    }

    fn build_targets(
        batch: &colon_sim::tools::burn_dataset::BurnBatch<ADBackend>,
        grid_h: usize,
        grid_w: usize,
        device: &<ADBackend as burn::tensor::backend::Backend>::Device,
    ) -> Result<(
        burn::tensor::Tensor<ADBackend, 4>,
        burn::tensor::Tensor<ADBackend, 4>,
        burn::tensor::Tensor<ADBackend, 4>,
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
        let obj_t = burn::tensor::Tensor::<ADBackend, 4>::from_floats(obj.as_slice(), device)
            .reshape([1, 1, grid_h, grid_w]);
        let boxes_t = burn::tensor::Tensor::<ADBackend, 4>::from_floats(tgt.as_slice(), device)
            .reshape([1, 4, grid_h, grid_w]);
        let mask_t = burn::tensor::Tensor::<ADBackend, 4>::from_floats(mask.as_slice(), device)
            .reshape([1, 4, grid_h, grid_w]);
        Ok((obj_t, boxes_t, mask_t))
    }

    fn load_checkpoint(
        dir: &str,
        model_name: &str,
        optim_name: &str,
        sched_name: &str,
        device: &<ADBackend as burn::tensor::backend::Backend>::Device,
        model: &mut TinyDet<ADBackend>,
        optim: &mut Optim,
        scheduler: &mut Scheduler,
    ) {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let model_path = Path::new(dir).join(model_name);
        let optim_path = Path::new(dir).join(optim_name);
        let sched_path = Path::new(dir).join(sched_name);

        if model_path.with_extension("bin").exists() {
            if let Ok(loaded) = model.clone().load_file(model_path, &recorder, device) {
                *model = loaded;
                println!("Loaded model checkpoint");
            }
        }
        if optim_path.with_extension("bin").exists() {
            if let Ok(record) = recorder.load(optim_path, device) {
                *optim = optim.clone().load_record(record);
                println!("Loaded optimizer checkpoint");
            }
        }
        if sched_path.with_extension("bin").exists() {
            if let Ok(record) =
                burn::record::Recorder::<ADBackend>::load(&recorder, sched_path, device)
            {
                *scheduler =
                    burn::lr_scheduler::LrScheduler::<ADBackend>::load_record(*scheduler, record);
                println!("Loaded scheduler checkpoint");
            }
        }
    }

    fn save_checkpoint(
        dir: &str,
        model_name: &str,
        optim_name: &str,
        sched_name: &str,
        device: &<ADBackend as burn::tensor::backend::Backend>::Device,
        model: &TinyDet<ADBackend>,
        optim: &Optim,
        scheduler: &Scheduler,
    ) {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let _ = fs::create_dir_all(dir);
        let model_path = Path::new(dir).join(model_name);
        let optim_path = Path::new(dir).join(optim_name);
        let sched_path = Path::new(dir).join(sched_name);
        if let Err(err) = recorder.record(model.clone().into_record(), model_path) {
            eprintln!("Failed to save model checkpoint: {:?}", err);
        }
        if let Err(err) = recorder.record(optim.to_record(), optim_path) {
            eprintln!("Failed to save optimizer checkpoint: {:?}", err);
        }
        let sched_record =
            burn::lr_scheduler::LrScheduler::<ADBackend>::to_record(scheduler);
        if let Err(err) =
            burn::record::Recorder::<ADBackend>::record(&recorder, sched_record, sched_path)
        {
            eprintln!("Failed to save scheduler checkpoint: {:?}", err);
        }
        // touch device to keep warning-free
        let _ = device;
    }

    fn mean_iou_host(
        pred_boxes: &burn::tensor::Tensor<ADBackend, 4>,
        target_boxes: &burn::tensor::Tensor<ADBackend, 4>,
        target_obj: &burn::tensor::Tensor<ADBackend, 4>,
    ) -> f32 {
        fn fast_sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        let pb = match pred_boxes.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let tb = match target_boxes.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let tobj = match target_obj.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let dims = pred_boxes.dims();
        if dims.len() != 4 {
            return 0.0;
        }
        let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let hw = h * w;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for bi in 0..b {
            for yi in 0..h {
                for xi in 0..w {
                    let idx = bi * hw + yi * w + xi;
                    if tobj[idx] <= 0.5 {
                        continue;
                    }
                    let base = bi * 4 * hw + yi * w + xi;
                    let mut pb_vals = [
                        fast_sigmoid(pb[base]),
                        fast_sigmoid(pb[base + hw]),
                        fast_sigmoid(pb[base + 2 * hw]),
                        fast_sigmoid(pb[base + 3 * hw]),
                    ];
                    pb_vals[0] = pb_vals[0].clamp(0.0, 1.0);
                    pb_vals[1] = pb_vals[1].clamp(0.0, 1.0);
                    pb_vals[2] = pb_vals[2].clamp(pb_vals[0], 1.0);
                    pb_vals[3] = pb_vals[3].clamp(pb_vals[1], 1.0);

                    let tb_vals = [
                        tb[base].clamp(0.0, 1.0),
                        tb[base + hw].clamp(0.0, 1.0),
                        tb[base + 2 * hw].clamp(0.0, 1.0),
                        tb[base + 3 * hw].clamp(0.0, 1.0),
                    ];

                    let inter_x0 = pb_vals[0].max(tb_vals[0]);
                    let inter_y0 = pb_vals[1].max(tb_vals[1]);
                    let inter_x1 = pb_vals[2].min(tb_vals[2]);
                    let inter_y1 = pb_vals[3].min(tb_vals[3]);
                    let inter_w = (inter_x1 - inter_x0).max(0.0);
                    let inter_h = (inter_y1 - inter_y0).max(0.0);
                    let inter = inter_w * inter_h;

                    let area_p =
                        (pb_vals[2] - pb_vals[0]).max(0.0) * (pb_vals[3] - pb_vals[1]).max(0.0);
                    let area_t =
                        (tb_vals[2] - tb_vals[0]).max(0.0) * (tb_vals[3] - tb_vals[1]).max(0.0);
                    let union = (area_p + area_t - inter).max(1e-6);
                    let iou = inter / union;
                    sum += iou;
                    count += 1.0;
                }
            }
        }
        if count == 0.0 { 0.0 } else { sum / count }
    }

    fn mean_iou_nms(
        obj_logits: &burn::tensor::Tensor<ADBackend, 4>,
        box_logits: &burn::tensor::Tensor<ADBackend, 4>,
        batch: &BurnBatch<ADBackend>,
        obj_thresh: f32,
        iou_thresh: f32,
    ) -> f32 {
        let obj = match obj_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let boxes = match box_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let dims = obj_logits.dims();
        if dims.len() != 4 {
            return 0.0;
        }
        let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let hw = h * w;

        // Collect GT boxes per image.
        let gt_boxes = match batch.boxes.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let gt_mask = match batch.box_mask.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let max_boxes = batch.boxes.dims()[1];

        let mut all_iou = 0.0f32;
        let mut all_matched = 0usize;

        for bi in 0..b {
            // Decode predictions for image bi.
            let mut preds = Vec::new();
            for yi in 0..h {
                for xi in 0..w {
                    let idx = bi * hw + yi * w + xi;
                    let score = 1.0 / (1.0 + (-obj[idx]).exp());
                    if score < obj_thresh {
                        continue;
                    }
                    let base = bi * 4 * hw + yi * w + xi;
                    let mut pb = [
                        1.0 / (1.0 + (-boxes[base]).exp()),
                        1.0 / (1.0 + (-boxes[base + hw]).exp()),
                        1.0 / (1.0 + (-boxes[base + 2 * hw]).exp()),
                        1.0 / (1.0 + (-boxes[base + 3 * hw]).exp()),
                    ];
                    pb[0] = pb[0].clamp(0.0, 1.0);
                    pb[1] = pb[1].clamp(0.0, 1.0);
                    pb[2] = pb[2].clamp(pb[0], 1.0);
                    pb[3] = pb[3].clamp(pb[1], 1.0);
                    preds.push((score, pb));
                }
            }
            if preds.is_empty() {
                continue;
            }
            preds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut boxes_only: Vec<[f32; 4]> = preds.iter().map(|p| p.1).collect();
            let scores_only: Vec<f32> = preds.iter().map(|p| p.0).collect();
            let keep = nms(boxes_only.clone(), scores_only, iou_thresh);
            boxes_only = keep.iter().map(|&i| boxes_only[i]).collect();

            let mut gt = Vec::new();
            for gi in 0..max_boxes {
                let m = gt_mask[bi * max_boxes + gi];
                if m <= 0.0 {
                    continue;
                }
                let base = (bi * max_boxes + gi) * 4;
                gt.push([
                    gt_boxes[base].clamp(0.0, 1.0),
                    gt_boxes[base + 1].clamp(0.0, 1.0),
                    gt_boxes[base + 2].clamp(0.0, 1.0),
                    gt_boxes[base + 3].clamp(0.0, 1.0),
                ]);
            }
            if gt.is_empty() || boxes_only.is_empty() {
                continue;
            }

            fn iou_pair(a: &[f32; 4], b: &[f32; 4]) -> f32 {
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

            let mut matched_gt = vec![false; gt.len()];
            for pb in boxes_only {
                let mut best = -1isize;
                let mut best_iou = 0.0f32;
                for (gidx, gb) in gt.iter().enumerate() {
                    if matched_gt[gidx] {
                        continue;
                    }
                    let i = iou_pair(gb, &pb);
                    if i > best_iou {
                        best_iou = i;
                        best = gidx as isize;
                    }
                }
                if best >= 0 && best_iou >= iou_thresh {
                    matched_gt[best as usize] = true;
                    all_iou += best_iou;
                    all_matched += 1;
                }
            }
        }

        if all_matched == 0 {
            0.0
        } else {
            all_iou / all_matched as f32
        }
    }

    #[allow(dead_code)]
    pub fn main() -> Result<()> {
        main_impl()
    }
}

#[cfg(feature = "burn_runtime")]
fn main() -> anyhow::Result<()> {
    real::main_impl()
}

#[cfg(not(feature = "burn_runtime"))]
fn main() {
    eprintln!("Enable --features burn_runtime to run the training harness.");
}
