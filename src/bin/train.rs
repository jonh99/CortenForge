#[cfg(feature = "burn_runtime")]
mod real {
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    use anyhow::Result;
    use burn::backend::autodiff::Autodiff;
    #[cfg(not(feature = "burn_wgpu"))]
    use burn::backend::ndarray::NdArray;
    use burn::lr_scheduler::{
        LrScheduler,
        cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
        linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    };
    use burn::module::Module;
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::AutodiffBackend;
    #[cfg(feature = "burn_wgpu")]
    use burn_wgpu::Wgpu;
    use clap::Parser;
    use colon_sim::burn_model::{TinyDet, TinyDetConfig, assign_targets_to_grid, nms};
    use colon_sim::tools::burn_dataset::{
        BatchIter, BurnBatch, DatasetConfig, SampleIndex, split_runs, split_runs_stratified,
    };
    use serde::{Deserialize, Serialize};
    use std::collections::HashSet;

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
        /// Scheduler type: linear or cosine.
        #[arg(long, default_value = "linear", value_parser = ["linear", "cosine"])]
        scheduler: String,
        /// Checkpoint every N steps (0 = disabled).
        #[arg(long, default_value_t = 0)]
        ckpt_every_steps: usize,
        /// Checkpoint every N epochs.
        #[arg(long, default_value_t = 1)]
        ckpt_every_epochs: usize,
        /// Validation objectness threshold for metric matching.
        #[arg(long, default_value_t = 0.3)]
        val_obj_thresh: f32,
        /// Validation IoU threshold for metric matching/NMS.
        #[arg(long, default_value_t = 0.5)]
        val_iou_thresh: f32,
        /// Optional comma-separated list of additional IoU thresholds for evaluation (e.g., "0.5,0.75").
        #[arg(long)]
        val_iou_sweep: Option<String>,
        /// Drop the last partial training batch (may help BN stability); val never drops.
        #[arg(long, default_value_t = false)]
        drop_last: bool,
        /// Early stop after N epochs without val IoU improvement (0 disables).
        #[arg(long, default_value_t = 0)]
        patience: usize,
        /// Minimum delta to consider val IoU improved.
        #[arg(long, default_value_t = 0.0)]
        patience_min_delta: f32,
        /// Optional separate validation root; when set, uses all runs under this path for val.
        #[arg(long)]
        real_val_dir: Option<String>,
        /// Print debug info for the first batch (targets and decoded preds).
        #[arg(long, default_value_t = false)]
        debug_batch: bool,
        /// Input root containing capture runs (default: assets/datasets/captures_filtered).
        #[arg(long, default_value = "assets/datasets/captures_filtered")]
        input_root: String,
        /// Use stratified split by box count (0/1/2+) instead of run-level random split.
        #[arg(long, default_value_t = false)]
        stratify_split: bool,
        /// Optional split manifest path; if present, load/save train/val label lists here for repeatable splits.
        #[arg(long)]
        split_manifest: Option<String>,
        /// Optional demo checkpoint path; if set, load this model at startup (skips optimizer/scheduler).
        #[arg(long)]
        demo_checkpoint: Option<String>,
        /// Optional metrics output path (JSONL); if set, appends per-epoch val metrics.
        #[arg(long)]
        metrics_out: Option<String>,
        /// Optional training status file (JSON) to report progress for UIs.
        #[arg(long)]
        status_file: Option<String>,
    }

    #[cfg(feature = "burn_wgpu")]
    type Backend = Wgpu<f32>;
    #[cfg(not(feature = "burn_wgpu"))]
    type Backend = NdArray<f32>;
    type ADBackend = Autodiff<Backend>;
    type Optim = OptimizerAdaptor<
        burn::optim::AdamW<<ADBackend as AutodiffBackend>::InnerBackend>,
        TinyDet<ADBackend>,
        ADBackend,
    >;

    #[derive(Clone, Copy)]
    enum Scheduler {
        Linear(LinearLrScheduler),
        Cosine(CosineAnnealingLrScheduler),
    }

    fn count_nonempty_samples(indices: &[SampleIndex]) -> usize {
        let mut count = 0usize;
        for idx in indices {
            if let Ok(n) = colon_sim::tools::burn_dataset::count_boxes(idx) {
                if n > 0 {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn main_impl() -> Result<()> {
        let args = TrainArgs::parse();
        let device = <ADBackend as burn::tensor::backend::Backend>::Device::default();
        let effective_seed = args.seed.or(Some(42));
        println!("Using seed {:?}", effective_seed);
        let batch_size = args.batch_size.max(1);
        let cfg = DatasetConfig {
            target_size: Some((128, 128)),
            flip_horizontal_prob: 0.5,
            max_boxes: 8,
            seed: effective_seed,
            drop_last: args.drop_last,
            ..Default::default()
        };

        let root = Path::new(&args.input_root);
        let indices = colon_sim::tools::burn_dataset::index_runs(root)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let (train_idx, default_val_idx) = {
            if let Some(manifest_path) = &args.split_manifest {
                let manifest_path = Path::new(manifest_path);
                if manifest_path.exists() {
                    println!("Loading split from {}", manifest_path.display());
                    load_split_manifest(manifest_path)?
                } else {
                    let split = if args.stratify_split {
                        split_runs_stratified(indices, args.val_ratio, args.seed)
                    } else {
                        split_runs(indices, args.val_ratio)
                    };
                    save_split_manifest(manifest_path, &split.0, &split.1, args.seed)?;
                    split
                }
            } else if args.stratify_split {
                split_runs_stratified(indices, args.val_ratio, args.seed)
            } else {
                split_runs(indices, args.val_ratio)
            }
        };
        let (val_idx, val_root) = if let Some(real_val_dir) = args.real_val_dir.as_ref() {
            let val_path = Path::new(real_val_dir).to_path_buf();
            let val_indices = colon_sim::tools::burn_dataset::index_runs(&val_path)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            (val_indices, val_path)
        } else {
            (default_val_idx, root.to_path_buf())
        };
        // Quick stats so users see progress before the first batch.
        let train_runs: HashSet<_> = train_idx.iter().map(|i| i.run_dir.clone()).collect();
        let val_runs: HashSet<_> = val_idx.iter().map(|i| i.run_dir.clone()).collect();
        let train_nonempty = count_nonempty_samples(&train_idx);
        let val_nonempty = count_nonempty_samples(&val_idx);
        println!(
            "Dataset: train runs {} samples {} (non-empty {}), val runs {} samples {} (non-empty {})",
            train_runs.len(),
            train_idx.len(),
            train_nonempty,
            val_runs.len(),
            val_idx.len(),
            val_nonempty
        );
        if train_nonempty == 0 {
            eprintln!("No non-empty training samples found (all frames missing boxes). Aborting.");
            return Ok(());
        }
        println!(
            "Entering training loop (log_every={}, batch_size={}, epochs={})",
            args.log_every, batch_size, args.epochs
        );
        let val_cfg = DatasetConfig {
            flip_horizontal_prob: 0.0,
            shuffle: false,
            drop_last: false,
            ..cfg
        };

        let mut model = TinyDet::<ADBackend>::new(TinyDetConfig::default(), &device);
        let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        let total_steps = {
            let per_epoch = (train_idx.len().max(1) + args.batch_size - 1) / args.batch_size;
            per_epoch.max(1) * args.epochs
        };
        let mut scheduler = match args.scheduler.as_str() {
            "cosine" => Scheduler::Cosine(
                CosineAnnealingLrSchedulerConfig::new(args.lr_start, total_steps.max(1))
                    .with_min_lr(args.lr_end)
                    .init(),
            ),
            _ => Scheduler::Linear(
                LinearLrSchedulerConfig::new(args.lr_start, args.lr_end, total_steps.max(1)).init(),
            ),
        };
        if let Some(demo) = &args.demo_checkpoint {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let path = Path::new(demo);
            match model.clone().load_file(path, &recorder, &device) {
                Ok(m) => {
                    model = m;
                    println!("Loaded demo checkpoint from {}", path.display());
                }
                Err(err) => {
                    eprintln!(
                        "Failed to load demo checkpoint {}: {:?}",
                        path.display(),
                        err
                    );
                }
            }
        } else {
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
        }

        let mut best_val = f32::NEG_INFINITY;
        let mut no_improve = 0usize;
        let mut stop_early = false;
        let mut debug_printed = false;
        for epoch in 0..args.epochs {
            println!("epoch {}", epoch + 1);
            if let Some(path) = args.status_file.as_ref() {
                let _ = write_status(
                    path,
                    serde_json::json!({
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "seed": effective_seed,
                        "status": "running"
                    }),
                );
            }
            let mut train = BatchIter::from_indices(train_idx.clone(), cfg.clone())
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let mut step = 0usize;
            let mut global_step = 0usize;

            while let Some(batch) = train
                .next_batch::<ADBackend>(batch_size, &device)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
            {
                step += 1;
                global_step += 1;
                let (obj_logits, box_logits) = model.forward(batch.images.clone());
                let (t_obj, t_boxes, t_mask) =
                    build_targets(&batch, obj_logits.dims()[2], obj_logits.dims()[3], &device)?;
                if args.debug_batch && !debug_printed {
                    debug_printed = true;
                    print_debug_batch(&obj_logits, &box_logits, &t_obj, &t_boxes, &t_mask)?;
                }
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
                let mean_iou_batch = mean_iou_host(&box_logits, &t_boxes, &t_obj);
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                let lr = scheduler_step(&mut scheduler);
                model = optim.step(lr, model, grads);

                if step % args.log_every == 0 {
                    println!(
                        "step {step}: loss={:.4}, mean_iou={:.4}",
                        loss_scalar, mean_iou_batch
                    );
                    if let Some(path) = args.status_file.as_ref() {
                        let _ = write_status(
                            path,
                            serde_json::json!({
                                "epoch": epoch + 1,
                                "epochs": args.epochs,
                                "step": step,
                                "lr": lr,
                                "loss": loss_scalar,
                                "status": "running"
                            }),
                        );
                    }
                }

                if args.ckpt_every_steps > 0 && global_step % args.ckpt_every_steps == 0 {
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
            }

            let mut val = BatchIter::from_indices(val_idx.clone(), val_cfg.clone())
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let mut iou_thresholds: Vec<f32> = vec![args.val_iou_thresh];
            if let Some(extra) = &args.val_iou_sweep {
                for part in extra.split(',') {
                    if let Ok(v) = part.trim().parse::<f32>() {
                        iou_thresholds.push(v);
                    }
                }
            }
            iou_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            iou_thresholds.dedup();
            struct ValAccum {
                iou: f32,
                val_sum: f32,
                batches: usize,
                tp: usize,
                fp: usize,
                fn_: usize,
                matched: usize,
                pr_curve: Vec<(f32, usize, usize, usize)>,
            }
            let mut val_accum: Vec<ValAccum> = iou_thresholds
                .iter()
                .map(|iou| ValAccum {
                    iou: *iou,
                    val_sum: 0.0,
                    batches: 0,
                    tp: 0,
                    fp: 0,
                    fn_: 0,
                    matched: 0,
                    pr_curve: (1..=19)
                        .map(|i| (i as f32 * 0.05, 0usize, 0usize, 0usize))
                        .collect(),
                })
                .collect();
            while let Some(val_batch) = val
                .next_batch::<ADBackend>(batch_size, &device)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
            {
                let (v_obj, v_boxes) = model.forward(val_batch.images.clone());
                for accum in val_accum.iter_mut() {
                    let (iou_sum, matched_count, batch_tp, batch_fp, batch_fn) = val_metrics_nms(
                        &v_obj,
                        &v_boxes,
                        &val_batch,
                        args.val_obj_thresh,
                        accum.iou,
                    );
                    accum.val_sum += iou_sum;
                    accum.matched += matched_count;
                    accum.tp += batch_tp;
                    accum.fp += batch_fp;
                    accum.fn_ += batch_fn;
                    for entry in accum.pr_curve.iter_mut() {
                        let th = entry.0;
                        let (btp, bfp, bfn) =
                            val_pr_threshold(&v_obj, &v_boxes, &val_batch, th, accum.iou);
                        entry.1 += btp;
                        entry.2 += bfp;
                        entry.3 += bfn;
                    }
                    accum.batches += 1;
                }
            }
            if val_accum.iter().any(|a| a.batches > 0) {
                if let Some(path) = &args.metrics_out {
                    if let Some(parent) = Path::new(path).parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let mut line = serde_json::json!({
                        "epoch": epoch + 1,
                        "seed": effective_seed,
                        "val_metrics": []
                    });
                    if let Some(arr) = line.get_mut("val_metrics").and_then(|v| v.as_array_mut()) {
                        for accum in val_accum.iter() {
                            if accum.batches == 0 {
                                continue;
                            }
                            let val_mean = if accum.matched > 0 {
                                accum.val_sum / accum.matched as f32
                            } else {
                                0.0
                            };
                            let precision = if accum.tp + accum.fp > 0 {
                                accum.tp as f32 / (accum.tp + accum.fp) as f32
                            } else {
                                0.0
                            };
                            let recall = if accum.tp + accum.fn_ > 0 {
                                accum.tp as f32 / (accum.tp + accum.fn_) as f32
                            } else {
                                0.0
                            };
                            let mut pr_points: Vec<(f32, usize, usize, usize)> =
                                accum.pr_curve.clone();
                            pr_points.sort_by(|a, b| {
                                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let map = compute_map(&pr_points);
                            arr.push(serde_json::json!({
                                "iou": accum.iou,
                                "mean_iou": val_mean,
                                "precision": precision,
                                "recall": recall,
                                "map": map,
                                "tp": accum.tp,
                                "fp": accum.fp,
                                "fn": accum.fn_,
                                "batches": accum.batches
                            }));
                        }
                    }
                    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
                        let _ = writeln!(f, "{}", line);
                    }
                }
                for accum in val_accum.iter() {
                    if accum.batches == 0 {
                        continue;
                    }
                    let val_mean = if accum.matched > 0 {
                        accum.val_sum / accum.matched as f32
                    } else {
                        0.0
                    };
                    let precision = if accum.tp + accum.fp > 0 {
                        accum.tp as f32 / (accum.tp + accum.fp) as f32
                    } else {
                        0.0
                    };
                    let recall = if accum.tp + accum.fn_ > 0 {
                        accum.tp as f32 / (accum.tp + accum.fn_) as f32
                    } else {
                        0.0
                    };
                    let mut pr_points: Vec<(f32, usize, usize, usize)> = accum.pr_curve.clone();
                    pr_points
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let map = compute_map(&pr_points);
                    println!(
                        "val@IoU{:.2} mean IoU = {:.4}, precision = {:.3}, recall = {:.3}, mAP ~= {:.3} (tp/fp/fn = {}/{}/{})",
                        accum.iou, val_mean, precision, recall, map, accum.tp, accum.fp, accum.fn_
                    );
                    if accum.iou == args.val_iou_thresh {
                        if val_mean > best_val + args.patience_min_delta {
                            best_val = val_mean;
                            no_improve = 0;
                        } else {
                            no_improve += 1;
                            if args.patience > 0 && no_improve >= args.patience {
                                println!(
                                    "Early stopping: no val improvement for {} epochs (best {:.4})",
                                    args.patience, best_val
                                );
                                stop_early = true;
                            }
                        }
                    }
                }
            } else {
                println!("No val batches found under {:?}", val_root);
            }
            if let Some(path) = args.status_file.as_ref() {
                let _ = write_status(
                    path,
                    serde_json::json!({
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "seed": effective_seed,
                        "status": "running"
                    }),
                );
            }

            if args.ckpt_every_epochs > 0 && (epoch + 1) % args.ckpt_every_epochs == 0 {
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

            if stop_early {
                break;
            }
        }
        if let Some(path) = args.status_file.as_ref() {
            let _ = write_status(
                path,
                serde_json::json!({
                    "epoch": args.epochs,
                    "epochs": args.epochs,
                    "seed": effective_seed,
                    "status": "done"
                }),
            );
        }

        Ok(())
    }

    fn write_status(path: &str, value: serde_json::Value) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut f = std::fs::File::create(path)?;
        use std::io::Write;
        f.write_all(serde_json::to_string(&value)?.as_bytes())?;
        Ok(())
    }

    fn print_debug_batch(
        obj_logits: &burn::tensor::Tensor<ADBackend, 4>,
        box_logits: &burn::tensor::Tensor<ADBackend, 4>,
        tgt_obj: &burn::tensor::Tensor<ADBackend, 4>,
        tgt_boxes: &burn::tensor::Tensor<ADBackend, 4>,
        tgt_mask: &burn::tensor::Tensor<ADBackend, 4>,
    ) -> Result<(), anyhow::Error> {
        let obj = obj_logits.to_data().to_vec::<f32>().unwrap_or_default();
        let boxes = box_logits.to_data().to_vec::<f32>().unwrap_or_default();
        let _tobj = tgt_obj.to_data().to_vec::<f32>().unwrap_or_default();
        let tboxes = tgt_boxes.to_data().to_vec::<f32>().unwrap_or_default();
        let tmask = tgt_mask.to_data().to_vec::<f32>().unwrap_or_default();
        let dims = obj_logits.dims();
        let hw = dims[2] * dims[3];
        let first_obj: Vec<f32> = obj
            .iter()
            .take(hw)
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        let first_boxes: Vec<[f32; 4]> = (0..4)
            .map(|c| {
                (0..hw)
                    .map(|i| {
                        let v = boxes[c * hw + i];
                        1.0 / (1.0 + (-v).exp())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .chunks_exact(4)
            .next()
            .map(|c| {
                (0..hw)
                    .take(4)
                    .map(|i| [c[0][i], c[1][i], c[2][i], c[3][i]])
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let tgt_boxes_first: Vec<[f32; 4]> = (0..hw)
            .take(4)
            .map(|i| {
                [
                    tboxes[i],
                    tboxes[hw + i],
                    tboxes[2 * hw + i],
                    tboxes[3 * hw + i],
                ]
            })
            .collect();
        println!(
            "DEBUG batch: obj min/max={:.3}/{:.3}, first obj cells (sigmoid) {:?}",
            first_obj.iter().cloned().fold(f32::INFINITY, f32::min),
            first_obj.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            &first_obj[..first_obj.len().min(8)]
        );
        println!(
            "DEBUG batch: first few pred boxes (sigmoid) {:?}, targets {:?}, target mask sum {:.1}",
            first_boxes,
            tgt_boxes_first,
            tmask.iter().sum::<f32>()
        );
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

        let dims = batch.boxes.dims();
        let batch_len = dims[0];
        let max_boxes = dims[1];
        let hw = grid_h * grid_w;

        let mut obj_all = vec![0.0f32; batch_len * hw];
        let mut boxes_all = vec![0.0f32; batch_len * 4 * hw];
        let mut mask_all = vec![0.0f32; batch_len * 4 * hw];

        for b in 0..batch_len {
            let mut valid_boxes = Vec::new();
            for i in 0..max_boxes {
                let m = box_mask_vec[b * max_boxes + i];
                if m > 0.0 {
                    let base = (b * max_boxes + i) * 4;
                    valid_boxes.push([
                        boxes_vec[base],
                        boxes_vec[base + 1],
                        boxes_vec[base + 2],
                        boxes_vec[base + 3],
                    ]);
                }
            }

            let (obj, tgt, mask) = assign_targets_to_grid(&valid_boxes, grid_h, grid_w);
            let obj_dst = &mut obj_all[b * hw..(b + 1) * hw];
            obj_dst.copy_from_slice(&obj);

            for c in 0..4 {
                let src = &tgt[c * hw..(c + 1) * hw];
                let dst = &mut boxes_all[(b * 4 + c) * hw..(b * 4 + c + 1) * hw];
                dst.copy_from_slice(src);

                let msrc = &mask[c * hw..(c + 1) * hw];
                let mdst = &mut mask_all[(b * 4 + c) * hw..(b * 4 + c + 1) * hw];
                mdst.copy_from_slice(msrc);
            }
        }

        let obj_t = burn::tensor::Tensor::<ADBackend, 1>::from_floats(obj_all.as_slice(), device)
            .reshape([batch_len, 1, grid_h, grid_w]);
        let boxes_t =
            burn::tensor::Tensor::<ADBackend, 1>::from_floats(boxes_all.as_slice(), device)
                .reshape([batch_len, 4, grid_h, grid_w]);
        let mask_t = burn::tensor::Tensor::<ADBackend, 1>::from_floats(mask_all.as_slice(), device)
            .reshape([batch_len, 4, grid_h, grid_w]);
        Ok((obj_t, boxes_t, mask_t))
    }

    fn scheduler_step(s: &mut Scheduler) -> f64 {
        match s {
            Scheduler::Linear(inner) => LrScheduler::<ADBackend>::step(inner),
            Scheduler::Cosine(inner) => LrScheduler::<ADBackend>::step(inner),
        }
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
            match scheduler {
                Scheduler::Linear(s) => {
                    if let Ok(record) = burn::record::Recorder::<ADBackend>::load(
                        &recorder,
                        sched_path.clone(),
                        device,
                    ) {
                        *s = burn::lr_scheduler::LrScheduler::<ADBackend>::load_record(*s, record);
                        println!("Loaded scheduler checkpoint (linear)");
                    }
                }
                Scheduler::Cosine(s) => {
                    if let Ok(record) = burn::record::Recorder::<ADBackend>::load(
                        &recorder,
                        sched_path.clone(),
                        device,
                    ) {
                        *s = burn::lr_scheduler::LrScheduler::<ADBackend>::load_record(*s, record);
                        println!("Loaded scheduler checkpoint (cosine)");
                    }
                }
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
        match scheduler {
            Scheduler::Linear(s) => {
                let sched_record = burn::lr_scheduler::LrScheduler::<ADBackend>::to_record(s);
                if let Err(err) = burn::record::Recorder::<ADBackend>::record(
                    &recorder,
                    sched_record,
                    sched_path.clone(),
                ) {
                    eprintln!("Failed to save scheduler checkpoint: {:?}", err);
                }
            }
            Scheduler::Cosine(s) => {
                let sched_record = burn::lr_scheduler::LrScheduler::<ADBackend>::to_record(s);
                if let Err(err) = burn::record::Recorder::<ADBackend>::record(
                    &recorder,
                    sched_record,
                    sched_path.clone(),
                ) {
                    eprintln!("Failed to save scheduler checkpoint: {:?}", err);
                }
            }
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

    fn val_metrics_nms(
        obj_logits: &burn::tensor::Tensor<ADBackend, 4>,
        box_logits: &burn::tensor::Tensor<ADBackend, 4>,
        batch: &BurnBatch<ADBackend>,
        obj_thresh: f32,
        iou_thresh: f32,
    ) -> (f32, usize, usize, usize, usize) {
        let obj = match obj_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return (0.0, 0, 0, 0, 0),
        };
        let boxes = match box_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return (0.0, 0, 0, 0, 0),
        };
        let dims = obj_logits.dims();
        if dims.len() != 4 {
            return (0.0, 0, 0, 0, 0);
        }
        let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let hw = h * w;

        // Collect GT boxes per image.
        let gt_boxes = match batch.boxes.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return (0.0, 0, 0, 0, 0),
        };
        let gt_mask = match batch.box_mask.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return (0.0, 0, 0, 0, 0),
        };
        let max_boxes = batch.boxes.dims()[1];

        let mut all_iou = 0.0f32;
        let mut all_matched = 0usize;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_total = 0usize;

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

            if preds.is_empty() {
                if !gt.is_empty() {
                    fn_total += gt.len();
                }
                continue;
            }
            preds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut boxes_only: Vec<[f32; 4]> = preds.iter().map(|p| p.1).collect();
            let scores_only: Vec<f32> = preds.iter().map(|p| p.0).collect();
            let keep = nms(&boxes_only, &scores_only, iou_thresh);
            boxes_only = keep.iter().map(|&i| boxes_only[i]).collect();

            if gt.is_empty() || boxes_only.is_empty() {
                if gt.is_empty() {
                    fp += boxes_only.len();
                } else {
                    fn_total += gt.len();
                }
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

            let matched_count = matched_gt.iter().filter(|m| **m).count();
            tp += matched_count;
            fp += preds.len().saturating_sub(matched_count);
            fn_total += gt.len().saturating_sub(matched_count);
        }

        if all_matched == 0 {
            (0.0, all_matched, tp, fp, fn_total)
        } else {
            (all_iou / all_matched as f32, all_matched, tp, fp, fn_total)
        }
    }

    fn val_pr_threshold(
        obj_logits: &burn::tensor::Tensor<ADBackend, 4>,
        box_logits: &burn::tensor::Tensor<ADBackend, 4>,
        batch: &BurnBatch<ADBackend>,
        obj_thresh: f32,
        iou_thresh: f32,
    ) -> (usize, usize, usize) {
        let (_, _, tp, fp, fn_) =
            val_metrics_nms(obj_logits, box_logits, batch, obj_thresh, iou_thresh);
        (tp, fp, fn_)
    }

    fn compute_map(points: &[(f32, usize, usize, usize)]) -> f32 {
        if points.is_empty() {
            return 0.0;
        }
        let mut pr: Vec<(f32, f32)> = points
            .iter()
            .map(|(_th, tp, fp, fn_)| {
                let tp = *tp as f32;
                let fp = *fp as f32;
                let fn_ = *fn_ as f32;
                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
                (recall, precision)
            })
            .collect();
        pr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut ap = 0.0f32;
        let mut prev_r = 0.0f32;
        for (r, p) in pr {
            let delta = (r - prev_r).max(0.0);
            ap += p * delta;
            prev_r = r;
        }
        ap
    }

    #[derive(Serialize, Deserialize)]
    struct SplitManifest {
        train: Vec<String>,
        val: Vec<String>,
        seed: Option<u64>,
    }

    fn save_split_manifest(
        path: &Path,
        train: &[colon_sim::tools::burn_dataset::SampleIndex],
        val: &[colon_sim::tools::burn_dataset::SampleIndex],
        seed: Option<u64>,
    ) -> Result<()> {
        let manifest = SplitManifest {
            train: train
                .iter()
                .map(|s| s.label_path.display().to_string())
                .collect(),
            val: val
                .iter()
                .map(|s| s.label_path.display().to_string())
                .collect(),
            seed,
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(path, json)?;
        println!("Saved split manifest to {}", path.display());
        Ok(())
    }

    fn load_split_manifest(
        path: &Path,
    ) -> Result<(
        Vec<colon_sim::tools::burn_dataset::SampleIndex>,
        Vec<colon_sim::tools::burn_dataset::SampleIndex>,
    )> {
        let raw = fs::read_to_string(path)?;
        let manifest: SplitManifest = serde_json::from_str(&raw)?;
        let train = manifest
            .train
            .iter()
            .map(|p| {
                let label_path = Path::new(p).to_path_buf();
                let run_dir = label_path
                    .parent()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new(""))
                    .to_path_buf();
                colon_sim::tools::burn_dataset::SampleIndex {
                    run_dir,
                    label_path,
                }
            })
            .collect();
        let val = manifest
            .val
            .iter()
            .map(|p| {
                let label_path = Path::new(p).to_path_buf();
                let run_dir = label_path
                    .parent()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new(""))
                    .to_path_buf();
                colon_sim::tools::burn_dataset::SampleIndex {
                    run_dir,
                    label_path,
                }
            })
            .collect();
        Ok((train, val))
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
