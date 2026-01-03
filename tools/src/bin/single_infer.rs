use clap::Parser;
use image::io::Reader as ImageReader;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use cli_support::common::ThresholdOpts;
use inference::prelude::{InferenceFactory, InferenceThresholds};
use vision_core::interfaces::Frame;
use vision_core::overlay::{draw_rect, normalize_box};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run detector on a single image and emit a boxed PNG"
)]
struct Args {
    /// Input image path (any format supported by the `image` crate).
    #[arg(long)]
    image: PathBuf,
    /// Output path for the boxed image (defaults to <stem>_boxed.png alongside the input).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Objectness threshold.
    #[arg(long, default_value_t = 0.3)]
    infer_obj_thresh: f32,
    /// IoU threshold for NMS.
    #[arg(long, default_value_t = 0.5)]
    infer_iou_thresh: f32,
}

fn default_out_path(input: &Path) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}_boxed.png"))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let in_path = &args.image;
    if !in_path.exists() {
        anyhow::bail!("input image not found: {}", in_path.display());
    }
    let out_path = args.out.unwrap_or_else(|| default_out_path(in_path));

    let img = ImageReader::open(in_path)?.decode()?.to_rgba8();
    let (w, h) = img.dimensions();
    let rgba = img.as_raw().clone();

    let thresh_opts = ThresholdOpts::new(args.infer_obj_thresh, args.infer_iou_thresh);
    let thresh = InferenceThresholds {
        obj_thresh: thresh_opts.obj_thresh,
        iou_thresh: thresh_opts.iou_thresh,
    };
    let factory = InferenceFactory;
    let mut detector = factory.build(thresh, None);

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let frame = Frame {
        id: 0,
        timestamp: ts,
        rgba: Some(rgba),
        size: (w, h),
        path: Some(in_path.clone()),
    };

    let result = detector.detect(&frame);
    let mut boxed = img.clone();
    if result.boxes.is_empty() {
        eprintln!("no detections (confidence {})", result.confidence);
    } else {
        for (i, bbox) in result.boxes.iter().enumerate() {
            let color = if i == 0 {
                image::Rgba([255, 64, 192, 255])
            } else {
                image::Rgba([64, 192, 255, 255])
            };
            if let Some(px_box) = normalize_box(*bbox, (w, h)) {
                draw_rect(&mut boxed, px_box, color, 2);
            }
        }
    }
    boxed.save(&out_path)?;
    println!(
        "saved boxed image to {} ({} boxes)",
        out_path.display(),
        result.boxes.len()
    );
    Ok(())
}
