use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use image::Rgba;
use serde::Deserialize;

use colon_sim::vision::overlay::{draw_rect, normalize_box};

#[derive(Deserialize)]
struct PolypLabel {
    #[allow(dead_code)]
    center_world: [f32; 3],
    bbox_px: Option<[f32; 4]>,
    #[allow(dead_code)]
    bbox_norm: Option<[f32; 4]>,
}

#[derive(Deserialize)]
struct CaptureMetadata {
    image: String,
    image_present: bool,
    #[allow(dead_code)]
    polyp_seed: Option<u64>,
    polyp_labels: Vec<PolypLabel>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let run_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/datasets/captures"));
    let out_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("overlays"));
    let labels_dir = run_dir.join("labels");

    fs::create_dir_all(&out_dir)?;

    for entry in fs::read_dir(&labels_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let meta: CaptureMetadata = serde_json::from_slice(&fs::read(&path)?)?;
        if !meta.image_present {
            continue;
        }
        let img_path = run_dir.join(&meta.image);
        if !img_path.exists() {
            eprintln!("missing image for {:?}", path.file_name());
            continue;
        }
        let mut img = image::open(&img_path)?.into_rgba8();
        for label in meta.polyp_labels.iter().filter_map(|l| l.bbox_px) {
            let color = Rgba([255, 64, 192, 255]);
            let dims = img.dimensions();
            if let Some(px_box) = normalize_box(
                [
                    label[0] / dims.0 as f32,
                    label[1] / dims.1 as f32,
                    label[2] / dims.0 as f32,
                    label[3] / dims.1 as f32,
                ],
                dims,
            ) {
                draw_rect(&mut img, px_box, color, 2);
            }
        }
        let filename = Path::new(&meta.image)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(meta.image);
        let out_path = out_dir.join(filename);
        img.save(out_path)?;
    }

    println!("Overlays written to {}", out_dir.display());
    Ok(())
}
