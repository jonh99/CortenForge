use image::{Rgba, RgbaImage};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct PolypLabel {
    bbox_px: Option<[f32; 4]>,
}

#[derive(Deserialize)]
struct CaptureMetadata {
    image: String,
    image_present: bool,
    polyp_labels: Vec<PolypLabel>,
}

pub fn generate_overlays(run_dir: &Path, labels_subdir: &str, overlays_subdir: &str) {
    let labels_dir = run_dir.join(labels_subdir);
    let out_dir = run_dir.join(overlays_subdir);
    if fs::create_dir_all(&out_dir).is_err() {
        return;
    }

    for entry in fs::read_dir(&labels_dir).into_iter().flatten() {
        let Ok(path) = entry.map(|e| e.path()) else {
            continue;
        };
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(meta) = fs::read(&path).and_then(|bytes| {
            serde_json::from_slice::<CaptureMetadata>(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }) else {
            continue;
        };
        if !meta.image_present {
            continue;
        }
        let img_path = run_dir.join(&meta.image);
        if !img_path.exists() {
            continue;
        }
        let Ok(mut img) = image::open(&img_path).map(|im| im.into_rgba8()) else {
            continue;
        };
        for label in meta.polyp_labels.iter().filter_map(|l| l.bbox_px) {
            draw_rect(&mut img, label, Rgba([255, 64, 192, 255]), 2);
        }
        let filename = Path::new(&meta.image)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(meta.image);
        let _ = img.save(out_dir.join(filename));
    }
}

fn draw_rect(img: &mut RgbaImage, bbox: [f32; 4], color: Rgba<u8>, thickness: u32) {
    let (w, h) = img.dimensions();
    let clamp = |v: f32, max: u32| -> u32 { v.max(0.0).min((max as i32 - 1) as f32) as u32 };
    let x0 = clamp(bbox[0], w);
    let y0 = clamp(bbox[1], h);
    let x1 = clamp(bbox[2], w);
    let y1 = clamp(bbox[3], h);
    if x0 >= w || y0 >= h || x1 >= w || y1 >= h {
        return;
    }
    for t in 0..thickness {
        let xx0 = x0 + t;
        let yy0 = y0 + t;
        let xx1 = x1.saturating_sub(t);
        let yy1 = y1.saturating_sub(t);
        if xx0 >= w || yy0 >= h || xx1 >= w || yy1 >= h || xx0 > xx1 || yy0 > yy1 {
            continue;
        }
        for x in xx0..=xx1 {
            if yy0 < h {
                img.put_pixel(x, yy0, color);
            }
            if yy1 < h {
                img.put_pixel(x, yy1, color);
            }
        }
        for y in yy0..=yy1 {
            if xx0 < w {
                img.put_pixel(xx0, y, color);
            }
            if xx1 < w {
                img.put_pixel(xx1, y, color);
            }
        }
    }
}
