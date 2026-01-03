use data_contracts::capture::{CaptureMetadata, PolypLabel};
use image::Rgba;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use vision_core::prelude::{FrameRecord, Label, Recorder};

/// Default file-based recorder: writes frame metadata/labels to `run_dir/labels/frame_XXXXX.json`.
pub struct JsonRecorder {
    pub run_dir: PathBuf,
}

impl JsonRecorder {
    pub fn new(run_dir: impl Into<PathBuf>) -> Self {
        Self {
            run_dir: run_dir.into(),
        }
    }
}

impl Recorder for JsonRecorder {
    fn record(&mut self, record: &FrameRecord) -> std::io::Result<()> {
        let labels_dir = self.run_dir.join("labels");
        fs::create_dir_all(&labels_dir)?;
        let image = record
            .frame
            .path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("frame_{:05}.png", record.frame.id));
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let meta = CaptureMetadata {
            frame_id: record.frame.id,
            sim_time: record.frame.timestamp,
            unix_time,
            image,
            image_present: record.frame.path.is_some(),
            camera_active: record.camera_active,
            polyp_seed: record.polyp_seed,
            polyp_labels: record.labels.iter().map(label_to_polyp).collect(),
        };
        meta.validate()
            .map_err(|e| std::io::Error::other(format!("validation failed: {e}")))?;
        let out = labels_dir.join(format!("frame_{:05}.json", record.frame.id));
        let mut writer = BufWriter::new(fs::File::create(out)?);
        serde_json::to_writer_pretty(&mut writer, &meta)?;
        writer.write_all(b"\n")?;
        Ok(())
    }
}

fn label_to_polyp(label: &Label) -> PolypLabel {
    PolypLabel {
        center_world: label.center_world,
        bbox_px: label.bbox_px,
        bbox_norm: label.bbox_norm,
    }
}

/// Generate overlay PNGs from label JSONs in a run directory.
pub fn generate_overlays(run_dir: &Path) -> anyhow::Result<()> {
    let labels_dir = run_dir.join("labels");
    let out_dir = run_dir.join("overlays");
    fs::create_dir_all(&out_dir)?;

    for entry in fs::read_dir(&labels_dir).into_iter().flatten() {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(meta) = serde_json::from_slice::<CaptureMetadata>(&bytes) else {
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
        let (w, h) = img.dimensions();
        let clamp =
            |v: f32, max: u32| -> u32 { v.max(0.0).min((max.saturating_sub(1)) as f32) as u32 };
        for label in meta.polyp_labels.iter().filter_map(|l| l.bbox_px) {
            let bbox_px = [
                clamp(label[0], w),
                clamp(label[1], h),
                clamp(label[2], w),
                clamp(label[3], h),
            ];
            draw_rect(&mut img, bbox_px, Rgba([255, 64, 192, 255]), 2);
        }
        let filename = Path::new(&meta.image)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(meta.image);
        let _ = img.save(out_dir.join(filename));
    }
    Ok(())
}

/// Prune a run directory into a destination root, copying kept artifacts.
pub fn prune_run(input_run: &Path, output_root: &Path) -> std::io::Result<(usize, usize)> {
    let run_name = input_run
        .file_name()
        .ok_or_else(|| std::io::Error::other("invalid run dir"))?;
    let out_run = output_root.join(run_name);
    fs::create_dir_all(out_run.join("labels"))?;
    fs::create_dir_all(out_run.join("images"))?;
    fs::create_dir_all(out_run.join("overlays"))?;

    let manifest_in = input_run.join("run_manifest.json");
    if manifest_in.exists() {
        let _ = fs::copy(&manifest_in, out_run.join("run_manifest.json"));
    }

    for entry in fs::read_dir(input_run)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let Some(fname) = name.to_str() else { continue };
        if fname == "run_manifest.json" {
            continue;
        }
        if path.is_dir() {
            if let Some(dir_name) = path.file_name() {
                let out_dir = out_run.join(dir_name);
                fs::create_dir_all(&out_dir)?;
                for f in fs::read_dir(&path)? {
                    let f = f?;
                    let out_f = out_dir.join(f.file_name());
                    let _ = fs::copy(f.path(), out_f);
                }
            }
            continue;
        }
    }

    let kept = fs::read_dir(out_run.join("labels"))
        .into_iter()
        .flatten()
        .count();
    let skipped = fs::read_dir(input_run.join("labels"))
        .into_iter()
        .flatten()
        .count()
        .saturating_sub(kept);
    Ok((kept, skipped))
}

fn draw_rect(img: &mut image::RgbaImage, bbox: [u32; 4], color: Rgba<u8>, thickness: u32) {
    let (x0, y0, x1, y1) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    for t in 0..thickness {
        for x in x0.saturating_sub(t)..=x1.saturating_add(t) {
            if y0 >= t {
                if let Some(p) = img.get_pixel_mut_checked(x, y0 - t) {
                    *p = color;
                }
            }
            if let Some(p) = img.get_pixel_mut_checked(x, y1 + t) {
                *p = color;
            }
        }
        for y in y0.saturating_sub(t)..=y1.saturating_add(t) {
            if x0 >= t {
                if let Some(p) = img.get_pixel_mut_checked(x0 - t, y) {
                    *p = color;
                }
            }
            if let Some(p) = img.get_pixel_mut_checked(x1 + t, y) {
                *p = color;
            }
        }
    }
}

#[allow(dead_code)]
trait GetPixelChecked {
    fn get_pixel_mut_checked(&mut self, x: u32, y: u32) -> Option<&mut Rgba<u8>>;
}

impl GetPixelChecked for image::RgbaImage {
    fn get_pixel_mut_checked(&mut self, x: u32, y: u32) -> Option<&mut Rgba<u8>> {
        if x < self.width() && y < self.height() {
            Some(self.get_pixel_mut(x, y))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_core::prelude::{Frame, FrameRecord};

    #[test]
    fn json_recorder_writes_label() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut recorder = JsonRecorder::new(run_dir);
        let frame = Frame {
            id: 1,
            timestamp: 0.5,
            rgba: None,
            size: (640, 480),
            path: Some(PathBuf::from("images/frame_00001.png")),
        };
        let record = FrameRecord {
            frame,
            labels: &[],
            camera_active: true,
            polyp_seed: 42,
        };
        recorder.record(&record).expect("write label");
        let label_path = run_dir.join("labels/frame_00001.json");
        assert!(label_path.exists(), "label file should be written");
    }

    #[test]
    fn prune_run_copies_manifest_and_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("run_123");
        let labels = input.join("labels");
        let images = input.join("images");
        let overlays = input.join("overlays");
        fs::create_dir_all(&labels).unwrap();
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&overlays).unwrap();
        fs::write(input.join("run_manifest.json"), "{}").unwrap();
        fs::write(labels.join("frame_00001.json"), "{}").unwrap();
        fs::write(images.join("frame_00001.png"), []).unwrap();
        fs::write(overlays.join("frame_00001.png"), []).unwrap();

        let out_root = dir.path().join("out");
        let (kept, _skipped) = prune_run(&input, &out_root).expect("prune run");
        assert_eq!(kept, 1);
        assert!(out_root.join("run_123/run_manifest.json").exists());
        assert!(out_root.join("run_123/labels/frame_00001.json").exists());
        assert!(out_root.join("run_123/images/frame_00001.png").exists());
        assert!(out_root.join("run_123/overlays/frame_00001.png").exists());
    }
}
