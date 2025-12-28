use image::{Rgba, RgbaImage};

/// Normalize a box from 0..1 space into pixel coordinates, clamped to image bounds.
pub fn normalize_box(bbox_norm: [f32; 4], dims: (u32, u32)) -> Option<[u32; 4]> {
    let (w, h) = dims;
    let clamp = |v: f32, max: u32| -> u32 { v.max(0.0).min((max as i32 - 1) as f32) as u32 };
    let x0 = clamp(bbox_norm[0] * w as f32, w);
    let y0 = clamp(bbox_norm[1] * h as f32, h);
    let x1 = clamp(bbox_norm[2] * w as f32, w);
    let y1 = clamp(bbox_norm[3] * h as f32, h);
    if x0 >= w || y0 >= h || x1 >= w || y1 >= h || x0 > x1 || y0 > y1 {
        return None;
    }
    Some([x0, y0, x1, y1])
}

/// Draw a rectangle border with given thickness.
pub fn draw_rect(img: &mut RgbaImage, bbox_px: [u32; 4], color: Rgba<u8>, thickness: u32) {
    let (w, h) = img.dimensions();
    let [x0, y0, x1, y1] = bbox_px;
    for t in 0..thickness {
        let xx0 = x0.saturating_add(t);
        let yy0 = y0.saturating_add(t);
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
