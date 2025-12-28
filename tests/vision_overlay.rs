use image::Rgba;

use colon_sim::vision::overlay::{draw_rect, normalize_box};

#[test]
fn normalize_and_draw_rect() {
    let dims = (10, 8);
    let mut img = image::RgbaImage::from_pixel(dims.0, dims.1, Rgba([0, 0, 0, 0]));
    let bbox_norm = [0.1, 0.25, 0.6, 0.75];
    let px = normalize_box(bbox_norm, dims).expect("normalize ok");
    draw_rect(&mut img, px, Rgba([255, 0, 0, 255]), 1);

    // Corners should be red; inside untouched.
    assert_eq!(img.get_pixel(px[0], px[1]), &Rgba([255, 0, 0, 255]));
    assert_eq!(img.get_pixel(px[2], px[3]), &Rgba([255, 0, 0, 255]));
    assert_eq!(img.get_pixel(px[0] + 1, px[1] + 1), &Rgba([0, 0, 0, 0]));
}
