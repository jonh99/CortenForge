use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use training::{BigDet, BigDetConfig};

// Force a CPU backend for this shape check to avoid requiring a GPU even when backend-wgpu is enabled.
type ADBackend = Autodiff<NdArray<f32>>;

#[test]
fn forward_shapes_bigdet() {
    let device = <ADBackend as burn::tensor::backend::Backend>::Device::default();
    let max_boxes = 5;
    let input_dim = 4 + 8; // match collate features shape

    let model = BigDet::<ADBackend>::new(
        BigDetConfig {
            max_boxes,
            input_dim: Some(input_dim),
            ..Default::default()
        },
        &device,
    );

    let batch = 2;
    let input = Tensor::<ADBackend, 2>::zeros([batch, input_dim], &device);
    let (boxes, scores) = model.forward_multibox(input);

    assert_eq!(boxes.dims(), [batch, max_boxes, 4]);
    assert_eq!(scores.dims(), [batch, max_boxes]);

    // Ensure outputs are in [0,1] due to sigmoid + clamp logic.
    let boxes_min = boxes.clone().min();
    let boxes_max = boxes.clone().max();
    let scores_min = scores.clone().min();
    let scores_max = scores.clone().max();

    let bmin: f32 = boxes_min.into_data().to_vec::<f32>().unwrap_or_default()[0];
    let bmax: f32 = boxes_max.into_data().to_vec::<f32>().unwrap_or_default()[0];
    let smin: f32 = scores_min.into_data().to_vec::<f32>().unwrap_or_default()[0];
    let smax: f32 = scores_max.into_data().to_vec::<f32>().unwrap_or_default()[0];

    assert!(bmin >= 0.0 - 1e-6 && bmax <= 1.0 + 1e-6);
    assert!(smin >= 0.0 - 1e-6 && smax <= 1.0 + 1e-6);
}
