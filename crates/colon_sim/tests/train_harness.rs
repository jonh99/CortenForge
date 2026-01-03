#![cfg(feature = "burn_runtime")]

// Simple smoke test to ensure TinyDet compiles and produces sane outputs with the current API.
use burn::backend::{autodiff::Autodiff, ndarray::NdArray};
use burn::tensor::Tensor;
use models::{TinyDet, TinyDetConfig};

type Backend = NdArray<f32>;
type ADBackend = Autodiff<Backend>;

#[test]
fn tiny_det_forward_shape() {
    let device = <ADBackend as burn::tensor::backend::Backend>::Device::default();
    let model = TinyDet::<ADBackend>::new(TinyDetConfig::default(), &device);

    // Input: B x 4 (single box features).
    let input = Tensor::<ADBackend, 2>::zeros([3, 4], &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [3, 1]);
}
