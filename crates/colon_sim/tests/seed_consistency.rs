use colon_sim_app::polyp::PolypRandom;
use rand::Rng;

#[test]
fn polyp_rng_is_deterministic_for_same_seed() {
    let seed = 42;
    let mut a = PolypRandom::new(seed);
    let mut b = PolypRandom::new(seed);

    let sample =
        |rng: &mut PolypRandom| -> Vec<f32> { (0..8).map(|_| rng.rng().r#gen::<f32>()).collect() };

    let va = sample(&mut a);
    let vb = sample(&mut b);
    assert_eq!(va, vb, "same seed should yield identical sequences");
}
