use bevy::prelude::*;
use bevy::color::Mix;
use bevy::math::primitives::Cylinder;
use bevy_rapier3d::prelude::*;

use crate::probe::ProbeHead;

#[derive(Component)]
pub struct TunnelRing {
    pub base_radius: f32,
    pub expanded_radius: f32,
    pub current_radius: f32,
    pub half_height: f32,
}

// Simple lerp helper for smooth transitions.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn setup_tunnel(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Align scales with existing probe (capsule radius ~0.8). Base radius ~0.9; expanded ~1.3.
    let num_rings = 140;
    let ring_spacing = 0.3;
    let start_z = -5.0;
    let base_radius = 0.9;
    let expanded_radius = 1.3;
    let half_height = 0.15;

    let ring_mesh = meshes.add(Mesh::from(Cylinder {
        radius: base_radius,
        half_height,
    }));

    let base_color = Color::srgb(0.7, 0.4, 0.4);
    let mat = materials.add(StandardMaterial {
        base_color,
        perceptual_roughness: 0.65,
        metallic: 0.02,
        ..default()
    });

    let wall_friction = 1.0;

    for i in 0..num_rings {
        let z = start_z + i as f32 * ring_spacing;

        commands.spawn((
            TunnelRing {
                base_radius,
                expanded_radius,
                current_radius: base_radius,
                half_height,
            },
            Mesh3d(ring_mesh.clone()),
            MeshMaterial3d(mat.clone()),
            Transform {
                translation: Vec3::new(0.0, 0.0, z),
                rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2), // orient cylinder axis to Z
                scale: Vec3::ONE,
                ..default()
            },
            Collider::cylinder(half_height, base_radius),
            Friction {
                coefficient: wall_friction,
                combine_rule: CoefficientCombineRule::Average,
                ..default()
            },
            RigidBody::Fixed,
            GlobalTransform::default(),
        ));
    }
}

pub fn tunnel_expansion_system(
    time: Res<Time>,
    probe_q: Query<&GlobalTransform, With<ProbeHead>>,
    mut rings_q: Query<(
        Entity,
        &mut TunnelRing,
        &mut Transform,
        &mut Friction,
        &MeshMaterial3d<StandardMaterial>,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let Ok(probe_tf) = probe_q.single() else {
        return;
    };
    let probe_z = probe_tf.translation().z;

    let strong_expand_radius = 2.5;
    let soft_expand_radius = 5.0;
    let expand_speed = 6.5;

    let base_color = Color::srgb(0.7, 0.4, 0.4);
    let expanded_color = Color::srgb(1.0, 0.7, 0.7);
    let high_friction = 1.2;
    let low_friction = 0.4;

    for (entity, mut ring, mut tf, mut friction, mat_handle) in rings_q.iter_mut() {
        let ring_z = tf.translation.z;
        let dz = (ring_z - probe_z).abs();

        let target_radius = if dz < strong_expand_radius {
            ring.expanded_radius
        } else if dz < soft_expand_radius {
            let t = (dz - strong_expand_radius) / (soft_expand_radius - strong_expand_radius);
            lerp(ring.expanded_radius, ring.base_radius, t)
        } else {
            ring.base_radius
        };

        let alpha = 1.0 - f32::exp(-expand_speed * time.delta_secs());
        ring.current_radius = lerp(ring.current_radius, target_radius, alpha);

        let scale_xy = ring.current_radius / ring.base_radius;
        tf.scale = Vec3::new(scale_xy, scale_xy, 1.0);

        commands
            .entity(entity)
            .insert(Collider::cylinder(ring.half_height, ring.current_radius));

        let expansion_factor = ((ring.current_radius - ring.base_radius)
            / (ring.expanded_radius - ring.base_radius))
            .clamp(0.0, 1.0);

        friction.coefficient = lerp(high_friction, low_friction, expansion_factor);

        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.base_color = base_color.mix(&expanded_color, expansion_factor);
        }
    }
}
