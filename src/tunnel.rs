use bevy::color::Mix;
use bevy::math::primitives::Cylinder;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use smallvec::{smallvec, SmallVec};

use crate::probe::ProbeHead;
use crate::balloon_control::BalloonControl;

pub const TUNNEL_START_Z: f32 = -20.0;
pub const TUNNEL_LENGTH: f32 = 108.0;
pub const TUNNEL_BEND_AMPLITUDE: f32 = 7.0;

#[derive(Component)]
pub struct TunnelRing {
    pub base_radius: f32,
    pub contracted_radius: f32,
    pub current_radius: f32,
    pub half_height: f32,
}

#[derive(Component)]
pub struct TunnelRingVisual;

// Simple lerp helper for smooth transitions.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Centerline of the tunnel: a gentle S-curve in X over the full Z span.
pub fn tunnel_centerline(z: f32) -> (Vec3, Vec3) {
    let progress = ((z - TUNNEL_START_Z) / TUNNEL_LENGTH).clamp(0.0, 1.0);
    let phase = progress * std::f32::consts::PI;
    let x = TUNNEL_BEND_AMPLITUDE * phase.sin();
    let dx_dz = if TUNNEL_LENGTH > 0.0 {
        TUNNEL_BEND_AMPLITUDE * std::f32::consts::PI / TUNNEL_LENGTH * phase.cos()
    } else {
        0.0
    };
    let tangent = Vec3::new(dx_dz, 0.0, 1.0).normalize_or_zero();
    (Vec3::new(x, 0.0, z), tangent)
}

/// Marches along the tunnel centerline by an arc length, returning the resulting position, tangent, and parameter z.
pub fn advance_centerline(z_start: f32, arc_distance: f32) -> (Vec3, Vec3, f32) {
    let mut remaining = arc_distance.max(0.0);
    let mut z = z_start;
    let max_z = TUNNEL_START_Z + TUNNEL_LENGTH;

    // Step in small arc segments to approximate the curve without heavy math.
    let mut steps = 0;
    while remaining > 1e-4 && steps < 512 {
        steps += 1;
        let progress = ((z - TUNNEL_START_Z) / TUNNEL_LENGTH).clamp(0.0, 1.0);
        let phase = progress * std::f32::consts::PI;
        let dx_dz = if TUNNEL_LENGTH > 0.0 {
            TUNNEL_BEND_AMPLITUDE * std::f32::consts::PI / TUNNEL_LENGTH * phase.cos()
        } else {
            0.0
        };
        let tang_mag = (1.0 + dx_dz * dx_dz).sqrt().max(1e-5);
        let dz = (remaining / tang_mag).min(0.35); // cap step for stability

        // Stop if we hit the bounds.
        let next_z = (z + dz).clamp(TUNNEL_START_Z, max_z);
        let actual_dz = next_z - z;
        if actual_dz.abs() < 1e-6 {
            break;
        }

        remaining -= tang_mag * actual_dz;
        z = next_z;
    }

    let (pos, tangent) = tunnel_centerline(z);
    (pos, tangent, z)
}

pub fn tunnel_tangent_rotation(tangent: Vec3) -> Quat {
    let forward = tangent.normalize_or_zero();
    if forward.length_squared() == 0.0 {
        return Quat::IDENTITY;
    }

    let mut up = Vec3::Y;
    if forward.cross(up).length_squared() < 1e-4 {
        up = Vec3::Z;
    }

    let right = forward.cross(up).normalize_or_zero();
    let corrected_up = right.cross(forward).normalize_or_zero();

    if right.length_squared() == 0.0 || corrected_up.length_squared() == 0.0 {
        Quat::IDENTITY
    } else {
        Quat::from_mat3(&Mat3::from_cols(right, corrected_up, forward))
    }
}

fn ring_shell_collider(radius: f32, half_height: f32) -> Collider {
    let wall_thickness = 0.08;
    let segments = 16;
    let angle_step = std::f32::consts::TAU / segments as f32;
    let wall_half = wall_thickness * 0.5;
    let tangent_half = radius * (angle_step * 0.5).tan() + wall_half;

    let mut shapes = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = i as f32 * angle_step;
        let dir = Vec2::new(angle.cos(), angle.sin());
        let center = Vec3::new(dir.x * radius, dir.y * radius, 0.0);
        let rot = Quat::from_rotation_z(angle);
        shapes.push((center, rot, Collider::cuboid(wall_half, tangent_half, half_height)));
    }

    Collider::compound(shapes)
}

pub fn setup_tunnel(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Align scales with existing probe (capsule radius ~0.8). Base radius stays roomy; contraction squeezes tighter.
    let num_rings = 360;
    let ring_spacing = TUNNEL_LENGTH / (num_rings - 1) as f32;
    let base_radius = 1.2;
    let contracted_radius = 0.9;
    let half_height = 0.15;

    let ring_mesh = meshes.add(Mesh::from(Cylinder {
        radius: base_radius,
        half_height,
    }));

    let base_color = Color::srgba(0.25, 0.22, 0.18, 0.28);
    let mat = materials.add(StandardMaterial {
        base_color,
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.65,
        metallic: 0.02,
        ..default()
    });

    let wall_friction = 1.0;

    for i in 0..num_rings {
        let z = TUNNEL_START_Z + i as f32 * ring_spacing;
        let (center, tangent) = tunnel_centerline(z);
        let ring_rotation = tunnel_tangent_rotation(tangent);

        commands.spawn((
            TunnelRing {
                base_radius,
                contracted_radius,
                current_radius: base_radius,
                half_height,
            },
            Transform {
                translation: center,
                rotation: ring_rotation,
                ..default()
            },
            GlobalTransform::default(),
            Visibility::default(),
            ring_shell_collider(base_radius, half_height),
            Friction {
                coefficient: wall_friction,
                combine_rule: CoefficientCombineRule::Average,
                ..default()
            },
            RigidBody::Fixed,
        ))
        .with_children(|child| {
            child.spawn((
                TunnelRingVisual,
                Mesh3d(ring_mesh.clone()),
                MeshMaterial3d(mat.clone()),
                Transform {
                    rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    ..default()
                },
                GlobalTransform::default(),
                Visibility::default(),
            ));
        });
    }
}

pub fn tunnel_expansion_system(
    time: Res<Time>,
    balloon: Res<BalloonControl>,
    probe_q: Query<&GlobalTransform, With<ProbeHead>>,
    mut rings_q: Query<(
        &mut TunnelRing,
        &GlobalTransform,
        &Children,
        &mut Collider,
        &mut Friction,
    )>,
    mut visuals: Query<(&mut Transform, &MeshMaterial3d<StandardMaterial>), With<TunnelRingVisual>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Ok(probe_tf) = probe_q.single() else {
        return;
    };
    let _probe_z = probe_tf.translation().z;

    let mut contraction_points: SmallVec<[f32; 2]> = smallvec![];
    if balloon.head_inflated {
        contraction_points.push(balloon.position.z);
    }
    if balloon.tail_inflated {
        contraction_points.push(balloon.rear_position.z);
    }

    // Keep the squeeze highly localized around each capsule end.
    let strong_contract_radius = 1.2;
    let soft_contract_radius = 2.4;
    let contract_speed = 6.5;

    let base_color = Color::srgba(0.25, 0.22, 0.18, 0.28);
    let balloon_color = Color::srgba(1.0, 0.85, 0.35, 0.6);
    let relaxed_friction = 1.2;
    let contracted_friction = 1.8;

    let falloff = |dz: f32| {
        if dz < strong_contract_radius {
            1.0
        } else if dz < soft_contract_radius {
            1.0 - (dz - strong_contract_radius) / (soft_contract_radius - strong_contract_radius)
        } else {
            0.0
        }
    };

    for (mut ring, tf, children, mut collider, mut friction) in rings_q.iter_mut() {
        let ring_z = tf.translation().z;
        let balloon_factor = contraction_points
            .iter()
            .map(|&bz| falloff((ring_z - bz).abs()))
            .fold(0.0, f32::max);

        let target_radius = if balloon_factor > 0.0 {
            lerp(
                ring.base_radius,
                ring.contracted_radius,
                balloon_factor.clamp(0.0, 1.0),
            )
        } else {
            ring.base_radius
        };

        let alpha = 1.0 - f32::exp(-contract_speed * time.delta_secs());
        ring.current_radius = lerp(ring.current_radius, target_radius, alpha);

        let scale_xy = ring.current_radius / ring.base_radius;
        for child in children.iter() {
            if let Ok((mut v_tf, v_mat)) = visuals.get_mut(child) {
                v_tf.scale = Vec3::new(scale_xy, scale_xy, 1.0);

                let contraction_factor = ((ring.base_radius - ring.current_radius)
                    / (ring.base_radius - ring.contracted_radius))
                    .clamp(0.0, 1.0);

                if let Some(mat) = materials.get_mut(&v_mat.0) {
                    // Balloon-induced contraction (green gradient highlights active squeeze).
                    mat.base_color = base_color.mix(&balloon_color, contraction_factor);
                }
            }
        }

        *collider = ring_shell_collider(ring.current_radius, ring.half_height);

        let contraction_factor = ((ring.base_radius - ring.current_radius)
            / (ring.base_radius - ring.contracted_radius))
            .clamp(0.0, 1.0);
        friction.coefficient = lerp(relaxed_friction, contracted_friction, contraction_factor);
    }
}
