use bevy::math::primitives::Cylinder;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::f32::consts::FRAC_PI_2;

use crate::controls::ControlParams;
use crate::balloon_control::BalloonControl;

const MIN_STRETCH: f32 = 1.0;
// Allow stretching up to +72% of the deflated length.
const MAX_STRETCH: f32 = 1.72;
const STRETCH_RATE: f32 = 0.6; // slower extension
const RETRACT_RATE: f32 = 0.9;

#[derive(Resource, Default)]
pub struct StretchState {
    pub factor: f32,
}

#[derive(Component)]
pub struct CapsuleProbe;

#[derive(Component)]
pub struct ProbeTip;

#[derive(Component)]
pub struct ProbeHead;

#[derive(Component)]
pub struct ProbeSegment;

#[derive(Component)]
pub struct SegmentSpring {
    pub base_rest: f32,
}

#[derive(Component)]
pub struct SegmentIndex(pub usize);

#[derive(Component)]
pub struct ProbeBody {
    pub base_radius: f32,
    pub base_length: f32,
    pub ring_count: usize,
}

#[derive(Component)]
pub struct ProbeRing {
    pub index: usize,
}

#[derive(Component)]
pub struct ProbeVisual;

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

pub fn spawn_probe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    control: Res<ControlParams>,
) {
    // Elastic probe tube built from ring colliders (like the tunnel) driven by stretch.
    let base_radius = 0.8;
    let base_length = 20.0;
    let ring_count = 24usize;
    let ring_spacing = base_length / ring_count as f32;
    let ring_half_height = ring_spacing * 0.45;
    // Keep similar placement to previous chain: tail back near -16, tip ahead in the straight.
    let tail_z = -10.0;

    let visual_mesh = meshes.add(Mesh::from(Cylinder {
        radius: base_radius * 0.9,
        half_height: base_length * 0.5,
    }));
    let material_handle = materials.add(Color::srgb(0.8, 0.2, 0.2));

    let mut root = commands.spawn((
        ProbeBody {
            base_radius,
            base_length,
            ring_count,
        },
        CapsuleProbe,
        RigidBody::KinematicPositionBased,
        Collider::ball(base_radius),
        Friction {
            coefficient: control.friction,
            combine_rule: CoefficientCombineRule::Average,
            ..default()
        },
        CollisionGroups::new(Group::GROUP_1, Group::ALL),
        Transform::from_translation(Vec3::new(0.0, 0.0, tail_z)),
        GlobalTransform::default(),
        Visibility::default(),
    ));

    root.with_children(|child| {
        // Head marker and collider.
        child.spawn((
            ProbeHead,
            ProbeTip,
            Collider::ball(base_radius * 0.9),
            Friction {
                coefficient: control.friction,
                combine_rule: CoefficientCombineRule::Average,
                ..default()
            },
            CollisionGroups::new(Group::GROUP_1, Group::ALL),
            Transform::from_translation(Vec3::new(0.0, 0.0, base_length)),
            GlobalTransform::default(),
        ));

        // Body rings for collision hull.
        for i in 0..=ring_count {
            let z = i as f32 * ring_spacing;
            child.spawn((
                ProbeRing { index: i },
                ring_shell_collider(base_radius, ring_half_height),
                Friction {
                    coefficient: control.friction,
                    combine_rule: CoefficientCombineRule::Average,
                    ..default()
                },
                CollisionGroups::new(Group::GROUP_1, Group::ALL),
                Transform::from_translation(Vec3::new(0.0, 0.0, z)),
                GlobalTransform::default(),
            ));
        }

        // Visual skin.
        child.spawn((
            ProbeVisual,
            Mesh3d(visual_mesh),
            MeshMaterial3d(material_handle),
            Transform {
                translation: Vec3::new(0.0, 0.0, base_length * 0.5),
                rotation: Quat::from_rotation_x(FRAC_PI_2),
                scale: Vec3::ONE,
            },
            GlobalTransform::default(),
            Visibility::default(),
        ));
    });
}

pub fn peristaltic_drive(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    control: Res<ControlParams>,
    balloon: Res<BalloonControl>,
    mut stretch: ResMut<StretchState>,
    mut tail_body: Query<(&ProbeBody, Entity, &mut RigidBody), With<CapsuleProbe>>,
    mut transforms: ParamSet<(
        Query<&mut Transform, (With<ProbeHead>, Without<ProbeVisual>)>,
        Query<&mut Transform, With<ProbeVisual>>,
        Query<(&ProbeRing, &mut Transform, &mut Collider, &mut Friction)>,
        Query<&mut Friction, With<CapsuleProbe>>,
        Query<&mut Transform, With<CapsuleProbe>>,
    )>,
) {
    let Ok((body, tail_entity, mut body_rb)) = tail_body.single_mut() else {
        return;
    };

    let mut tail_tf_query = transforms.p4();
    let Ok(mut body_tf) = tail_tf_query.get_mut(tail_entity) else {
        return;
    };

    // Capture current head world position before we change length so head-anchored deflate pulls the tail forward.
    let current_length = body.base_length * stretch.factor.max(MIN_STRETCH);
    let head_anchor_z = if balloon.head_inflated {
        Some(body_tf.translation.z + current_length)
    } else {
        None
    };

    // Manual extension/retraction:
    // - Extend: tail balloon on, head balloon off, hold Up/I.
    // - Retract/deflate: Down/K always shrinks length, regardless of anchor state.
    // - Head balloon on locks length from auto change, but manual deflate still works.
    let extend_command = balloon.tail_inflated
        && !balloon.head_inflated
        && (keys.pressed(KeyCode::ArrowUp) || keys.pressed(KeyCode::KeyI));
    let retract_command = keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyK);

    let dt = time.delta_secs();
    if extend_command {
        stretch.factor = (stretch.factor + STRETCH_RATE * dt).min(MAX_STRETCH);
    } else if retract_command {
        stretch.factor = (stretch.factor - RETRACT_RATE * dt).max(MIN_STRETCH);
    }

    // Anchor tail rigidly when tail balloon is on; release when off.
    *body_rb = if balloon.tail_inflated {
        RigidBody::KinematicPositionBased
    } else {
        RigidBody::Dynamic
    };

    let length = body.base_length * stretch.factor.max(MIN_STRETCH);

    // If the head is anchored, slide the tail forward/back to keep head world position fixed.
    if let Some(anchor_z) = head_anchor_z {
        body_tf.translation.z = anchor_z - length;
    }

    let spacing = length / body.ring_count as f32;
    let ring_half_height = spacing * 0.45;

    // Update head transform to new tip position.
    if let Ok(mut head_tf) = transforms.p0().single_mut() {
        head_tf.translation = Vec3::new(0.0, 0.0, length);
    }

    // Update visual skin.
    if let Ok(mut vis_tf) = transforms.p1().single_mut() {
        vis_tf.translation = Vec3::new(0.0, 0.0, length * 0.5);
        vis_tf.scale = Vec3::new(1.0, stretch.factor, 1.0);
    }

    // Tail friction spikes when anchored; head friction spikes when head balloon is on.
    if let Ok(mut tail_fric) = transforms.p3().single_mut() {
        let tail_anchor = if balloon.tail_inflated { 8.0 } else { 1.0 };
        tail_fric.coefficient = control.friction * tail_anchor;
    }

    // Update rings to cover the stretched length and adjust friction gradient.
    for (ring, mut tf, mut collider, mut fric) in transforms.p2().iter_mut() {
        tf.translation = Vec3::new(0.0, 0.0, ring.index as f32 * spacing);
        *collider = ring_shell_collider(body.base_radius, ring_half_height);

        let t = ring.index as f32 / body.ring_count as f32;
        let anchor_scale = match (balloon.tail_inflated, balloon.head_inflated) {
            (true, false) => {
                // Tail anchored, head free.
                (1.0 - t) * 8.0 + t * 0.25
            }
            (false, true) => {
                // Head anchored, tail free (for retraction).
                t * 8.0 + (1.0 - t) * 0.25
            }
            (true, true) => {
                // Both anchored; clamp both ends.
                ((1.0 - t) * 8.0 + t * 8.0).max(1.0)
            }
            _ => 1.0,
        };
        fric.coefficient = control.friction * anchor_scale;
    }
}

pub fn distributed_thrust(
    balloon: Res<BalloonControl>,
    mut query: Query<
        (
            &SegmentIndex,
            &mut ExternalForce,
            &mut ExternalImpulse,
            &mut Velocity,
        ),
        With<ProbeSegment>,
    >,
) {
    // Pneumatic extension replaces thrust; keep external forces cleared.
    let active = balloon.tail_inflated && !balloon.head_inflated;
    for (_, mut force, mut impulse, mut velocity) in &mut query {
        force.force = Vec3::ZERO;
        impulse.impulse = Vec3::ZERO;
        if !active {
            velocity.linvel = Vec3::ZERO;
        }
    }
}
