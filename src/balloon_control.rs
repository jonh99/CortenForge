use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::probe::{CapsuleProbe, ProbeHead};

#[derive(Resource)]
pub struct BalloonControl {
    pub max_offset: f32,
    pub move_speed: f32,
    pub head_inflated: bool,
    pub tail_inflated: bool,
    pub deflated_radius: f32,
    pub inflated_radius: f32,
    pub half_length: f32,
    pub position: Vec3,
    pub rear_position: Vec3,
    pub initialized: bool,
}

impl Default for BalloonControl {
    fn default() -> Self {
        Self {
            max_offset: 8.0,
            move_speed: 3.0,
            head_inflated: false,
            tail_inflated: false,
            deflated_radius: 0.3,
            inflated_radius: 1.6,
            half_length: 0.8,
            position: Vec3::ZERO,
            rear_position: Vec3::ZERO,
            initialized: false,
        }
    }
}

/// Simple input for virtual balloons anchored to the probe ends.
/// B toggles the head inflation; N toggles the tail inflation. Each squeezes the tunnel independently.
pub fn balloon_control_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut balloon: ResMut<BalloonControl>,
    tip_q: Query<&GlobalTransform, With<ProbeHead>>,
    tail_q: Query<&GlobalTransform, With<CapsuleProbe>>,
) {
    let Some(tip_tf) = tip_q.iter().next() else {
        return;
    };
    let Some(tail_tf) = tail_q.iter().next() else {
        return;
    };

    let tip_pos = tip_tf.translation();
    let tail_pos = tail_tf.translation();

    if !balloon.initialized {
        balloon.initialized = true;
    }

    if keys.just_pressed(KeyCode::KeyB) {
        balloon.head_inflated = !balloon.head_inflated;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        balloon.tail_inflated = !balloon.tail_inflated;
    }

    // Keep balloons centered on their respective capsules.
    balloon.position = tip_pos;
    balloon.rear_position = tail_pos;
}

#[derive(Component)]
pub struct BalloonMarker;

#[derive(Component)]
pub struct BalloonMarkerRear;

pub fn spawn_balloon_marker(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Mesh::from(bevy::math::primitives::Sphere {
        radius: 1.0,
    }));
    let mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.55, 1.0, 0.55, 0.15),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });

    let marker_bundle = (
        Mesh3d(mesh.clone()),
        MeshMaterial3d(mat.clone()),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
    );

    commands.spawn((BalloonMarker, marker_bundle.clone()));
    commands.spawn((BalloonMarkerRear, marker_bundle));
}

pub fn balloon_marker_update(
    balloon: Res<BalloonControl>,
    mut markers: ParamSet<(
        Query<&mut Transform, With<BalloonMarker>>,
        Query<&mut Transform, With<BalloonMarkerRear>>,
    )>,
) {
    let head_radius = if balloon.head_inflated {
        balloon.inflated_radius
    } else {
        balloon.deflated_radius
    };
    let tail_radius = if balloon.tail_inflated {
        balloon.inflated_radius
    } else {
        balloon.deflated_radius
    };

    if let Ok(mut tf) = markers.p0().single_mut() {
        tf.translation = balloon.position;
        tf.scale = Vec3::splat(head_radius);
    }

    if let Ok(mut tf) = markers.p1().single_mut() {
        tf.translation = balloon.rear_position;
        tf.scale = Vec3::splat(tail_radius);
    }
}

#[derive(Component)]
pub struct BalloonBody;
#[derive(Component)]
pub struct BalloonWall;

pub fn spawn_balloon_body(mut commands: Commands) {
    commands.spawn((
        BalloonBody,
        Transform::default(),
        GlobalTransform::default(),
        RigidBody::KinematicPositionBased,
        Collider::capsule_z(0.5, 0.3),
        Sensor,
        CollisionGroups::new(Group::GROUP_2, Group::ALL ^ (Group::GROUP_1 | Group::GROUP_3)),
    ));

    commands.spawn((
        BalloonWall,
        Transform::default(),
        GlobalTransform::default(),
        RigidBody::KinematicPositionBased,
        Collider::ball(0.4),
        CollisionGroups::new(Group::GROUP_4, Group::GROUP_3),
    ));
}

pub fn balloon_body_update(
    balloon: Res<BalloonControl>,
    mut parts: ParamSet<(
        Query<(&mut Transform, &mut Collider), With<BalloonBody>>,
        Query<&mut Transform, With<BalloonWall>>,
    )>,
) {
    let mut body_q = parts.p0();
    let Ok((mut tf, mut collider)) = body_q.single_mut() else {
        return;
    };

    // Keep collider centered on the balloon position (matches capsule center).
    tf.translation = balloon.position;
    tf.rotation = Quat::IDENTITY;

    let radius = if balloon.head_inflated {
        balloon.inflated_radius
    } else {
        balloon.deflated_radius
    };
    *collider = Collider::capsule_z(balloon.half_length, radius);

    let mut wall_q = parts.p1();
    if let Ok(mut wall_tf) = wall_q.single_mut() {
        // Place stop further ahead of the balloon tip to align with the expanded tunnel bulge.
        let front_offset = balloon.half_length * 5.0;
        wall_tf.translation = balloon.position + Vec3::new(0.0, 0.0, front_offset);
        wall_tf.rotation = Quat::IDENTITY;
    }
}
