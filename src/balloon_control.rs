use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::probe::ProbeHead;

#[derive(Resource)]
pub struct BalloonControl {
    pub max_offset: f32,
    pub move_speed: f32,
    pub inflated: bool,
    pub deflated_radius: f32,
    pub inflated_radius: f32,
    pub half_length: f32,
    pub position: Vec3,
    pub initialized: bool,
}

impl Default for BalloonControl {
    fn default() -> Self {
        Self {
            max_offset: 8.0,
            move_speed: 3.0,
            inflated: false,
            deflated_radius: 0.3,
            inflated_radius: 1.6,
            half_length: 0.8,
            position: Vec3::ZERO,
            initialized: false,
        }
    }
}

/// Simple input for a virtual balloon separate from the probe tip.
/// B toggles inflate/deflate; V moves it forward and C moves it back along the tunnel (global +Z).
pub fn balloon_control_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut balloon: ResMut<BalloonControl>,
    tip_q: Query<&GlobalTransform, With<ProbeHead>>,
) {
    let Ok(tip_tf) = tip_q.single() else {
        return;
    };
    let tip_z = tip_tf.translation().z;
    let forward = Vec3::Z;

    if !balloon.initialized {
        balloon.position = Vec3::new(0.0, 0.0, tip_z + 2.0);
        balloon.initialized = true;
    }

    if keys.just_pressed(KeyCode::KeyB) {
        balloon.inflated = !balloon.inflated;
    }

    let step = balloon.move_speed * time.delta_secs();
    if keys.pressed(KeyCode::KeyV) {
        balloon.position += forward * step;
    }
    if keys.pressed(KeyCode::KeyC) {
        balloon.position -= forward * step;
    }

    let dist = balloon.position.z - tip_z;
    let clamped = dist.clamp(0.0, balloon.max_offset);
    balloon.position = Vec3::new(0.0, 0.0, tip_z + clamped);
}

#[derive(Component)]
pub struct BalloonMarker;

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

    commands.spawn((
        BalloonMarker,
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
    ));
}

pub fn balloon_marker_update(
    balloon: Res<BalloonControl>,
    mut marker_q: Query<&mut Transform, With<BalloonMarker>>,
) {
    let Ok(mut tf) = marker_q.single_mut() else {
        return;
    };

    let radius = if balloon.inflated {
        balloon.inflated_radius
    } else {
        balloon.deflated_radius
    };

    tf.translation = balloon.position;
    tf.scale = Vec3::splat(radius);
}

#[derive(Component)]
pub struct BalloonBody;

pub fn spawn_balloon_body(mut commands: Commands) {
    commands.spawn((
        BalloonBody,
        Transform::default(),
        GlobalTransform::default(),
        RigidBody::KinematicPositionBased,
        Collider::capsule_z(0.5, 0.3),
        CollisionGroups::default(),
    ));
}

pub fn balloon_body_update(
    balloon: Res<BalloonControl>,
    mut body_q: Query<(&mut Transform, &mut Collider), With<BalloonBody>>,
) {
    let Ok((mut tf, mut collider)) = body_q.single_mut() else {
        return;
    };

    tf.translation = balloon.position;
    tf.rotation = Quat::IDENTITY;

    let radius = if balloon.inflated {
        balloon.inflated_radius
    } else {
        balloon.deflated_radius
    };
    *collider = Collider::capsule_z(balloon.half_length, radius);
}
