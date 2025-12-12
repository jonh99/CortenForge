use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::f32::consts::FRAC_PI_2;

use crate::controls::ControlParams;
use crate::balloon_control::BalloonControl;

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

pub fn spawn_probe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    control: Res<ControlParams>,
) {
    // Capsule dimensions stretched longer than the tunnel (radius ~0.8, length ~16.0).
    let collider_radius = 0.8;
    let total_length = 32.0;
    let segment_count = 12;
    let segment_length = total_length / segment_count as f32;
    let segment_half_height = segment_length * 0.5 - collider_radius;
    let span = segment_half_height + collider_radius;

    let mesh = meshes.add(Mesh::from(Capsule3d::new(
        collider_radius,
        segment_half_height * 2.0,
    )));
    let material_handle = materials.add(Color::srgb(0.8, 0.2, 0.2));

    // Place the chain well inside the long straight (length_a ~60); keep tail inside and tip shy of the bend.
    let joint_front_second_z = 12.0;
    let base_rotation = Quat::from_rotation_x(FRAC_PI_2);

    let mut segments = Vec::new();
    for i in 0..segment_count {
        let center_z = joint_front_second_z + span - (2.0 * span * i as f32);
        let mut entity = commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material_handle.clone()),
            Transform {
                translation: Vec3::new(0.0, 0.0, center_z),
                rotation: base_rotation,
                ..default()
            },
            RigidBody::Dynamic,
            Collider::capsule_y(segment_half_height, collider_radius),
            Friction {
                coefficient: control.friction,
                combine_rule: CoefficientCombineRule::Average,
                ..default()
            },
            Velocity::default(),
            Sleeping::disabled(),
            ExternalImpulse::default(),
            ExternalForce::default(),
            Damping {
                linear_damping: control.linear_damping,
                angular_damping: 0.3,
            },
            ProbeSegment,
            SegmentIndex(i),
        ));

        if i == 0 {
            entity.insert((ProbeTip, ProbeHead));
        }
        if i == segment_count - 1 {
            entity.insert(CapsuleProbe);
        }

        // Front four segments (including tip) get their own collision group to be ignored by balloon shell.
        let groups = if i < 4 {
            CollisionGroups::new(Group::GROUP_3, Group::ALL)
        } else {
            CollisionGroups::new(Group::GROUP_1, Group::ALL)
        };
        entity.insert(groups);

        segments.push(entity.id());
    }

    let anchor_top = Vec3::Y * span;
    let anchor_bottom = Vec3::NEG_Y * span;
    let base_rest = 2.0 * span;
    let base_stiffness = control.stiffness;
    let base_damping = control.damping;
    let rest_length = base_rest * control.tension;

    for window in segments.windows(2) {
        let parent = window[1];
        let child = window[0];
        let mut joint = SphericalJointBuilder::new()
            .local_anchor1(anchor_top)
            .local_anchor2(anchor_bottom)
            .build();
        joint.set_contacts_enabled(false);
        commands
            .entity(child)
            .insert(ImpulseJoint::new(parent, joint));

        let mut spring = GenericJointBuilder::new(JointAxesMask::empty())
            .local_anchor1(anchor_top)
            .local_anchor2(anchor_bottom)
            .motor_position(JointAxis::LinY, rest_length, base_stiffness, base_damping)
            .build();
        spring.set_contacts_enabled(false);
        spring.set_motor_model(JointAxis::LinY, MotorModel::AccelerationBased);
        spring.set_motor_max_force(JointAxis::LinY, 10_000.0);

        commands.spawn((
            SegmentSpring { base_rest },
            ImpulseJoint::new(parent, TypedJoint::GenericJoint(spring)),
            Transform::default(),
            GlobalTransform::default(),
            ChildOf(child),
        ));
    }
}

pub fn peristaltic_drive(
    time: Res<Time>,
    control: Res<ControlParams>,
    balloon: Res<BalloonControl>,
    mut joints: Query<(&mut ImpulseJoint, &SegmentSpring, &SegmentIndex)>,
    mut frictions: Query<(&SegmentIndex, &mut Friction), With<ProbeSegment>>,
) {
    if !(balloon.head_inflated || balloon.tail_inflated) {
        return;
    }

    let amp = 0.35;
    let freq = 0.6;
    let phase_step = 0.9;
    let anchor_on = 1.4;
    let anchor_off = 0.6;
    let omega = freq * std::f32::consts::TAU;
    let t = time.elapsed_secs();

    for (mut joint, spring, idx) in &mut joints {
        if let TypedJoint::GenericJoint(ref mut j) = joint.data {
            let base = spring.base_rest * control.tension;
            let wave = 1.0 + amp * (omega * t + phase_step * idx.0 as f32).sin();
            let target = (base * wave).max(0.05);
            j.set_motor_position(JointAxis::LinY, target, control.stiffness, control.damping);
            j.set_motor_model(JointAxis::LinY, MotorModel::AccelerationBased);
            j.set_motor_max_force(JointAxis::LinY, 10_000.0);
        }
    }

    // Modulate friction in phase with the wave: contracting segments anchor harder, relaxing segments slip easier.
    for (idx, mut fric) in &mut frictions {
        let wave = (omega * t + phase_step * idx.0 as f32).sin();
        let scale = if wave > 0.0 { anchor_on } else { anchor_off };
        fric.coefficient = control.friction * scale;
    }
}

pub fn distributed_thrust(
    keys: Res<ButtonInput<KeyCode>>,
    control: Res<ControlParams>,
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
    if !(balloon.head_inflated || balloon.tail_inflated) {
        for (_, mut force, mut impulse, mut velocity) in &mut query {
            force.force = Vec3::ZERO;
            impulse.impulse = Vec3::ZERO;
            velocity.linvel = Vec3::ZERO;
        }
        return;
    }
    let impulse_strength = 1.5;
    let thrust_links = 4usize;

    let forward_pressed = keys.pressed(KeyCode::ArrowUp) || keys.pressed(KeyCode::KeyI);
    let backward_pressed = keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyK);
    let dir = if forward_pressed {
        Vec3::Z
    } else if backward_pressed {
        -Vec3::Z
    } else {
        Vec3::ZERO
    };

    for (idx, mut force, mut impulse, mut velocity) in &mut query {
        let weight = if idx.0 < thrust_links {
            1.0 - idx.0 as f32 / thrust_links as f32
        } else {
            0.2
        };

        if dir != Vec3::ZERO {
            let thrust = control.thrust * weight.max(0.1);
            let target_speed = control.target_speed * weight.max(0.1);
            force.force = dir * thrust;
            impulse.impulse = dir * impulse_strength * weight;
            velocity.linvel = dir * target_speed;
        } else {
            force.force = Vec3::ZERO;
            impulse.impulse = Vec3::ZERO;
        }
    }
}
