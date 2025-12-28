use bevy::prelude::*;
use bevy::ui::{
    AlignItems, BackgroundColor, BorderColor, BorderRadius, Display, FlexDirection, JustifyContent,
    PositionType, UiRect, Val,
};

use crate::controls::ControlParams;
use crate::polyp::PolypTelemetry;
use crate::vision::overlay::normalize_box;
use crate::probe::TipSense;
use crate::vision::{
    BurnInferenceState, DetectionOverlayState, DetectorHandle, FrontCameraState, RecorderState,
};

#[derive(Component)]
pub struct ControlText;
#[derive(Component)]
pub struct DetectionOverlayRoot;
#[derive(Component)]
pub struct DetectionBoxUI;
#[derive(Component)]
pub struct DetectionLabelUI;
#[derive(Component)]
pub struct FallbackBanner;

pub fn spawn_controls_ui(mut commands: Commands) {
    let bg = Color::srgba(0.04, 0.08, 0.14, 0.82);
    let border = Color::srgba(0.0, 0.8, 0.75, 0.85);
    let accent = Color::srgba(0.28, 0.9, 1.0, 0.95);
    let line = Color::srgba(0.75, 0.88, 1.0, 0.9);
    let soft = Color::srgba(0.5, 0.74, 1.0, 0.85);

    commands.spawn((
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::FlexStart,
            align_items: AlignItems::FlexStart,
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            padding: UiRect::axes(Val::Px(14.0), Val::Px(12.0)),
            border: UiRect::all(Val::Px(1.5)),
            width: Val::Px(360.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        BorderRadius::all(Val::Px(10.0)),
        Text::new("AUX // PROBE HUD\n───────────────\n"),
        TextFont {
            font_size: 17.0,
            ..default()
        },
        TextColor(accent),
        ControlText,
        children![
            (
                TextSpan::from("TNS :: 0.50 [ [ ] ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("STF :: 500 [ ; ' ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("DMP :: 20 [ , . ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(soft),
            ),
            (
                TextSpan::from("THR :: 40 [ 1 2 ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("SPD :: 1.20 [ 3 4 ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(soft),
            ),
            (
                TextSpan::from("LIN :: 0.20 [ 5 6 ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("FRI :: 1.20 [ 7 8 ]\n"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("TIP PRESS R/U/F: 0.0 / 0.0 / 0.0\n"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(soft),
            ),
            (
                TextSpan::from("TIP STEER >> 0.0 0.0 0.0 (0.0)"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(accent),
            ),
            (
                TextSpan::from("POLYPS :: 0/0\n"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(line),
            ),
            (
                TextSpan::from("NEAREST :: -- m"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(soft),
            ),
            (
                TextSpan::from("VISION :: cam=OFF burn=--\n"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(soft),
            ),
            (
                TextSpan::from("CONSENSUS :: hold"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(accent),
            ),
            (
                TextSpan::from("REC :: off\n"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(Color::srgba(0.9, 0.2, 0.2, 0.95)),
            ),
            (
                TextSpan::from("REMOVAL :: idle"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(accent),
            ),
        ],
    ));
}

pub fn update_detection_overlay_ui(
    mut commands: Commands,
    overlay: Res<DetectionOverlayState>,
    root_q: Query<Entity, With<DetectionOverlayRoot>>,
    boxes_q: Query<Entity, With<DetectionBoxUI>>,
    labels_q: Query<Entity, With<DetectionLabelUI>>,
    mut fallback_q: Query<(&mut Node, &mut Text), With<FallbackBanner>>,
) {
    if overlay.is_changed() {
        for e in boxes_q.iter() {
            commands.entity(e).despawn();
        }
        for e in labels_q.iter() {
            commands.entity(e).despawn();
        }
        let Some(root) = root_q.iter().next() else {
            return;
        };
        let mut boxes = overlay
            .boxes
            .iter()
            .zip(overlay.scores.iter().cloned())
            .collect::<Vec<_>>();
        boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, score) in boxes {
            let Some([x0, y0, x1, y1]) = normalize_box(*b, overlay.size) else {
                continue;
            };
            let w = ((x1.saturating_sub(x0)) as f32).clamp(0.0, overlay.size.0 as f32)
                / overlay.size.0 as f32
                * 100.0;
            let h = ((y1.saturating_sub(y0)) as f32).clamp(0.0, overlay.size.1 as f32)
                / overlay.size.1 as f32
                * 100.0;
            let color = Color::srgba(0.1, 0.8, 1.0, 0.9 * score.clamp(0.25, 1.0));
            commands.entity(root).with_children(|parent| {
                let _ = parent
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Percent((x0 as f32 / overlay.size.0 as f32) * 100.0),
                            top: Val::Percent((y0 as f32 / overlay.size.1 as f32) * 100.0),
                            width: Val::Percent(w),
                            height: Val::Percent(h),
                            border: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BorderColor::all(color),
                        BackgroundColor(Color::srgba(0.1, 0.7, 1.0, 0.08)),
                        DetectionBoxUI,
                    ))
                    .id();
                // label
                parent.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Percent((b[0] * 100.0).clamp(0.0, 100.0)),
                        top: Val::Percent(((b[1] * 100.0) - 2.0).clamp(0.0, 100.0)),
                        ..default()
                    },
                    Text::new(format!("polyp {:.2}", score)),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgba(0.95, 0.98, 1.0, 0.95)),
                    DetectionLabelUI,
                ));
            });
        }

        if let Some((mut node, mut text)) = fallback_q.iter_mut().next() {
            match overlay.fallback.as_ref() {
                Some(msg) => {
                    node.display = Display::Flex;
                    *text = Text::new(msg.clone());
                }
                None => {
                    node.display = Display::None;
                }
            }
        }
    }
}

pub fn spawn_detection_overlay(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        BackgroundColor(Color::NONE),
        DetectionOverlayRoot,
    ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            right: Val::Px(10.0),
            padding: UiRect::axes(Val::Px(8.0), Val::Px(6.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.12, 0.04, 0.04, 0.82)),
        BorderColor::all(Color::srgba(0.8, 0.2, 0.2, 0.9)),
        BorderRadius::all(Val::Px(6.0)),
        Text::new(""),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 0.9, 0.9, 0.95)),
        FallbackBanner,
    ));
}

pub fn update_controls_ui(
    control: Res<ControlParams>,
    sense: Res<TipSense>,
    polyps: Res<PolypTelemetry>,
    front_cam: Res<FrontCameraState>,
    burn: Res<BurnInferenceState>,
    overlay: Res<DetectionOverlayState>,
    handle: Option<Res<DetectorHandle>>,
    recorder: Res<RecorderState>,
    ui: Single<Entity, (With<ControlText>, With<Text>)>,
    mut writer: TextUiWriter,
) {
    if control.is_changed()
        || sense.is_changed()
        || polyps.is_changed()
        || front_cam.is_changed()
        || burn.is_changed()
        || recorder.is_changed()
    {
        *writer.text(*ui, 1) = format!("TNS :: {:.2} [ [ ] ]\n", control.tension);
        *writer.text(*ui, 2) = format!("STF :: {:.0} [ ; ' ]\n", control.stiffness);
        *writer.text(*ui, 3) = format!("DMP :: {:.1} [ , . ]\n", control.damping);
        *writer.text(*ui, 4) = format!("THR :: {:.1} [ 1 2 ]\n", control.thrust);
        *writer.text(*ui, 5) = format!("SPD :: {:.2} [ 3 4 ]\n", control.target_speed);
        *writer.text(*ui, 6) = format!("LIN :: {:.2} [ 5 6 ]\n", control.linear_damping);
        *writer.text(*ui, 7) = format!("FRI :: {:.2} [ 7 8 ]\n", control.friction);

        // Local pressure components: +X right, +Y up, +Z forward along the tip.
        let px = sense.pressure_local.x;
        let py = sense.pressure_local.y;
        let pz = sense.pressure_local.z;
        *writer.text(*ui, 8) = format!("TIP PRESS R/U/F: {:.2} / {:.2} / {:.2}\n", px, py, pz);
        *writer.text(*ui, 9) = format!(
            "TIP STEER >> {:.2} {:.2} {:.2} ({:.2})",
            sense.steer_dir.x, sense.steer_dir.y, sense.steer_dir.z, sense.steer_strength
        );
        *writer.text(*ui, 10) = format!("POLYPS :: {}/{}\n", polyps.remaining, polyps.total);
        let nearest_str = polyps
            .nearest_distance
            .map(|d| format!("{:.2} m", d))
            .unwrap_or_else(|| "--".into());
        *writer.text(*ui, 11) = format!("NEAREST :: {}", nearest_str);
        let cam_state = if front_cam.active { "ON" } else { "OFF" };
        let burn_state = match (burn.last_result.as_ref(), handle) {
            (Some(r), Some(handle)) => {
                let boxes = overlay.boxes.len();
                let max_score = overlay.scores.iter().cloned().fold(0.0_f32, f32::max);
                let latency = overlay
                    .inference_ms
                    .map(|ms| format!(" ; {:.1} ms", ms))
                    .unwrap_or_default();
                let kind = match handle.kind {
                    #[cfg(feature = "burn_runtime")]
                    crate::vision::DetectorKind::Burn => "BURN",
                    _ => "HEUR",
                };
                if r.positive {
                    if boxes > 0 {
                        format!(
                            "{} ON ({:.0}% ; {} boxes max {:.2}{})",
                            kind,
                            r.confidence * 100.0,
                            boxes,
                            max_score,
                            latency
                        )
                    } else {
                        format!(
                            "{} ON ({:.0}% ; 0 boxes{})",
                            kind,
                            r.confidence * 100.0,
                            latency
                        )
                    }
                } else {
                    format!(
                        "{} off ({:.0}% ; {} boxes{})",
                        kind,
                        r.confidence * 100.0,
                        boxes,
                        latency
                    )
                }
            }
            _ => "--".to_string(),
        };
        *writer.text(*ui, 12) = format!("VISION :: cam={} burn={}", cam_state, burn_state);
        let consensus = if polyps.consensus_ready {
            "go"
        } else if polyps.vision_detected {
            "wait classic"
        } else if polyps.classic_detected {
            "wait vision"
        } else {
            "hold"
        };
        *writer.text(*ui, 13) = format!("CONSENSUS :: {}", consensus);
        let rec_state = if recorder.enabled {
            if recorder.paused {
                format!("REC :: paused (#{})", recorder.frame_idx)
            } else {
                format!("REC ● on (#{})", recorder.frame_idx)
            }
        } else {
            "REC :: off".to_string()
        };
        *writer.text(*ui, 14) = rec_state;
        let removal_str = if polyps.removing {
            format!("REMOVAL :: {:.0}%", polyps.remove_progress * 100.0)
        } else {
            "REMOVAL :: idle".to_string()
        };
        *writer.text(*ui, 15) = removal_str;
    }
}
