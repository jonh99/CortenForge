use bevy::prelude::*;

use crate::balloon_control::BalloonControl;
use crate::cli::RunMode;
use crate::polyp::PolypRemoval;
use crate::probe::StretchState;
use crate::probe::ProbeHead;
use crate::tunnel::{CecumState, TUNNEL_LENGTH, TUNNEL_START_Z};
use crate::camera::{Flycam, PovState, ProbePovCamera};
use crate::vision::{AutoRecordTimer, RecorderState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoStage {
    AnchorTail,
    Extend,
    AnchorHead,
    ReleaseTail,
    Contract,
    ReleaseHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoDir {
    Forward,
    Reverse,
}

#[derive(Resource)]
pub struct AutoDrive {
    pub enabled: bool,
    pub stage: AutoStage,
    pub timer: f32,
    pub extend: bool,
    pub retract: bool,
    pub dir: AutoDir,
    pub last_head_z: f32,
    pub stuck_time: f32,
    pub primed_reverse: bool,
}

impl Default for AutoDrive {
    fn default() -> Self {
        Self {
            enabled: false,
            stage: AutoStage::AnchorTail,
            timer: 0.0,
            extend: false,
            retract: false,
            dir: AutoDir::Forward,
            last_head_z: 0.0,
            stuck_time: 0.0,
            primed_reverse: false,
        }
    }
}


pub fn auto_toggle(keys: Res<ButtonInput<KeyCode>>, mut auto: ResMut<AutoDrive>) {
    if keys.just_pressed(KeyCode::KeyP) {
        auto.enabled = !auto.enabled;
        auto.stage = AutoStage::AnchorTail;
        auto.timer = 0.2;
        auto.extend = false;
        auto.retract = false;
        auto.dir = AutoDir::Forward;
        auto.last_head_z = 0.0;
        auto.stuck_time = 0.0;
        auto.primed_reverse = false;
    }
}

#[derive(Resource, Default)]
pub struct DataRun {
    pub active: bool,
}

#[derive(Resource, Default)]
pub struct DatagenInit {
    pub started: bool,
    pub elapsed: f32,
}

pub fn data_run_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut auto: ResMut<AutoDrive>,
    mut pov: ResMut<PovState>,
    mut free_cams: Query<&mut Camera, (With<Flycam>, Without<ProbePovCamera>)>,
    mut probe_cams: Query<&mut Camera, With<ProbePovCamera>>,
    mut auto_timer: ResMut<AutoRecordTimer>,
    recorder: Res<RecorderState>,
    mut data_run: ResMut<DataRun>,
) {
    if !keys.just_pressed(KeyCode::KeyO) {
        return;
    }

    let enabling = !data_run.active;
    data_run.active = enabling;

    // Toggle autopilot to match data run state.
    auto.enabled = enabling;
    auto.stage = AutoStage::AnchorTail;
    auto.timer = 0.2;
    auto.extend = false;
    auto.retract = false;
    auto.dir = AutoDir::Forward;
    auto.last_head_z = 0.0;
    auto.stuck_time = 0.0;
    auto.primed_reverse = false;

    // Force probe POV active when starting.
    if enabling {
        pov.use_probe = true;
        for mut cam in &mut free_cams {
            cam.is_active = false;
        }
        for mut cam in &mut probe_cams {
            cam.is_active = true;
        }
    }

    // Reset/arm auto-record timer when starting a data run.
    if enabling && !recorder.enabled {
        auto_timer.timer = Timer::from_seconds(8.0, TimerMode::Once);
    } else {
        auto_timer.timer.reset();
    }
}

pub fn datagen_autostart(
    mode: Res<RunMode>,
    mut init: ResMut<DatagenInit>,
    mut auto: ResMut<AutoDrive>,
    mut data_run: ResMut<DataRun>,
    mut pov: ResMut<PovState>,
    mut free_cams: Query<&mut Camera, (With<Flycam>, Without<ProbePovCamera>)>,
    mut probe_cams: Query<&mut Camera, With<ProbePovCamera>>,
) {
    if *mode != RunMode::Datagen || init.started {
        return;
    }
    init.started = true;
    init.elapsed = 0.0;
    data_run.active = true;

    auto.enabled = true;
    auto.stage = AutoStage::AnchorTail;
    auto.timer = 0.2;
    auto.extend = false;
    auto.retract = false;
    auto.dir = AutoDir::Forward;
    auto.last_head_z = 0.0;
    auto.stuck_time = 0.0;
    auto.primed_reverse = false;

    pov.use_probe = true;
    for mut cam in &mut free_cams {
        cam.is_active = false;
    }
    for mut cam in &mut probe_cams {
        cam.is_active = true;
    }
}

pub fn auto_inchworm(
    time: Res<Time>,
    removal: Res<PolypRemoval>,
    mut auto: ResMut<AutoDrive>,
    mut balloon: ResMut<BalloonControl>,
    stretch: Res<StretchState>,
    head_q: Query<&GlobalTransform, With<ProbeHead>>,
    mut cecum: ResMut<CecumState>,
    mut start: ResMut<crate::tunnel::StartState>,
) {
    // If autopilot off, do nothing.
    if !auto.enabled {
        auto.extend = false;
        auto.retract = false;
        return;
    }

    // Pause during removal dwell.
    if removal.in_progress {
        auto.extend = false;
        auto.retract = false;
        return;
    }

    // Flip to reverse when reaching the cecum/end.
    if let Ok(head_tf) = head_q.single() {
        let head_z = head_tf.translation().z;
        // Start reversal a bit before the physical end to avoid ramming the marker.
        let end_z = TUNNEL_START_Z + TUNNEL_LENGTH - 2.0;
        let start_z = TUNNEL_START_Z + 0.5;

        // Track stall near end to force reversal even if marker missed.
        let dz = head_z - auto.last_head_z;
        if auto.dir == AutoDir::Forward && head_z > end_z - 2.0 {
            if dz.abs() < 0.01 {
                auto.stuck_time += time.delta_secs();
            } else {
                auto.stuck_time = 0.0;
            }
        } else {
            auto.stuck_time = 0.0;
        }
        auto.last_head_z = head_z;

        let should_flip = auto.dir == AutoDir::Forward
            && (head_z >= end_z || cecum.reached || auto.stuck_time > 0.6);
        if should_flip {
            auto.dir = AutoDir::Reverse;
            auto.stage = AutoStage::Contract; // clamp head, pull tail back
            auto.timer = 0.0;
            auto.extend = false;
            auto.retract = true;
            balloon.head_inflated = true;
            balloon.tail_inflated = false;
            auto.primed_reverse = true;
        } else if auto.dir == AutoDir::Reverse && (head_z <= start_z || start.reached) {
            // Arrived back at start; stop autopilot and neutralize.
            auto.enabled = false;
            auto.extend = false;
            auto.retract = false;
            auto.stage = AutoStage::AnchorTail;
            auto.dir = AutoDir::Forward;
            auto.primed_reverse = false;
            balloon.head_inflated = false;
            balloon.tail_inflated = false;
            start.reached = false;
            cecum.reached = false;
            return;
        }
    }
    // If cecum is flagged while going forward, ensure we stop pushing.
    if auto.dir == AutoDir::Forward && cecum.reached {
        auto.extend = false;
        auto.retract = true;
        auto.dir = AutoDir::Reverse;
        auto.stage = AutoStage::Contract;
        auto.timer = 0.0;
        balloon.head_inflated = true;
        balloon.tail_inflated = false;
        auto.primed_reverse = true;
    }

    auto.timer += time.delta_secs();
    auto.extend = false;
    auto.retract = false;

    // If we just flipped to reverse, enforce initial clamp state before continuing the reverse cycle.
    if auto.dir == AutoDir::Reverse && auto.primed_reverse {
        balloon.head_inflated = true;
        balloon.tail_inflated = false;
        auto.stage = AutoStage::Contract;
        auto.retract = true;
        if auto.timer > 0.1 {
            auto.primed_reverse = false;
            auto.timer = 0.0;
            auto.retract = false;
        }
    }

    match (auto.dir, auto.stage) {
        (AutoDir::Forward, AutoStage::AnchorTail) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = false;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::Extend;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Forward, AutoStage::Extend) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = false;
            auto.extend = true;
            if stretch.factor >= crate::probe::MAX_STRETCH - 0.02 {
                auto.stage = AutoStage::AnchorHead;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Forward, AutoStage::AnchorHead) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = true;
            if auto.timer > 0.2 {
                auto.stage = AutoStage::ReleaseTail;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Forward, AutoStage::ReleaseTail) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = true;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::Contract;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Forward, AutoStage::Contract) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = true;
            auto.retract = true;
            if stretch.factor <= crate::probe::MIN_STRETCH + 0.02 {
                auto.stage = AutoStage::ReleaseHead;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Forward, AutoStage::ReleaseHead) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = false;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::AnchorTail;
                auto.timer = 0.0;
            }
        }
        // Reverse direction: swap roles to move toward the start.
        (AutoDir::Reverse, AutoStage::AnchorHead) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = true;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::Extend;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Reverse, AutoStage::Extend) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = true;
            auto.extend = true;
            if stretch.factor >= crate::probe::MAX_STRETCH - 0.02 {
                auto.stage = AutoStage::AnchorTail;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Reverse, AutoStage::AnchorTail) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = true;
            if auto.timer > 0.2 {
                auto.stage = AutoStage::ReleaseHead;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Reverse, AutoStage::ReleaseHead) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = false;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::Contract;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Reverse, AutoStage::Contract) => {
            balloon.tail_inflated = true;
            balloon.head_inflated = false;
            auto.retract = true;
            if stretch.factor <= crate::probe::MIN_STRETCH + 0.02 {
                auto.stage = AutoStage::ReleaseTail;
                auto.timer = 0.0;
            }
        }
        (AutoDir::Reverse, AutoStage::ReleaseTail) => {
            balloon.tail_inflated = false;
            balloon.head_inflated = false;
            if auto.timer > 0.1 {
                auto.stage = AutoStage::AnchorHead;
                auto.timer = 0.0;
            }
        }
    }
}
