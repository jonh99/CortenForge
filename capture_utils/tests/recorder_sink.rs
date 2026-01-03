use capture_utils::JsonRecorder;
use std::fs;
use tempfile::tempdir;
use vision_core::prelude::{Frame, FrameRecord, Label, Recorder};

#[test]
fn json_recorder_writes_label_file() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");

    let mut recorder = JsonRecorder::new(&run_dir);
    let frame = Frame {
        id: 1,
        timestamp: 0.0,
        rgba: None,
        size: (1, 1),
        path: Some(run_dir.join("images/frame_00001.png")),
    };
    let labels: [Label; 1] = [Label {
        center_world: [0.0, 0.0, 0.0],
        bbox_px: Some([0.0, 0.0, 1.0, 1.0]),
        bbox_norm: Some([0.0, 0.0, 1.0, 1.0]),
    }];
    let record = FrameRecord {
        frame,
        labels: &labels,
        camera_active: true,
        polyp_seed: 42,
    };

    recorder.record(&record).expect("record");

    let labels_dir = run_dir.join("labels");
    let files: Vec<_> = fs::read_dir(labels_dir).unwrap().flatten().collect();
    assert_eq!(files.len(), 1);
    let contents = fs::read_to_string(files[0].path()).unwrap();
    assert!(contents.contains("\"frame_id\": 1"));
}
