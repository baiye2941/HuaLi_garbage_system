use huali_garbage_core::{iou, filter_overlapping_boxes, dedupe_track_events, BBox, TrackEvent};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum Request {
    ComputeIou { a: BBox, b: BBox },
    FilterOverlappingBoxes { boxes: Vec<BBox>, threshold: f64 },
    DedupeTrackEvents { events: Vec<TrackEvent>, cooldown_ms: i64, iou_threshold: f64 },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response {
    OkIou { value: f64 },
    OkBoxes { boxes: Vec<BBox> },
    OkEvents { events: Vec<TrackEvent> },
    Err { message: String },
}

fn process_request(req: Request) -> Response {
    match req {
        Request::ComputeIou { a, b } => Response::OkIou { value: iou(a, b) },
        Request::FilterOverlappingBoxes { boxes, threshold } => Response::OkBoxes {
            boxes: filter_overlapping_boxes(boxes, threshold),
        },
        Request::DedupeTrackEvents { events, cooldown_ms, iou_threshold } => Response::OkEvents {
            events: dedupe_track_events(events, cooldown_ms, iou_threshold),
        },
    }
}

/// Reads one JSON object per line from stdin, writes one JSON response per line to stdout.
/// Runs until stdin is closed (EOF).  This persistent-process mode lets the Python caller
/// reuse a single subprocess across many requests, eliminating per-call process-spawn overhead.
fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(_) => break,
        };

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => process_request(req),
            Err(err) => Response::Err {
                message: format!("invalid request: {err}"),
            },
        };

        let _ = writeln!(out, "{}", serde_json::to_string(&response).unwrap());
        let _ = out.flush();
    }
}
