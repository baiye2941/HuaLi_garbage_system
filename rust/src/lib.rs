use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl BBox {
    pub fn width(&self) -> i32 {
        (self.x2 - self.x1).max(0)
    }

    pub fn height(&self) -> i32 {
        (self.y2 - self.y1).max(0)
    }

    pub fn area(&self) -> i32 {
        self.width() * self.height()
    }
}

pub fn iou(a: BBox, b: BBox) -> f64 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0);
    let inter_h = (inter_y2 - inter_y1).max(0);
    let inter_area = (inter_w * inter_h) as f64;

    let area_a = a.area() as f64;
    let area_b = b.area() as f64;
    let union = area_a + area_b - inter_area;

    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

pub fn filter_overlapping_boxes(boxes: Vec<BBox>, threshold: f64) -> Vec<BBox> {
    let mut kept: Vec<BBox> = Vec::new();

    'outer: for candidate in boxes {
        for existing in &kept {
            if iou(candidate, *existing) >= threshold {
                continue 'outer;
            }
        }
        kept.push(candidate);
    }

    kept
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackEvent {
    pub class_id: i32,
    pub bbox: BBox,
    pub timestamp_ms: i64,
}

pub fn dedupe_track_events(events: Vec<TrackEvent>, cooldown_ms: i64, iou_threshold: f64) -> Vec<TrackEvent> {
    let mut kept: Vec<TrackEvent> = Vec::new();

    'event_loop: for event in events {
        for existing in &kept {
            if existing.class_id != event.class_id {
                continue;
            }
            if event.timestamp_ms - existing.timestamp_ms > cooldown_ms {
                continue;
            }
            if iou(existing.bbox, event.bbox) >= iou_threshold {
                continue 'event_loop;
            }
        }
        kept.push(event);
    }

    kept
}
