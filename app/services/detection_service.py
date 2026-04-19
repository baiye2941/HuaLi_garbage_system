from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.constants import ALL_CLASSES
from app.services.inference import InferenceBackend, OnnxYoloBackend, UltralyticsBackend
from app.utils import frame_to_base64


@dataclass
class DetectorBundle:
    key: str
    class_mapping: dict[int, int]
    backend: InferenceBackend | None = None


class AlertCooldown:
    COOLDOWN_CONFIG = {
        "overflow": 15 * 60,
        "garbage": 15 * 60,
        "fire": 90,
        "smoke": 90,
    }

    def __init__(self) -> None:
        self._last_alert_time: dict[int, float] = {}

    def _get_cooldown_seconds(self, class_id: int) -> int:
        class_name = ALL_CLASSES.get(class_id, {}).get("en", "")
        return self.COOLDOWN_CONFIG.get(class_name, 15 * 60)

    def can_alert(self, class_id: int) -> bool:
        cooldown = self._get_cooldown_seconds(class_id)
        now = time.time()
        last = self._last_alert_time.get(class_id, 0.0)
        if now - last >= cooldown:
            self._last_alert_time[class_id] = now
            return True
        return False


class DetectionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cooldown = AlertCooldown()
        self.detectors = self._build_detectors()

    def _select_backend(self, onnx_path: Path, pt_path: Path) -> InferenceBackend | None:
        onnx_backend = OnnxYoloBackend(onnx_path)
        if onnx_backend.loaded:
            return onnx_backend

        pt_backend = UltralyticsBackend(pt_path)
        if pt_backend.loaded:
            return pt_backend

        return None

    def _build_detectors(self) -> list[DetectorBundle]:
        return [
            DetectorBundle(
                key="garbage",
                class_mapping={0: 2, 1: 0, 2: 1},
                backend=self._select_backend(self.settings.garbage_onnx_model, self.settings.garbage_pt_model),
            ),
            DetectorBundle(
                key="fire",
                class_mapping={0: 3},
                backend=self._select_backend(self.settings.fire_onnx_model, self.settings.fire_pt_model),
            ),
            DetectorBundle(
                key="smoke",
                class_mapping={1: 4},
                backend=self._select_backend(self.settings.smoke_onnx_model, self.settings.smoke_pt_model),
            ),
        ]

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {bundle.key: bool(bundle.backend and bundle.backend.loaded) for bundle in self.detectors}

    def detect(self, image: np.ndarray) -> list[dict]:
        if any(self.models_loaded.values()):
            return self._run_models(image)
        return self._fake_detect(image)

    def _run_models(self, image: np.ndarray) -> list[dict]:
        result_list: list[dict] = []
        h, w = image.shape[:2]
        min_box_area = w * h * 0.005

        for bundle in self.detectors:
            if bundle.backend is None or not bundle.backend.loaded:
                continue

            if bundle.key == "fire":
                conf_threshold = 0.15
            elif bundle.key == "smoke":
                conf_threshold = 0.15
            else:
                conf_threshold = self.settings.default_conf_threshold
            for prediction in bundle.backend.predict(
                image=image,
                conf_threshold=conf_threshold,
                iou_threshold=self.settings.default_iou_threshold,
            ):
                class_id = bundle.class_mapping.get(prediction.class_id)
                if class_id is None:
                    continue

                x1, y1, x2, y2 = prediction.bbox
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue

                info = ALL_CLASSES[class_id]
                result_list.append(
                    {
                        "class_id": class_id,
                        "class_name": info["name"],
                        "confidence": round(prediction.confidence, 3),
                        "bbox": [x1, y1, x2, y2],
                        "alert": info["alert"],
                        "color": info["color"],
                        "icon": info["icon"],
                        "source_model": bundle.key if bundle.backend else "unknown",
                    },
                )

        return self._suppress_garbage_when_fire_overlaps(result_list)

    @staticmethod
    def _iou(box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _suppress_garbage_when_fire_overlaps(self, detections: list[dict]) -> list[dict]:
        # If a scattered-garbage box heavily overlaps a fire box, prefer fire and drop garbage.
        overlap_threshold = 0.35
        fire_boxes = [d["bbox"] for d in detections if d["class_id"] == 3]
        if not fire_boxes:
            return detections

        filtered: list[dict] = []
        for det in detections:
            if det["class_id"] != 2:
                filtered.append(det)
                continue

            has_near_fire = any(self._iou(det["bbox"], fire_box) >= overlap_threshold for fire_box in fire_boxes)
            if not has_near_fire:
                filtered.append(det)

        return filtered

    def _fake_detect(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        seed_val = int(image[h // 2, w // 2, 0]) if image.ndim == 3 else 0
        rng = random.Random(seed_val + int(time.time() // 3))
        result_list: list[dict] = []
        chosen = rng.sample(list(ALL_CLASSES.keys()), rng.randint(1, 4))

        for class_id in chosen:
            info = ALL_CLASSES[class_id]
            margin = 30
            x1 = rng.randint(margin, max(margin + 1, w // 2))
            y1 = rng.randint(margin, max(margin + 1, h // 2))
            x2 = rng.randint(max(x1 + 1, w // 2), max(x1 + 2, w - margin))
            y2 = rng.randint(max(y1 + 1, h // 2), max(y1 + 2, h - margin))
            result_list.append(
                {
                    "class_id": class_id,
                    "class_name": info["name"],
                    "confidence": round(rng.uniform(0.55, 0.96), 3),
                    "bbox": [x1, y1, x2, y2],
                    "alert": info["alert"],
                    "color": info["color"],
                    "icon": info["icon"],
                    "source_model": "demo",
                },
            )

        return result_list

    def apply_cooldown(self, detections: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for detection in detections:
            if detection["alert"] and not self.cooldown.can_alert(detection["class_id"]):
                detection = detection.copy()
                detection["alert"] = False
            filtered.append(detection)
        return filtered

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        en_label_map = {
            "垃圾桶": "GarbageBin",
            "垃圾溢出": "Overflow",
            "散落垃圾": "Garbage",
            "火焰": "FIRE",
            "烟雾": "SMOKE",
            "未知目标": "Unknown",
        }

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box_color = detection.get("color", (0, 255, 0))
            en_name = en_label_map.get(detection["class_name"], detection["class_name"])
            label = f"{en_name} {detection['confidence']:.0%}"
            line_w = 3 if detection["alert"] else 2

            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, line_w)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            top = max(0, y1 - text_h - 8)
            cv2.rectangle(output, (x1, top), (x1 + text_w + 6, y1), box_color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 3, max(text_h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if detection["alert"]:
                cv2.putText(
                    output,
                    "! ALERT",
                    (x1, min(output.shape[0] - 10, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
        return output

    def analyze_scene(self, detections: list[dict]) -> dict:
        alert_list = [item for item in detections if item["alert"]]
        alert_types = list({item["class_name"] for item in alert_list})
        class_ids = {item["class_id"] for item in detections}

        status = "normal"
        if 3 in class_ids:
            status = "fire"
        elif 4 in class_ids:
            status = "smoke"
        elif 1 in class_ids:
            status = "overflow"
        elif alert_list:
            status = "warning"

        return {
            "status": status,
            "alert_count": len(alert_list),
            "alert_types": alert_types,
            "normal_count": len([item for item in detections if not item["alert"]]),
            "total": len(detections),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def build_response(self, image: np.ndarray, detections: list[dict], with_image: bool = True) -> dict:
        scene = self.analyze_scene(detections)
        result_image = None
        if with_image:
            rendered = self.draw_boxes(image, detections)
            result_image = frame_to_base64(rendered)
        return {"scene": scene, "detections": detections, "result_image": result_image}
