from __future__ import annotations

from pathlib import Path

import cv2
import imageio

from app.services.detection_service import DetectionService


class VideoProcessingError(RuntimeError):
    pass


class VideoProcessingService:
    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service

    @staticmethod
    def _bgr_to_rgb(frame):
        # OpenCV uses BGR, while imageio/ffmpeg writer expects RGB arrays.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        skip_frames: int,
        progress_callback=None,
    ) -> dict:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoProcessingError("无法读取视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or fps > 120:
            fps = 30.0

        frame_count = 0
        total_detections = 0
        total_alerts = 0
        alert_frames = 0
        prev_result = None
        effective_skip = max(skip_frames, 1)

        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Detect on frame 1, 1+skip, 1+2*skip ...
                # When skip_frames=1, every frame is processed.
                if (frame_count - 1) % effective_skip != 0:
                    frame_to_write = prev_result if prev_result is not None else frame
                    writer.append_data(self._bgr_to_rgb(frame_to_write))
                    if progress_callback and total_frames:
                        progress_callback(frame_count, total_frames)
                    continue

                detections = self.detection_service.detect(frame)
                rendered = self.detection_service.draw_boxes(frame, detections)
                prev_result = rendered.copy()

                frame_alerts = sum(1 for item in detections if item.get("alert", False))
                if frame_alerts > 0:
                    alert_frames += 1
                total_alerts += frame_alerts
                total_detections += len(detections)

                cv2.putText(
                    rendered,
                    f"Frame {frame_count}: {len(detections)} detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                writer.append_data(self._bgr_to_rgb(rendered))

                if progress_callback and total_frames:
                    progress_callback(frame_count, total_frames)
        finally:
            writer.close()
            cap.release()

        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "video_info": f"{width}x{height}, {fps:.1f}fps",
        }
