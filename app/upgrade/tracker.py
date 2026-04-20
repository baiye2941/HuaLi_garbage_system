from __future__ import annotations

from dataclasses import dataclass
from itertools import count


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


class TrackEngine:
    """Simple tracker scaffold.

    Placeholder implementation assigns incremental IDs.
    You can replace `update()` internals with ByteTrack later.
    """

    def __init__(self):
        self._id_gen = count(1)

    def update(self, detections) -> list[Track]:
        tracks: list[Track] = []
        for det in detections:
            tracks.append(
                Track(
                    track_id=next(self._id_gen),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
        return tracks
