from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app.core.constants import ALL_CLASSES
from app.infrastructure.ml.model_registry import ModelRegistry


class InferenceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        # Persistent executor — avoid per-call thread pool creation overhead.
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self, n_workers: int) -> ThreadPoolExecutor:
        if self._executor is None or self._executor._max_workers != n_workers:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=n_workers)
        return self._executor

    def detect(self, image: np.ndarray) -> list[dict]:
        bundles = [
            b for b in self.registry.items()
            if b.backend is not None and b.backend.loaded
        ]
        if not bundles:
            return []
        if len(bundles) == 1:
            return self._run_bundle(bundles[0], image)

        # Run multiple models in parallel — each model owns its own thread-safe
        # backend session, so concurrent predict() calls are safe.
        results: list[dict] = []
        executor = self._get_executor(len(bundles))
        futures = {
            executor.submit(self._run_bundle, bundle, image): bundle
            for bundle in bundles
        }
        for future in as_completed(futures):
            results.extend(future.result())
        return results

    def _run_bundle(self, bundle, image: np.ndarray) -> list[dict]:
        predictions = bundle.backend.predict(
            image=image,
            conf_threshold=0.5,
            iou_threshold=0.3,
        )
        out: list[dict] = []
        for pred in predictions:
            class_id = bundle.descriptor.class_mapping.get(pred.class_id, pred.class_id)
            info = ALL_CLASSES.get(class_id, {})
            out.append(
                {
                    "class_id": class_id,
                    "class_name": info.get("name", "unknown"),
                    "confidence": pred.confidence,
                    "bbox": pred.bbox,
                    "alert": info.get("alert", False),
                    "color": info.get("color", (0, 255, 0)),
                    "icon": info.get("icon", ""),
                    "source_model": bundle.descriptor.key,
                }
            )
        return out
