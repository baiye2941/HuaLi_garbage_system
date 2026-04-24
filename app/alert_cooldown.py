from __future__ import annotations

import warnings

from app.services.alert_policy_service import AlertPolicyService


class AlertCooldown:
    def __init__(self):
        warnings.warn(
            "app.alert_cooldown.AlertCooldown is deprecated; use app.services.alert_policy_service.AlertPolicyService instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._service = AlertPolicyService()

    def can_alert(self, category: str) -> bool:
        category_map = {
            "overflow": 1,
            "garbage": 2,
            "fire": 3,
            "smoke": 4,
        }
        class_id = category_map.get(category.lower())
        if class_id is None:
            return True
        return self._service.can_alert(class_id)

    def reset_category(self, category: str):
        category_map = {
            "overflow": 1,
            "garbage": 2,
            "fire": 3,
            "smoke": 4,
        }
        class_id = category_map.get(category.lower())
        if class_id is not None:
            self._service._last_alert_time.pop(class_id, None)


cooldown_manager = AlertCooldown()

__all__ = ["AlertCooldown", "cooldown_manager"]
