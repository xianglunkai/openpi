"""Minimal teleoperator interface (aligned with lerobot Teleoperator)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class TeleoperatorConfig:
    """Base teleop configuration."""

    id: str = "default"


class Teleoperator(abc.ABC):
    """Base class for teleoperation devices."""

    name: str = "teleoperator"

    def __init__(self, config: TeleoperatorConfig | None = None) -> None:
        self.config = config or TeleoperatorConfig()

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        pass

    @abc.abstractmethod
    def configure(self) -> None:
        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
