"""Data transforms that inject RECAP advantage tags into the language prompt."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from openpi.recap.advantages import load_advantage_lookup
from openpi.recap.routing import compute_cfg_routing_masks
from openpi.recap.tags import build_acp_tagged_task
from openpi.transforms import DataDict, DataTransformFn, compose


def _as_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _lookup_nested(data: DataDict, key: str) -> Any:
    if key in data:
        return data[key]
    if "/" in key:
        node: Any = data
        for part in key.split("/"):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
    if "." in key:
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
    return None


_INTERVENTION_KEYS = (
    "is_intervention",
    "intervention",
    "complementary_info.is_intervention",
    "complementary_info/is_intervention",
)


@dataclasses.dataclass(frozen=True)
class KeepFields(DataTransformFn):
    """Re-attach fields that ``RepackTransform`` would otherwise drop."""

    transform: DataTransformFn
    fields: tuple[str, ...]

    def __call__(self, data: DataDict) -> DataDict:
        saved = {key: data[key] for key in self.fields if key in data}
        out = self.transform(data)
        for key, value in saved.items():
            if key not in out:
                out[key] = value
        return out


@dataclasses.dataclass(frozen=True)
class LoadAdvantageLabel(DataTransformFn):
    """Attach a boolean ``advantage`` flag from a sidecar or in-sample column."""

    path: str | None = None
    column: str = "advantage"
    force_intervention_positive: bool = True
    positive_fraction: float = 0.3

    def __post_init__(self) -> None:
        lookup: dict[tuple[int, int], bool] = {}
        if self.path:
            lookup = load_advantage_lookup(self.path, positive_fraction=self.positive_fraction)
        object.__setattr__(self, "_lookup", lookup)

    def __call__(self, data: DataDict) -> DataDict:
        advantage = None
        lookup: dict[tuple[int, int], bool] = getattr(self, "_lookup")
        if lookup:
            if "episode_index" not in data or "frame_index" not in data:
                raise KeyError(
                    "RECAP sidecar lookup requires 'episode_index' and 'frame_index' in the raw sample."
                )
            key = (int(_as_python_scalar(data["episode_index"])), int(_as_python_scalar(data["frame_index"])))
            if key not in lookup:
                raise KeyError(f"No RECAP advantage label for episode/frame {key} in {self.path}")
            advantage = lookup[key]
        else:
            raw = _lookup_nested(data, self.column)
            if raw is None:
                for fallback in ("advantage", "complementary_info.acp_indicator", "acp_indicator"):
                    raw = _lookup_nested(data, fallback)
                    if raw is not None:
                        break
            if raw is None:
                raise KeyError(
                    "RECAP enabled but no advantage sidecar/column found. "
                    "Run scripts/recap/compute_advantages.py or set recap.advantage_path."
                )
            advantage = bool(_as_python_scalar(raw))

        if self.force_intervention_positive:
            for key in _INTERVENTION_KEYS:
                raw = _lookup_nested(data, key)
                if raw is not None and float(_as_python_scalar(raw)) > 0.5:
                    advantage = True
                    break

        data["advantage"] = np.asarray(bool(advantage))
        return data


@dataclasses.dataclass(frozen=True)
class InjectAdvantagePrompt(DataTransformFn):
    """Append ACP tags to ``prompt`` using RLinf positive-only routing."""

    positive_only_conditional: bool = True
    unconditional_prob: float = 0.1
    seed: int = 0

    def __call__(self, data: DataDict) -> DataDict:
        if "advantage" not in data:
            raise KeyError("InjectAdvantagePrompt requires 'advantage' (run LoadAdvantageLabel first).")
        if "prompt" not in data:
            raise KeyError("InjectAdvantagePrompt requires 'prompt'.")

        prompt = data["prompt"]
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

        is_positive = bool(_as_python_scalar(data["advantage"]))
        index = int(_as_python_scalar(data["index"])) if "index" in data else 0
        rng = np.random.default_rng(self.seed + index)
        routing = compute_cfg_routing_masks(
            np.asarray([is_positive]),
            positive_only_conditional=self.positive_only_conditional,
            unconditional_prob=self.unconditional_prob,
            rng=rng,
        )
        if routing["positive_conditional_mask"][0]:
            prompt = build_acp_tagged_task(prompt, is_positive=True)
        elif routing["negative_conditional_mask"][0]:
            prompt = build_acp_tagged_task(prompt, is_positive=False)

        data["prompt"] = np.asarray(prompt)
        data.pop("advantage", None)
        return data


def keep_fields(transform: DataTransformFn | list[DataTransformFn], fields: tuple[str, ...]) -> KeepFields:
    if isinstance(transform, list):
        transform = compose(transform)
    return KeepFields(transform=transform, fields=fields)
