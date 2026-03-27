from __future__ import annotations

from .base import BaseEvalAdapter


def _resolve_adapter_class(name: str):
    if name in {"showo", "showo_weighted"}:
        from .showo_weighted import ShowoWeightedAdapter

        return ShowoWeightedAdapter
    if name == "bagel_tp":
        from .bagel_tp import BagelTPAdapter

        return BagelTPAdapter
    if name == "bagel_ip":
        from .bagel_ip import BagelIPAdapter

        return BagelIPAdapter
    if name == "janus_tp":
        from .janus_tp import JanusTPAdapter

        return JanusTPAdapter
    if name == "janus_ip":
        from .janus_ip import JanusIPAdapter

        return JanusIPAdapter
    raise ValueError(f"Unknown adapter: {name}")


def build_adapter(name: str, **kwargs) -> BaseEvalAdapter:
    adapter_class = _resolve_adapter_class(name)
    return adapter_class(**kwargs)

__all__ = ["BaseEvalAdapter", "build_adapter"]
