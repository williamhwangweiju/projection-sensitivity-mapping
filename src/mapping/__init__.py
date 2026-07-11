"""Shared mapping representations and evaluation utilities."""

from .objective import (
    build_policy_summary,
    evaluate_placement_over_trace,
)
from .placement import (
    Placement,
    ShardPlacement,
)
from .projection_catalog import (
    MappedModuleSpec,
    ProjectionSpec,
)
from .sharding import (
    MappedShardRecord,
    ProjectionShard,
)

__all__ = [
    "MappedModuleSpec",
    "MappedShardRecord",
    "Placement",
    "ProjectionShard",
    "ProjectionSpec",
    "ShardPlacement",
    "build_policy_summary",
    "evaluate_placement_over_trace",
]