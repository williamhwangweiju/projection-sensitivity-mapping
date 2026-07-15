"""Phase-3 mapping interfaces."""

from .objective import build_policy_summary, evaluate_placement_over_trace
from .placement import (
    Placement,
    ShardPlacement,
    build_placement_from_mapped_shards,
    mapped_shards_with_placement_to_rows,
)
from .projection_catalog import (
    MappedModuleSpec,
    ProjectionSpec,
    build_group_total_weights,
    build_mapped_module_specs,
    iter_mappable_modules,
    load_phase1_sensitivity_lookup,
    mapped_module_specs_to_rows,
    module_projection_metadata,
    order_modules_for_policy,
)
from .sharding import (
    MappedShardRecord,
    ProjectionShard,
    extract_shards_from_3dcim_mapping,
    mapped_shard_records_to_placement_rows,
)

__all__ = [
    "MappedModuleSpec",
    "MappedShardRecord",
    "Placement",
    "ProjectionShard",
    "ProjectionSpec",
    "ShardPlacement",
    "build_group_total_weights",
    "build_mapped_module_specs",
    "build_placement_from_mapped_shards",
    "build_policy_summary",
    "evaluate_placement_over_trace",
    "extract_shards_from_3dcim_mapping",
    "iter_mappable_modules",
    "load_phase1_sensitivity_lookup",
    "mapped_module_specs_to_rows",
    "mapped_shard_records_to_placement_rows",
    "mapped_shards_with_placement_to_rows",
    "module_projection_metadata",
    "order_modules_for_policy",
]
