"""Adaptive mapping strategies that respond to changing tile fidelity."""
from typing import Dict
import numpy as np
from .base_mapper import BaseMapper


class AdaptiveMapper(BaseMapper):
    """
    Adaptive mapper that responds to changing tile fidelity.
    Considers both model sensitivity and migration costs.
    """
    
    def __init__(self, projections: Dict, tile_fidelity_model, 
                 sensitivities: Dict = None, projection_sizes: Dict = None):
        """
        Initialize adaptive mapper.
        
        Args:
            projections: Dictionary of GPT-2 projections
            tile_fidelity_model: Fidelity model for tiles
            sensitivities: Projection sensitivity scores
            projection_sizes: Size of each projection in bytes
        """
        super().__init__(projections, tile_fidelity_model)
        self.sensitivities = sensitivities or {}
        self.projection_sizes = projection_sizes or {}
        self.last_remap_time = -float('inf')
    
    def compute_mapping(self) -> Dict:
        """Compute adaptive mapping considering fidelity and migration cost."""
        if not self.current_mapping:
            # Initial mapping
            return self._compute_greedy_mapping()
        
        # Check if remapping is justified
        if self._should_remap():
            new_mapping = self._compute_greedy_mapping()
            return new_mapping
        
        return self.current_mapping
    
    def _should_remap(self) -> bool:
        """Check if the expected benefit of remapping exceeds migration cost."""
        if self.fidelity_model.time_step - self.last_remap_time < \
           self.fidelity_model.config.remapping_cooldown:
            return False
        
        current_error = self._compute_sensitivity_weighted_error(self.current_mapping)
        candidate_mapping = self._compute_greedy_mapping()
        candidate_error = self._compute_sensitivity_weighted_error(candidate_mapping)
        
        improvement = (current_error - candidate_error) / max(current_error, 1e-6)
        migration_cost_estimate = self._estimate_migration_cost(
            self.current_mapping, candidate_mapping
        )
        
        threshold = self.fidelity_model.config.migration_threshold
        return improvement > threshold and improvement > migration_cost_estimate
    
    def _compute_greedy_mapping(self) -> Dict:
        """Compute greedy mapping: assign sensitive projections to good tiles."""
        mapping = {}
        tiles = self.fidelity_model.tiles
        
        # Sort projections by sensitivity (descending)
        proj_list = [
            (block_id, proj_name)
            for block_id in self.projections
            for proj_name in self.projections[block_id]
        ]
        
        proj_list.sort(
            key=lambda p: self.sensitivities.get(f"{p[0]}_{p[1]}", 0.0),
            reverse=True
        )
        
        # Sort tiles by fidelity (descending)
        tile_order = sorted(range(len(tiles)), 
                           key=lambda i: tiles[i].fidelity, 
                           reverse=True)
        
        # Greedy assignment
        tile_idx = 0
        for block_id, proj_name in proj_list:
            proj_id = f"{block_id}_{proj_name}"
            mapping[proj_id] = tiles[tile_order[tile_idx % len(tile_order)]].tile_id
            tile_idx += 1
        
        return mapping
    
    def _compute_sensitivity_weighted_error(self, mapping: Dict) -> float:
        """Compute total sensitivity-weighted error for a mapping."""
        total_error = 0.0
        tiles = self.fidelity_model.tiles
        
        for proj_id, tile_id in mapping.items():
            sensitivity = self.sensitivities.get(proj_id, 0.0)
            tile_fidelity = tiles[tile_id].fidelity
            error = sensitivity * (1.0 - tile_fidelity)
            total_error += error
        
        return total_error
    
    def _estimate_migration_cost(self, old_mapping: Dict, new_mapping: Dict) -> float:
        """Estimate normalized cost of migrating to new mapping."""
        weight_moved = 0
        for proj_id in new_mapping:
            if proj_id not in old_mapping or old_mapping[proj_id] != new_mapping[proj_id]:
                weight_moved += self.projection_sizes.get(proj_id, 0)
        
        total_weight = sum(self.projection_sizes.values()) or 1.0
        migration_cost_norm = weight_moved / total_weight
        
        return migration_cost_norm
