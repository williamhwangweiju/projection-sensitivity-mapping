"""Static mapping strategies (baselines)."""
import random
from typing import Dict
from .base_mapper import BaseMapper


class StaticMapper(BaseMapper):
    """Base class for static mappers that don't adapt over time."""
    
    def compute_mapping(self) -> Dict:
        """Compute mapping (does not change after initial assignment)."""
        if not self.current_mapping:
            self.current_mapping = self._initial_mapping()
        return self.current_mapping
    
    def _initial_mapping(self) -> Dict:
        """Compute initial mapping (implemented by subclasses)."""
        raise NotImplementedError


class RandomMapper(StaticMapper):
    """Randomly assign projections to tiles."""
    
    def _initial_mapping(self) -> Dict:
        """Random assignment."""
        mapping = {}
        tiles = self.fidelity_model.tiles
        
        proj_list = [
            (block_id, proj_name)
            for block_id in self.projections
            for proj_name in self.projections[block_id]
        ]
        
        for block_id, proj_name in proj_list:
            tile = random.choice(tiles)
            proj_id = f"{block_id}_{proj_name}"
            mapping[proj_id] = tile.tile_id
        
        return mapping


class SequentialMapper(StaticMapper):
    """Sequentially assign projections to tiles."""
    
    def _initial_mapping(self) -> Dict:
        """Sequential tile assignment."""
        mapping = {}
        tiles = self.fidelity_model.tiles
        tile_idx = 0
        
        for block_id in self.projections:
            for proj_name in self.projections[block_id]:
                proj_id = f"{block_id}_{proj_name}"
                mapping[proj_id] = tiles[tile_idx % len(tiles)].tile_id
                tile_idx += 1
        
        return mapping
