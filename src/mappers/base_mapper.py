"""Base mapper class defining the mapping interface."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseMapper(ABC):
    """Abstract base class for all mapping strategies."""
    
    def __init__(self, projections: Dict, tile_fidelity_model):
        """Initialize mapper with projections and fidelity model."""
        self.projections = projections
        self.fidelity_model = tile_fidelity_model
        self.current_mapping = {}
        self.mapping_history = []
        self.remapping_events = 0
    
    @abstractmethod
    def compute_mapping(self) -> Dict:
        """Compute and return a projection-to-tile mapping."""
        pass
    
    def apply_mapping(self, mapping: Dict) -> Dict:
        """Apply a computed mapping and track changes."""
        prev_mapping = self.current_mapping.copy()
        self.current_mapping = mapping
        self.mapping_history.append(mapping.copy())
        
        changes = self._count_changes(prev_mapping, mapping)
        if changes > 0:
            self.remapping_events += 1
        
        return {
            "mapping": mapping,
            "changes": changes,
            "prev_mapping": prev_mapping
        }
    
    def _count_changes(self, prev: Dict, new: Dict) -> int:
        """Count number of projection reassignments."""
        changes = 0
        for proj_id in new:
            if proj_id not in prev or prev[proj_id] != new[proj_id]:
                changes += 1
        return changes
    
    def get_stats(self) -> Dict:
        """Get mapping statistics."""
        return {
            "remapping_events": self.remapping_events,
            "current_mapping": self.current_mapping,
            "history_length": len(self.mapping_history)
        }
