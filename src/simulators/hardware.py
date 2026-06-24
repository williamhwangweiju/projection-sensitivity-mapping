"""Hardware configuration and modeling for 3D CiM accelerators."""
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class HardwareConfig:
    """Configuration for 3D CiM accelerator hardware."""
    num_tiles: int = 64
    tile_capacity_bytes: int = 1024 * 1024  # 1 MB per tile
    array_size: int = 256  # 256x256 array per tile
    
    # Timing and energy (estimated parameters)
    programming_bandwidth_gbps: float = 10.0
    programming_energy_per_cell_pj: float = 1.0  # picojoules
    communication_latency_ns: float = 100.0
    communication_energy_pj: float = 50.0
    
    # Device parameters
    conductance_drift_rate: float = 0.001  # per time step
    temperature_variation: float = 0.05
    aging_rate: float = 0.0005
    
    # Mapping parameters
    migration_threshold: float = 0.1  # Minimum sensitivity-weighted improvement
    remapping_cooldown: int = 10  # Time steps before allowing remapping
    
    tier_layout: Dict[str, int] = field(default_factory=lambda: {
        "HBM": 16,
        "L4": 32,
        "L3": 16
    })


class Tile:
    """Represents a single compute tile in the accelerator."""
    
    def __init__(self, tile_id: int, capacity_bytes: int):
        self.tile_id = tile_id
        self.capacity_bytes = capacity_bytes
        self.current_usage_bytes = 0
        self.fidelity = 1.0  # Normalized fidelity (1.0 = perfect)
        self.assignments = []  # List of (block_id, proj_name) tuples
    
    def add_assignment(self, block_id: str, proj_name: str, size_bytes: int) -> bool:
        """Try to assign a projection to this tile. Returns success."""
        if self.current_usage_bytes + size_bytes <= self.capacity_bytes:
            self.assignments.append((block_id, proj_name))
            self.current_usage_bytes += size_bytes
            return True
        return False
    
    def remove_assignment(self, block_id: str, proj_name: str):
        """Remove a projection assignment from this tile."""
        self.assignments = [a for a in self.assignments 
                           if not (a[0] == block_id and a[1] == proj_name)]
    
    def get_available_space(self) -> int:
        """Get available space in bytes."""
        return self.capacity_bytes - self.current_usage_bytes
    
    def degrade(self, degradation_rate: float):
        """Apply degradation to tile fidelity."""
        self.fidelity = max(0.0, self.fidelity - degradation_rate)
