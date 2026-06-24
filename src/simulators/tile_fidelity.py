"""Tile fidelity and degradation model."""
import numpy as np
from typing import List, Dict
from .hardware import Tile, HardwareConfig


class TileFidelityModel:
    """Models time-varying tile fidelity for heterogeneous hardware."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.tiles = [Tile(i, config.tile_capacity_bytes) 
                      for i in range(config.num_tiles)]
        self.time_step = 0
        self._initialize_fidelity()
    
    def _initialize_fidelity(self):
        """Initialize tiles with different fidelity levels."""
        num_high = self.config.num_tiles // 3
        num_med = self.config.num_tiles // 3
        num_low = self.config.num_tiles - num_high - num_med
        
        for i, tile in enumerate(self.tiles):
            if i < num_high:
                tile.fidelity = 1.0
            elif i < num_high + num_med:
                tile.fidelity = 0.7
            else:
                tile.fidelity = 0.4
    
    def step(self, degradation_scenario: str = "gradual"):
        """Advance time and update tile fidelities."""
        self.time_step += 1
        
        if degradation_scenario == "gradual":
            for tile in self.tiles:
                tile.degrade(self.config.conductance_drift_rate)
        
        elif degradation_scenario == "localized":
            if self.time_step == 50:
                for i in range(10, 15):
                    self.tiles[i].fidelity = 0.1
        
        elif degradation_scenario == "thermal":
            for tile in self.tiles:
                thermal_noise = np.random.normal(0, self.config.temperature_variation)
                tile.fidelity = max(0.0, tile.fidelity + thermal_noise)
    
    def get_tile_by_fidelity(self, fidelity_threshold: float) -> List[Tile]:
        """Get tiles with fidelity above threshold."""
        return [t for t in self.tiles if t.fidelity >= fidelity_threshold]
    
    def get_fidelity_ranks(self) -> List[int]:
        """Get tile indices ranked by fidelity (descending)."""
        return sorted(range(len(self.tiles)), 
                     key=lambda i: self.tiles[i].fidelity, 
                     reverse=True)
    
    def get_fidelities(self) -> Dict[int, float]:
        """Get fidelity of all tiles."""
        return {i: tile.fidelity for i, tile in enumerate(self.tiles)}
