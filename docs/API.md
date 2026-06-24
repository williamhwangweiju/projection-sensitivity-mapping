# API Documentation

## Core Modules

### src.models.GPT2Analyzer
Extract and analyze GPT-2 projection layers.

```python
from src.models import GPT2Analyzer

# Initialize analyzer
analyzer = GPT2Analyzer(model_name="gpt2")

# Get number of transformer blocks
num_blocks = analyzer.get_num_blocks()

# Get projection layer sizes (in bytes)
sizes = analyzer.get_projection_sizes()
# Returns: {"block_0": {"q_proj": 3686400, "out_proj": 1843200, ...}, ...}

# Access projections directly
projections = analyzer.projections
# Structure: {block_id: {proj_name: nn.Linear}}
```

---

### src.profilers.SensitivityProfiler
Measure hardware-noise sensitivity of projections.

```python
from src.profilers import SensitivityProfiler

# Initialize profiler
profiler = SensitivityProfiler(model, tokenizer, device="cuda")

# Profile a single projection
result = profiler.profile_projection(
    block_id="block_0",
    proj_name="q_proj",
    dataset=test_dataset,
    noise_std=0.05
)
# Returns: {
#   "block_id": "block_0",
#   "proj_name": "q_proj",
#   "ppl_clean": 42.5,
#   "ppl_noisy": [42.6, 43.1, 44.2, 45.8, 48.3],
#   "sensitivities": [0.0023, 0.0141, 0.0400, 0.0764, 0.1270]
# }
```

---

### src.simulators.HardwareConfig
Configure 3D CiM accelerator hardware parameters.

```python
from src.simulators import HardwareConfig

config = HardwareConfig(
    num_tiles=64,
    tile_capacity_bytes=1024*1024,
    programming_bandwidth_gbps=10.0,
    conductance_drift_rate=0.001
)
```

---

### src.simulators.TileFidelityModel
Simulate heterogeneous and time-varying tile fidelity.

```python
from src.simulators import TileFidelityModel, HardwareConfig

config = HardwareConfig()
fidelity_model = TileFidelityModel(config)

# Initialize tiles with three fidelity classes
# High (1.0), Medium (0.7), Low (0.4)

# Simulate one timestep with degradation
fidelity_model.step(degradation_scenario="gradual")

# Get fidelity of all tiles
fidelities = fidelity_model.get_fidelities()
# Returns: {0: 0.998, 1: 0.997, ...}

# Get tiles ranked by fidelity
ranked_tiles = fidelity_model.get_fidelity_ranks()
# Returns: [0, 2, 1, 3, ...] (tile IDs ranked by fidelity)

# Get tiles with fidelity above threshold
high_fidelity_tiles = fidelity_model.get_tile_by_fidelity(0.9)
# Returns: [Tile, Tile, ...]
```

---

### src.mappers.BaseMapper
Abstract base class for all mapping strategies.

```python
from src.mappers import BaseMapper

# Subclasses must implement compute_mapping()
# and inherit apply_mapping(), get_stats()

result = mapper.compute_mapping()
# Returns: {"block_0_q_proj": 0, "block_0_out_proj": 1, ...}

application = mapper.apply_mapping(result)
# Returns: {
#   "mapping": result,
#   "changes": 5,  # Number of reassignments
#   "prev_mapping": {...}
# }

stats = mapper.get_stats()
# Returns: {
#   "remapping_events": 3,
#   "current_mapping": {...},
#   "history_length": 15
# }
```

---

### src.mappers.RandomMapper
Randomly assign projections to tiles.

```python
from src.mappers import RandomMapper

mapper = RandomMapper(projections, fidelity_model)
mapping = mapper.compute_mapping()
```

---

### src.mappers.SequentialMapper
Sequentially assign projections to tiles (round-robin).

```python
from src.mappers import SequentialMapper

mapper = SequentialMapper(projections, fidelity_model)
mapping = mapper.compute_mapping()
```

---

### src.mappers.AdaptiveMapper
Migration-aware adaptive mapper with threshold-based remapping.

```python
from src.mappers import AdaptiveMapper

mapper = AdaptiveMapper(
    projections=projections,
    tile_fidelity_model=fidelity_model,
    sensitivities={"block_0_q_proj": 0.45, ...},  # Normalized scores
    projection_sizes={"block_0_q_proj": 3686400, ...}  # Bytes
)

# Initially computes greedy mapping
mapping = mapper.compute_mapping()

# Later calls check if remapping is justified
# (benefit > migration cost + threshold)
mapping = mapper.compute_mapping()

# Check if the latest mapping changed
if mapper.remapping_events > 0:
    print("Remapping occurred")
```

**Key method**: `_should_remap()`
- Compares current error to candidate mapping error
- Estimates migration cost
- Remaps only if: `improvement > max(threshold, migration_cost)`
- Enforces cooldown to avoid frequent oscillation

---

### src.utils.config
Load and save YAML configuration files.

```python
from src.utils import load_config, save_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Modify configuration
config["mapping"]["migration_threshold"] = 0.15

# Save modified configuration
save_config(config, "configs/modified_config.yaml")
```

---

### src.utils.logger
Set up structured logging.

```python
from src.utils import setup_logger

logger = setup_logger(
    name="experiment",
    log_dir="./logs",
    level=logging.INFO
)

logger.info("Experiment started")
logger.error("An error occurred")
```

---

## Typical Workflow

### Phase 1: Profile Sensitivities
```python
analyzer = GPT2Analyzer()
projections = analyzer.projections
profiler = SensitivityProfiler(model, tokenizer)

sensitivities = {}
for block_id, projs in projections.items():
    for proj_name in projs:
        result = profiler.profile_projection(block_id, proj_name, dataset)
        sensitivity = result["sensitivities"][-1]  # Peak sensitivity
        sensitivities[f"{block_id}_{proj_name}"] = sensitivity
```

### Phase 2: Set Up Hardware
```python
config = HardwareConfig(num_tiles=64)
fidelity_model = TileFidelityModel(config)
```

### Phase 3-4: Compare Mappings
```python
from src.mappers import RandomMapper, SequentialMapper, AdaptiveMapper

mappers = {
    "random": RandomMapper(projections, fidelity_model),
    "sequential": SequentialMapper(projections, fidelity_model),
    "adaptive": AdaptiveMapper(projections, fidelity_model, sensitivities, sizes)
}

for timestep in range(100):
    fidelity_model.step(degradation_scenario="gradual")
    
    for name, mapper in mappers.items():
        mapping = mapper.compute_mapping()
        # Evaluate mapping quality
```

---

## Data Structures

### Projection ID Format
`"{block_id}_{proj_name}"` where:
- `block_id`: "block_0", "block_1", ..., "block_11" (for GPT-2 Small)
- `proj_name`: "q_proj", "out_proj", "fc1", "fc2"
- Example: "block_0_q_proj"

### Mapping Format
Dictionary: `{proj_id: tile_id}`
- Keys: Projection identifiers (e.g., "block_0_q_proj")
- Values: Tile indices (0 to num_tiles-1)
- Example: `{"block_0_q_proj": 0, "block_0_out_proj": 1, ...}`

### Sensitivity Format
Dictionary: `{proj_id: normalized_sensitivity}`
- Values: Float in [0, 1], normalized across all projections
- Higher values = more sensitive to noise
- Example: `{"block_0_q_proj": 0.85, "block_0_out_proj": 0.32, ...}`
