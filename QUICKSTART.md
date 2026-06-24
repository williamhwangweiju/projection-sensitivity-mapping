# Quick Start Guide

## Setup (5 minutes)
```bash
cd /Users/hwangweiju/ML_HW_Reasearch.worktrees/agents-adaptive-mapping-gpt2-3d-accelerators-b78f2db5
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify
python -c "from src.models import GPT2Analyzer; print('✓ Framework ready')"
```

## Framework Tour (15 minutes)
```bash
# Read the main overview
cat README_FRAMEWORK.md

# Understand the project structure
cat docs/PROJECT_STRUCTURE.md

# Review the 5-phase roadmap
cat docs/PHASE_ROADMAP.md

# Check the API reference
cat docs/API.md
```

## Core Classes (at a glance)

### 1. Load Model & Extract Projections
```python
from src.models import GPT2Analyzer

analyzer = GPT2Analyzer("gpt2")
projections = analyzer.projections
sizes = analyzer.get_projection_sizes()
```

### 2. Create Hardware Model
```python
from src.simulators import HardwareConfig, TileFidelityModel

config = HardwareConfig(num_tiles=64)
fidelity = TileFidelityModel(config)
fidelity.step(degradation_scenario="gradual")
```

### 3. Profile Sensitivities (Phase 1)
```python
from src.profilers import SensitivityProfiler

profiler = SensitivityProfiler(model, tokenizer)
result = profiler.profile_projection("block_0", "q_proj", dataset)
# Returns: sensitivities at different noise levels
```

### 4. Compare Mapping Strategies
```python
from src.mappers import RandomMapper, AdaptiveMapper

random = RandomMapper(projections, fidelity)
adaptive = AdaptiveMapper(projections, fidelity, sensitivities, sizes)

# Get mappings
r_map = random.compute_mapping()
a_map = adaptive.compute_mapping()
```

## Project Structure at a Glance
```
src/
  ├── models/          → GPT2Analyzer
  ├── profilers/       → SensitivityProfiler
  ├── simulators/      → HardwareConfig, TileFidelityModel
  ├── mappers/         → BaseMapper, RandomMapper, SequentialMapper, AdaptiveMapper
  └── utils/           → config, logger

experiments/
  ├── phase1_sensitivity/
  ├── phase2_fidelity/
  ├── phase3_baselines/
  ├── phase4_adaptive/
  └── phase5_evaluation/

configs/
  └── default_config.yaml

docs/
  ├── PROJECT_STRUCTURE.md
  ├── SETUP.md
  ├── PHASE_ROADMAP.md
  └── API.md
```

## Configuration (one file)
Edit `configs/default_config.yaml`:
```yaml
# Model
model:
  name: "gpt2"  # or distilgpt2

# Hardware
hardware:
  num_tiles: 64
  
# Mapping
mapping:
  migration_threshold: 0.1      # Min quality improvement
  remapping_cooldown: 10         # Time steps before next remap
  strategy: "adaptive"           # random | sequential | adaptive

# Experiment
experiment:
  degradation_scenario: "gradual"  # gradual | localized | thermal
  num_timesteps: 100
```

## Run Phase 1 (Sensitivity Profiling)
```bash
cd experiments/phase1_sensitivity
python run_sensitivity_profile.py --config ../../configs/default_config.yaml
```

Expected output:
- Sensitivity scores for each projection
- Perplexity degradation curves
- Profiles saved to `data/profiles/`

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Projection** | A linear layer (query, attention output, feed-forward) |
| **Sensitivity** | How much a projection's output degrades under noise |
| **Tile** | A compute unit with limited capacity and fidelity |
| **Fidelity** | Probability of correct computation (1.0 = perfect) |
| **Mapping** | Assignment of projections to tiles |
| **Migration Cost** | Energy/latency overhead of moving weights |

## The Core Algorithm (AdaptiveMapper)

**Decision**: Should we remap now?

1. Compute current error: `E = Σ(sensitivity × (1 - fidelity))`
2. Generate candidate mapping: greedy (sensitive→good tiles)
3. Compute candidate error: `E' = Σ(sensitivity × (1 - fidelity))`
4. Compute improvement: `improvement = (E - E') / E`
5. Estimate migration cost: `cost = weight_moved / total_weight`
6. **Remap if**: `improvement > max(threshold, cost)` AND `cooldown_elapsed`

## Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| 1 | 2-3 weeks | Profile sensitivities |
| 2 | 1 week | Validate fidelity model |
| 3 | 1 week | Baseline mappers |
| 4 | 2 weeks | Adaptive mapping |
| 5 | 2-3 weeks | Evaluation & writing |
| **Total** | **2-3 months** | Complete framework |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Verify `pip install -r requirements.txt` |
| AIHWKit install fails | Install Xcode Command Line Tools (macOS) |
| Out of memory | Reduce batch size or use distilgpt2 |
| Slow profiling | Use fewer noise levels or smaller dataset |

## Key Files to Edit

1. **configs/default_config.yaml** - Change experiment parameters
2. **experiments/phase{1-5}_*/run_*.py** - Implement each phase
3. **src/mappers/adaptive_mapper.py** - Modify mapping algorithm
4. **src/simulators/tile_fidelity.py** - Change degradation model

## Getting Help

- **API Reference**: `docs/API.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Phase Details**: `docs/PHASE_ROADMAP.md`
- **Installation**: `docs/SETUP.md`

## Success Criteria

✓ Phase 1: Sensitivities show clear projection differences  
✓ Phase 2: Fidelity model matches degradation patterns  
✓ Phase 3: Baselines establish performance bounds  
✓ Phase 4: Adaptive mapper reduces overhead vs. naive  
✓ Phase 5: Clear evidence projection sensitivity guides mapping  

---

**Next Step**: `cat docs/PHASE_ROADMAP.md` to understand the research plan.
