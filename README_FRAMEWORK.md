# Fidelity-Aware Adaptive Projection Mapping for GPT-2 on 3D Analog CiM Accelerators

A comprehensive simulation framework for evaluating adaptive mapping strategies for transformer inference on heterogeneous analog compute-in-memory hardware.

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify installation
python -c "from src.models import GPT2Analyzer; print('✓ Framework ready')"

# 3. Run Phase 1 experiment (skeleton)
python experiments/phase1_sensitivity/run_sensitivity_profile.py

# 4. Read documentation
cat docs/PROJECT_STRUCTURE.md
cat docs/PHASE_ROADMAP.md
cat docs/API.md
```

## Project Structure

```
src/                        Core framework
├── models/                 GPT-2 analysis
├── profilers/              Noise sensitivity profiling
├── mappers/                Mapping algorithms (6 strategies)
├── simulators/             Hardware simulation
└── utils/                  Configuration & logging

experiments/                5 phases of evaluation
├── phase1_sensitivity/     Profile projections
├── phase2_fidelity/        Validate tile degradation
├── phase3_baselines/       Static mapping baselines
├── phase4_adaptive/        Adaptive mapping with costs
└── phase5_evaluation/      Comprehensive comparison

docs/                       Documentation
├── PROJECT_STRUCTURE.md    Detailed organization
├── SETUP.md               Installation & config
├── PHASE_ROADMAP.md       5-phase research plan
└── API.md                 Module reference

configs/                   Configuration templates
data/                      Profiles & results
tests/                     Unit & integration tests
```

## Research Overview

### Central Question
Can projection-level sensitivity combined with time-varying tile-fidelity measurements preserve GPT-2 inference quality while limiting runtime remapping overhead?

### Key Innovation
A **migration-aware adaptive mapper** that:
1. Profiles hardware-noise sensitivity of each GPT-2 projection
2. Tracks time-varying tile fidelity (degradation, thermal, aging)
3. Assigns sensitive projections to reliable tiles
4. Remaps **only when the predicted accuracy gain justifies the migration cost**

### Five Phases

| Phase | Goal | Key Output |
|-------|------|-----------|
| 1 | Profile projection sensitivities | Sensitivity scores per projection |
| 2 | Model heterogeneous tile fidelity | Time-varying fidelity simulation |
| 3 | Establish baselines | 4 static mapping strategies |
| 4 | Develop adaptive algorithm | Migration-aware mapper |
| 5 | Comprehensive evaluation | Trade-off curves, recommendations |

## Core Concepts

### Projections
GPT-2 has 4 types of projections per transformer block:
- **q_proj**: Query-key-value attention projection
- **out_proj**: Attention output projection
- **fc1**: Feed-forward expansion
- **fc2**: Feed-forward contraction

Each has different sensitivity to hardware noise.

### Tiles
64 compute tiles with:
- Different initial fidelity classes (high/medium/low)
- Time-varying degradation (gradual drift, thermal, aging)
- Limited capacity for weights

### Mapping
Assigns projections to tiles to minimize:
- **Quality loss** = Σ(sensitivity × (1 - tile_fidelity))
- **Migration cost** = Σ(weights_moved × programming_cost)

### Adaptive Strategy
Remaps only when: `quality_improvement > max(threshold, migration_cost)`

## Module Highlights

### GPT2Analyzer
```python
analyzer = GPT2Analyzer("gpt2")
projections = analyzer.projections
sizes = analyzer.get_projection_sizes()
```

### SensitivityProfiler
```python
profiler = SensitivityProfiler(model, tokenizer)
result = profiler.profile_projection("block_0", "q_proj", dataset)
```

### TileFidelityModel
```python
fidelity_model = TileFidelityModel(config)
fidelity_model.step(degradation_scenario="gradual")
fidelities = fidelity_model.get_fidelities()
```

### Mappers (6 strategies)
```python
RandomMapper(projections, fidelity_model)
SequentialMapper(projections, fidelity_model)
StaticMapper(projections, fidelity_model)  # Hardware-aware baseline
AdaptiveMapper(projections, fidelity_model, sensitivities, sizes)
```

## Key Files

- **`src/mappers/adaptive_mapper.py`**: Core algorithm with migration cost
- **`src/simulators/tile_fidelity.py`**: Hardware degradation simulation
- **`src/profilers/sensitivity_profiler.py`**: Projection profiling
- **`configs/default_config.yaml`**: Experiment parameters
- **`docs/PHASE_ROADMAP.md`**: Detailed 5-phase plan
- **`docs/API.md`**: Complete module reference

## Configuration

Edit `configs/default_config.yaml` to customize:
- Model size (gpt2, distilgpt2)
- Hardware (num_tiles, capacity, degradation)
- Mapping strategy
- Evaluation metrics

```yaml
mapping:
  migration_threshold: 0.1        # Min quality improvement to justify remapping
  remapping_cooldown: 10          # Time steps between remapping decisions
  strategy: "adaptive"            # Options: random, sequential, adaptive

experiment:
  degradation_scenario: "gradual" # Options: gradual, localized, thermal
  num_timesteps: 100
```

## Expected Contributions

1. **Projection Sensitivity Profile**: GPT-2 under analog hardware noise
2. **Tile Fidelity Model**: Time-varying heterogeneous hardware
3. **Capacity-Aware Sharding**: Handle projections larger than tiles
4. **Migration-Aware Mapper**: Balance quality vs. remapping overhead
5. **Trade-Off Curves**: Quality vs. migration cost analysis
6. **Reproducible Framework**: AIHWKit + Python + 3D-CiM simulator

## References

### Related Work
- Projection-level sensitivity: [Prior work on heterogeneous mapping]
- Dynamic mapping: [CiM fault-aware mapping studies]
- Migration cost: [Energy & latency of weight transfer]

### Tools
- **AIHWKit**: Analog neural network simulation with device models
- **Transformers**: Hugging Face model loading
- **PyTorch**: Deep learning framework

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Read Phase Roadmap**: `cat docs/PHASE_ROADMAP.md`
3. **Review API**: `cat docs/API.md`
4. **Implement Phase 1**: Start with sensitivity profiling
5. **Run experiments**: Execute phase scripts sequentially

## Timeline

Estimated 2-3 months with part-time effort:
- Weeks 1-3: Phase 1 (profiling)
- Week 4: Phase 2 (fidelity model)
- Week 5: Phase 3 (baselines)
- Weeks 6-7: Phase 4 (adaptive mapper)
- Weeks 8-10: Phase 5 (evaluation & writing)

## Contact & Contributions

For questions, bug reports, or contributions, open an issue or reach out.

---

**Status**: Framework complete. Ready for Phase 1 implementation. ✓
