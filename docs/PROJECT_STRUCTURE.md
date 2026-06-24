# Project Structure

## Fidelity-Aware Adaptive Projection Mapping for GPT-2

```
src/
├── __init__.py
├── models/                      # GPT-2 model utilities
│   ├── __init__.py
│   └── gpt2_model.py           # GPT2Analyzer: extract and analyze projections
│
├── profilers/                   # Sensitivity profiling
│   ├── __init__.py
│   └── sensitivity_profiler.py # SensitivityProfiler: measure noise sensitivity
│
├── mappers/                     # Mapping algorithms
│   ├── __init__.py
│   ├── base_mapper.py          # BaseMapper: abstract interface
│   ├── static_mapper.py        # StaticMapper, RandomMapper, SequentialMapper
│   └── adaptive_mapper.py      # AdaptiveMapper: threshold-based dynamic mapping
│
├── simulators/                  # Hardware simulation
│   ├── __init__.py
│   ├── hardware.py             # HardwareConfig, Tile classes
│   └── tile_fidelity.py        # TileFidelityModel: time-varying fidelity
│
└── utils/                       # Utilities
    ├── __init__.py
    ├── config.py               # Configuration loading/saving
    └── logger.py               # Logging setup

experiments/                     # Phase-specific experiments
├── phase1_sensitivity/         # Projection sensitivity profiling
├── phase2_fidelity/            # Tile fidelity model validation
├── phase3_baselines/           # Static mapping baselines
├── phase4_adaptive/            # Adaptive mapping with cost awareness
└── phase5_evaluation/          # Comprehensive evaluation

configs/                        # Configuration files
└── default_config.yaml        # Default experiment configuration

data/                           # Data storage
├── profiles/                   # Sensitivity profiles
└── results/                    # Evaluation results

tests/                          # Unit and integration tests
├── unit/                       # Unit tests for individual components
└── integration/                # Integration tests across phases

docs/                           # Documentation
├── PROJECT_STRUCTURE.md
├── SETUP.md
├── PHASE_ROADMAP.md
└── API.md

scripts/                        # Utility scripts
```

## Module Responsibilities

### src/models
- **GPT2Analyzer**: Load pretrained models, extract projection layers, compute sizes
- Input: Model name, dataset
- Output: Projection dictionary, layer metadata

### src/profilers
- **SensitivityProfiler**: Inject controlled noise, measure perplexity/KL divergence
- Input: Model, dataset, noise parameters
- Output: Per-projection sensitivity scores

### src/mappers
- **BaseMapper**: Abstract interface with apply_mapping, get_stats
- **StaticMapper variants**: Random, sequential, hardware-aware, static-sensitive
- **AdaptiveMapper**: Greedy assignment with threshold-based remapping
- Input: Projections, tile fidelity, sensitivities, sizes
- Output: Projection-to-tile mapping

### src/simulators
- **HardwareConfig**: Tile capacity, programming costs, device parameters
- **Tile**: Individual tile state, capacity tracking, fidelity
- **TileFidelityModel**: Collection of tiles, degradation simulation, fidelity queries
- Input: Hardware parameters, degradation scenario
- Output: Tile fidelity at each timestep

### src/utils
- **config.py**: YAML-based configuration management
- **logger.py**: Structured logging to file and console

## Phase Organization

Each phase has its own experiment directory with:
- Main experiment script (`run_*.py`)
- Phase-specific configuration
- Results and logs
- Documentation of findings
