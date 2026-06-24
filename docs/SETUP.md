# Setup Instructions

## Environment Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install IBM Analog Hardware Acceleration Kit (Optional)
For Phase 1 (sensitivity profiling) with realistic analog noise models:
```bash
pip install -r requirements-ai-hw-kit.txt
```

## Project Organization

- **src/**: Core framework code (models, profilers, mappers, simulators)
- **experiments/**: Phase-specific experiments
- **configs/**: Configuration templates
- **data/**: Profiles and results
- **tests/**: Unit and integration tests
- **docs/**: Documentation

## Quick Start

### 1. Run a Simple Test
```bash
python -c "from src.models import GPT2Analyzer; print(GPT2Analyzer('gpt2').get_num_blocks())"
```

### 2. Run Phase 1 Experiment
```bash
python experiments/phase1_sensitivity/run_sensitivity_profile.py
```

### 3. View Results
Results are saved to `data/results/` with timestamps.
Use provided analysis scripts in `scripts/` to visualize results.

## Configuration

Edit `configs/default_config.yaml` to modify:
- Model size and dataset
- Hardware parameters
- Mapping strategy
- Degradation scenario
- Evaluation metrics

Pass custom config to experiments:
```bash
python experiments/phase1_sensitivity/run_sensitivity_profile.py --config configs/custom_config.yaml
```

## Troubleshooting

- **AIHWKit not installing**: AIHWKit requires a C++ compiler. On macOS, install Xcode Command Line Tools.
- **Out of memory**: Reduce batch size or use a smaller model (distilgpt2).
- **Slow profiling**: Use fewer noise levels or a smaller dataset for testing.
