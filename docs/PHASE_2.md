# Phase 2 Fidelity Simulation Components

## `src/simulators/hardware.py`

Defines the structural representation of the simulated hardware.

Main responsibilities:

* Defines `HardwareConfig`, including the number of tiles, tiers per tile, tile dimensions, and thermal zones.
* Defines `TileState`, which stores each tile’s baseline noise, current noise, fidelity class, drift rate, thermal zone, availability, and fault status.
* Defines `HardwareState`, a validated collection of all tile states.
* Validates hardware dimensions, tile identifiers, thermal-zone assignments, and noise values.
* Provides helper methods for cloning, resetting, serializing, and accessing tile states.

This file does not simulate degradation or update tile fidelity over time. It only describes the hardware configuration and current tile state.

---

## `src/simulators/tile_fidelity.py`

Implements the heterogeneous and time-varying tile-fidelity simulation.

Main responsibilities:

* Parses and validates Phase 2 fidelity configuration.
* Initializes tiles into high-, medium-, and low-fidelity classes.
* Samples baseline noise values from class-specific ranges.
* Assigns tile-specific gradual drift rates.
* Models correlated thermal fluctuations using an AR(1) process.
* Schedules sudden localized faults for selected tiles.
* Updates each tile’s effective noise at every timestep.
* Converts noise into descriptive fidelity scores and dynamic fidelity classes.
* Generates complete trace arrays with shape:

```text
[num_timesteps, num_tiles]
```

The effective tile noise is modeled as:

[
\sigma_i(t)
===========

\operatorname{clip}
\left[
\sigma_{i,0}
\left(
1+
D_i(t)+
H_{z(i)}(t)+
L_i(t)
\right)
\right]
]

where:

* (\sigma_{i,0}) is the tile’s baseline noise.
* (D_i(t)) is gradual drift.
* (H_{z(i)}(t)) is thermal-zone variation.
* (L_i(t)) is localized fault degradation.

The primary hardware-quality value is `current_noise_std`. Fidelity scores and high/medium/low labels are mainly used for interpretation and visualization.

The generated trace includes:

* Effective noise per tile and timestep
* Fidelity scores
* Dynamic fidelity classes
* Tile availability
* Fault status
* Baseline noise
* Drift rates
* Thermal-zone assignments
* Fault onset times
* Fault severity

---

## `experiments/phase2_fidelity/run_fidelity_model.py`

Provides the command-line entry point for running Phase 2 experiments.

Main responsibilities:

* Loads the Phase 2 YAML configuration.
* Applies optional command-line seed and output-directory overrides.
* Constructs `TileFidelityModel`.
* Generates the complete fidelity trace.
* Saves machine-readable and human-readable results.
* Computes aggregate statistics for tiles and timesteps.
* Reports initial and final noise, fidelity, faults, and tile-ranking changes.

Example command:

```bash
python experiments/phase2_fidelity/run_fidelity_model.py \
    --config configs/phase2_fidelity/mixed.yaml
```

Generated outputs include:

```text
data/fidelity_traces/<experiment_name>/seed_<seed>/
├── trace.npz
├── config.yaml
├── metadata.json
├── tile_summary.csv
└── timestep_summary.csv
```

* `trace.npz`: complete arrays consumed by later mapping phases.
* `config.yaml`: effective configuration used for the run.
* `metadata.json`: experiment metadata and aggregate results.
* `tile_summary.csv`: one summary row per tile.
* `timestep_summary.csv`: aggregate hardware statistics per timestep.

Together, these files establish a reproducible Phase 2 pipeline that produces heterogeneous, time-varying hardware-fidelity traces for the static and adaptive mapping algorithms used in later phases.
