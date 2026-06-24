# Phase Roadmap

## Phase 1: Projection-Sensitivity Profiling
**Goal**: Establish projection-level hardware-noise sensitivity for GPT-2

**Deliverables**:
- Sensitivity scores for each projection in each transformer block
- Perplexity and KL-divergence metrics under controlled noise
- Profile saved for use in later phases

**Files**:
- `experiments/phase1_sensitivity/run_sensitivity_profile.py`
- `src/profilers/sensitivity_profiler.py`
- `src/models/gpt2_model.py`

**Key Steps**:
1. Load pretrained GPT-2 model
2. For each projection, inject Gaussian noise at multiple levels
3. Measure perplexity degradation
4. Compute normalized sensitivity scores
5. Save profiles to `data/profiles/`

**Expected Output**:
- Sensitivity profile: `{block_id}_{proj_name} -> sensitivity_score`
- Quality metrics: perplexity, KL divergence
- Sensitivity distribution across projections

---

## Phase 2: Heterogeneous Tile-Fidelity Model
**Goal**: Model hardware tiles with different and time-varying fidelity

**Deliverables**:
- Tile collection with heterogeneous fidelity levels
- Temporal degradation simulation
- Fidelity tracking over time

**Files**:
- `experiments/phase2_fidelity/run_fidelity_model.py`
- `src/simulators/tile_fidelity.py`
- `src/simulators/hardware.py`

**Key Steps**:
1. Initialize 64 tiles with three fidelity classes (high, medium, low)
2. Simulate time-varying degradation:
   - Gradual drift: linear fidelity decrease
   - Localized degradation: sudden drop in select tiles
   - Thermal variation: noise-based fluctuation
3. Validate fidelity tracking

**Expected Output**:
- Fidelity evolution over 100+ timesteps
- Distribution of tiles by fidelity class over time
- Degradation rate validation

---

## Phase 3: Static Mapping Baselines
**Goal**: Establish baseline mapping strategies to compare against

**Deliverables**:
- Random mapping: Assign projection shards to available tiles randomly while respecting capacity constraints.
- Sequential mapping: Place projection shards onto tiles in model execution order.
- Hardware-only (fidelity-based) mapping: Assign projections using tile fidelity without considering projection sensitivity.
- Static sensitivity-aware mapping: Place the most sensitive projections on the highest-fidelity tiles once, without runtime remapping.

**Files**:
- `experiments/phase3_baselines/run_baseline_mappings.py`
- `src/mappers/base_mapper.py`
- `src/mappers/static_mapper.py`

**Key Steps**:
1. Implement RandomMapper (baseline)
2. Implement SequentialMapper (baseline)
3. Implement static sensitivity-aware mapper (greedy)
4. Evaluate each against metrics
5. Compare perplexity, remapping events, etc.

**Expected Output**:
- Baseline evaluation results
- Perplexity under each mapping strategy
- Sensitivity-weighted error metrics
- Performance comparison table

---

## Phase 4: Adaptive Mapping Algorithm
**Goal**: Implement migration-aware adaptive mapping with threshold-based remapping

**Deliverables**:
- AdaptiveMapper with greedy assignment
- Remapping decision logic based on migration cost
- Comparison with non-cost-aware adaptive mapping

**Files**:
- `experiments/phase4_adaptive/run_adaptive_mapping.py`
- `src/mappers/adaptive_mapper.py`

**Key Steps**:
1. Implement greedy assignment (sensitivity to fidelity)
2. Add migration cost estimation
3. Implement remapping threshold logic
4. Add cooldown/hysteresis mechanism
5. Evaluate with varying threshold values

**Expected Output**:
- Adaptive mapping trade-off curves (quality vs. migration overhead)
- Remapping event frequency at different thresholds
- Weight migration volume over time
- Comparison with baselines

---

## Phase 5: Comprehensive Evaluation
**Goal**: Evaluate all mapping strategies and generate final results

**Deliverables**:
- Complete comparison of all 6 mapping strategies
- Trade-off analysis (quality vs. overhead)
- Metrics: perplexity, energy, latency, remapping costs
- Final recommendation and insights

**Files**:
- `experiments/phase5_evaluation/run_full_evaluation.py`
- `experiments/phase5_evaluation/analyze_results.py`

**Key Metrics**:
- Language-model perplexity
- KL divergence from clean model
- Sensitivity-weighted tile error
- Inference latency
- Inference energy
- Number of remapping events
- Total weight data moved
- Remapping overhead (energy + latency)

**Expected Output**:
- Side-by-side comparison tables
- Trade-off curves for each degradation scenario
- Recommendations for deployment
- Publication-ready figures

---

## Success Criteria

- [ ] Phase 1: Sensitivity profiles show clear differentiation between projections
- [ ] Phase 2: Fidelity model matches expected degradation patterns
- [ ] Phase 3: Baselines establish meaningful performance bounds
- [ ] Phase 4: Adaptive mapping reduces overhead vs. naive adaptation
- [ ] Phase 5: Clear evidence that projection sensitivity guides effective mapping

## Timeline

Estimated effort per phase (assuming part-time work):
- Phase 1: 2-3 weeks (dependency on AIHWKit setup)
- Phase 2: 1 week (simulation only)
- Phase 3: 1 week (implementation of baselines)
- Phase 4: 2 weeks (algorithm development + evaluation)
- Phase 5: 2-3 weeks (comprehensive analysis + writing)

**Total**: ~2-3 months
