# Shared references

Keyed bibliography cited by the phase guides and configuration comments.

| Key | Reference | Used for |
| --- | --- | --- |
| `joshi2020` | V. Joshi, M. Le Gallo, S. Haefeli, I. Boybat, S. R. Nandakumar, C. Piveteau, M. Dazzi, B. Rajendran, A. Sebastian, E. Eleftheriou, "Accurate deep neural network inference using computational phase-change memory," *Nature Communications* 11, 2473 (2020). | PCM programming-noise magnitude (σ of a few percent of the conductance range); hardware-aware training via noise injection on the forward pass with clean-weight updates (Phase 0); analog MVM energy scale. |
| `nandakumar2019` | S. R. Nandakumar et al., "Mixed-precision deep learning based on computational memory," *Frontiers in Neuroscience* (2020); companion PCM characterization (2019). | Device-to-device programming-noise spread motivating the Phase-2 fidelity-class multiplier ranges. |
| `legallo2018` | M. Le Gallo, A. Sebastian et al., "Mixed-precision in-memory computing," *Nature Electronics* 1, 246–253 (2018); M. Le Gallo et al. on PCM conductance drift (ν ≈ 0.03–0.1). | Conductance-drift exponent underlying the Phase-2 gradual-drift surrogate; analog MAC energy scale. |
| `wan2022` | W. Wan, R. Kubendran, C. Schaefer et al., "A compute-in-memory chip based on resistive random-access memory," *Nature* 608, 504–512 (2022). | Spatially correlated cross-chip variation motivating the Phase-2 thermal-variation correlation; ReRAM CIM system context. |
| `horowitz2014` | M. Horowitz, "1.1 Computing's energy problem (and what we can do about it)," *ISSCC* (2014). | Digital 8-bit MAC energy constant (`cost_model.e_mac_digital_pj`). |
| `murmann-adc` | B. Murmann, "ADC Performance Survey 1997–2023" (continuously updated), https://github.com/bmurmann/ADC-survey. | 8-bit SAR ADC conversion energy (`cost_model.e_adc_pj`). |
| `rasch2023` | M. J. Rasch et al., "Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators," *Nature Communications* 14, 5282 (2023). | AIHWKit hardware-aware training practice contextualizing the Phase-0 recipe. |

Constants derived from these sources are first-order and centralized in the
`cost_model` configuration section so alternative numbers are a config edit,
not a code change.
