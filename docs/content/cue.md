# DigiMicPy carbon-use-efficiency recipe

The [platform CUE workflow](https://digimicorg.github.io/workflows/carbon-use-efficiency/)
owns the definitions, reference-environment choices, interpretation, and
reporting guidance. DigiMicPy does not currently expose a CUE helper; the
calculation uses public parameter arrays and a solver endpoint.

```python
import numpy as np

C_eq = result.y[:parameters.n_consumers, -1]
R0 = np.asarray(reference_resources, dtype=float)
survivors = C_eq > 1e-5

eta = 1.0 - parameters.leakage.sum(axis=2)
uptake0 = np.sum(parameters.uptake * R0[None, :], axis=1)
retained0 = np.sum(parameters.uptake * eta * R0[None, :], axis=1)

valid = uptake0 > 0.0
species_cue = np.full(parameters.n_consumers, np.nan)
species_cue[valid] = (
    retained0[valid] - parameters.mortality[valid]
) / uptake0[valid]

included = survivors & valid
if not np.any(included):
    raise ValueError("No surviving consumers have positive reference uptake")

biomass_cue = np.average(
    species_cue[included],
    weights=C_eq[included],
)
flux_cue = np.average(
    species_cue[included],
    weights=C_eq[included] * uptake0[included],
)
```

The quantity computed with the mortality term is a reference-condition
net-retention proxy, not a realised or measured community CUE.
`parameters.mortality` is a generic first-order loss of consumer biomass: within
the model it cannot be separated into maintenance, death, or washout. The proxy
is evaluated at the reference resource vector `R0`, not at the realised resource
levels, and a realised CUE would need an explicit accounting of biomass
production and carbon fates. Remove the mortality term for the gross retained
fraction, which is set by leakage alone.

Use the same `R0`, survivor threshold, and zero-uptake rule across every
community being compared, and report them. The proxy depends on `R0`. For a
uniform reference level `R0 = r` on every resource, both summaries equal
`gross - K / r`, where `K` is the weighted mean of
`mortality_i / sum_j uptake_ij` and does not depend on `r`. The sign therefore
changes at `r = K / gross`, and two treatments with different gross retention
and different `K` can change rank as `r` changes; only treatments with the same
gross retention keep their order. Repeat the calculation at more than one
reference level, and limit qualitative statements to those that hold at each.
