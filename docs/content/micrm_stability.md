# Numerical Jacobian recipe

The [platform stability workflow](https://digimicorg.github.io/workflows/stability/)
defines equilibrium checks, local stability, reactivity, feasibility, and
reporting requirements. DigiMicPy has no public Jacobian or stability helper,
but its pure `micrm_rhs` can be differentiated numerically.

```python
import numpy as np

from digimicpy import micrm_rhs


def numerical_jacobian(parameters, state, relative_step=1e-6):
    state = np.asarray(state, dtype=float)
    jacobian = np.empty((state.size, state.size))
    steps = relative_step * np.maximum(1.0, np.abs(state))

    for column, step in enumerate(steps):
        offset = np.zeros_like(state)
        offset[column] = step
        jacobian[:, column] = (
            micrm_rhs(0.0, state + offset, parameters)
            - micrm_rhs(0.0, state - offset, parameters)
        ) / (2.0 * step)

    return jacobian


state_equilibrium = result.y[:, -1]
residual = np.max(
    np.abs(micrm_rhs(result.t[-1], state_equilibrium, parameters))
)
J = numerical_jacobian(parameters, state_equilibrium)
eigenvalues = np.linalg.eigvals(J)
leading_real_part = np.max(eigenvalues.real)
reactivity = np.max(np.linalg.eigvalsh((J + J.T) / 2.0))
```

Confirm that `residual` is sufficiently small before interpreting the spectrum.
Repeat with smaller and larger finite-difference steps, and report the step rule,
solver tolerances, endpoint residual, and state variables retained in the
Jacobian.

## Survivor per-capita rates and integration horizon

A small endpoint `residual` does not show that the set of survivors has settled.
The residual is an absolute rate, so a consumer with small biomass that is still
declining contributes very little to it while remaining above the survivor
threshold. Check the per-capita rate of change of the survivors as well, and
repeat the integration with a longer horizon:

```python
n = parameters.n_consumers
C_end = result.y[:n, -1]
survivors = C_end > 1e-5

rhs_end = micrm_rhs(result.t[-1], result.y[:, -1], parameters)
per_capita_rate = np.max(np.abs(rhs_end[:n][survivors] / C_end[survivors]))
```

Interpret survivor counts, and any equilibrium analysis, only when
`per_capita_rate` is small and the survivor set does not change when the
horizon is extended. Report the horizon, solver method and tolerances, survivor
threshold, endpoint residual, and largest survivor per-capita rate. The horizon
needed depends on the parameter set; the short horizons in the package examples
are illustrative.

### Example of a slow transient

The script below integrates one randomly drawn community of 20 consumers and 10
resources, without leakage, from the same initial state to four horizons. It
needs only DigiMicPy and NumPy.

```python
import numpy as np

from digimicpy import (
    MiCRMParameters,
    generate_l_tensor,
    micrm_rhs,
    modular_uptake,
    solve_micrm,
)

SEED = 4
N_CONSUMERS, N_RESOURCES = 20, 10
SURVIVOR_THRESHOLD = 1e-5

rng = np.random.default_rng(SEED)
uptake = modular_uptake(N_CONSUMERS, N_RESOURCES, 2, 8.0, rng=rng)
uptake = uptake * rng.uniform(0.5, 1.5, (N_CONSUMERS, 1))
leakage_unit = generate_l_tensor(N_CONSUMERS, N_RESOURCES, 2, 8.0, 1.0, rng=rng)
parameters = MiCRMParameters(
    uptake=uptake,
    mortality=rng.uniform(0.15, 0.25, N_CONSUMERS),
    resource_supply=np.full(N_RESOURCES, 1.0),
    resource_decay=np.full(N_RESOURCES, 0.5),
    leakage=leakage_unit * 0.0,
    leakage_fraction=np.zeros(N_RESOURCES),
)
initial = np.concatenate([np.full(N_CONSUMERS, 0.01), np.ones(N_RESOURCES)])

print("  t_end  survivors  max |residual|  max |dC/dt|/C  smallest survivor")
for horizon in (5000.0, 20000.0, 60000.0, 120000.0):
    result = solve_micrm(
        parameters, initial, (0.0, horizon), method="LSODA", rtol=1e-8, atol=1e-12
    )
    state = result.y[:, -1]
    consumers = state[:N_CONSUMERS]
    alive = consumers > SURVIVOR_THRESHOLD
    rhs = micrm_rhs(result.t[-1], state, parameters)
    rate = np.max(np.abs(rhs[:N_CONSUMERS][alive] / consumers[alive]))
    print(
        f"{horizon:7.0f}  {alive.sum():9d}  {np.max(np.abs(rhs)):14.1e}"
        f"  {rate:13.1e}  {consumers[alive].min():17.1e}"
    )
```

Output recorded with DigiMicPy 0.1.0 at commit `ac9b9fe`, NumPy 2.5.3 and SciPy
1.18.1; seed 4 (`numpy.random.default_rng`), solver LSODA with `rtol=1e-8` and
`atol=1e-12`, survivor threshold `1e-5`, initial consumers 0.01 and resources
1.0; every integration reported `success`:

```text
  t_end  survivors  max |residual|  max |dC/dt|/C  smallest survivor
   5000          5         3.0e-08        1.9e-03            1.6e-05
  20000          4         7.5e-14        2.4e-15            1.5e+00
  60000          4         4.4e-16        3.3e-17            1.5e+00
 120000          4         4.2e-15        6.6e-17            1.5e+00
```

At `t = 5000` the residual is already small, but one consumer above the
threshold is still declining at a per-capita rate near `2e-3` and is lost at
the longer horizons. The residual alone would have reported five survivors.
