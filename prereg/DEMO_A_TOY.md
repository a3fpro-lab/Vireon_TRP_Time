# DEMO A (Toy Environment) — Pre-Registered

## Claim
Under fixed R, increasing P via TRP training increases effective time T = Σ R_t P_t dt
without violating the KL leash in the toy model.

## Model / Code
- src/vireon_trp/models.py
- scripts/demo_toy_env.py

## Fixed Parameters
- steps = 300
- R_mode = "constant"
- u = 0.02
- gamma = 0.05
- alpha = 0.02
- seed list: 1,2,3,4,5,6,7,8,9,10

## Metrics
Primary:
- Final P
- Effective time T
Secondary:
- Final divergence D

## Win Condition
Median(T_real) ≥ 1.4 × Median(T_baseline P=1)
AND Median(D_final) ≤ 1.0 (arbitrary stability cap)

## Controls / Falsifiers
- Shuffle P_t → destroys structured gain. Expect T_shuf ≈ T_baseline.
- Poissonize P_t → same mean, no structure. Expect no consistent advantage.
If controls match real, claim fails.
