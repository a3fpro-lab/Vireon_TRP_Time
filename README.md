![CI](https://github.com/a3fpro-lab/Vireon_TRP_Time/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-blue.svg)](LICENSE-DOCS)

# VIREON TRP Time (Canonical)

**Core law:**  
\[
T = R \times P
\]
Effective time is the product of external structure (“Reality bandwidth” \(R\)) and internal gain (“Perception” \(P\)), governed by a stability constraint (KL-Leash) and tested through preregistered falsifiers.

## Mathematical Specification (Canonical)

### Core law
\[
T = R \times P
\]
Operationally, TRP models time as a local step dilation driven by external structure \(R_t\) and internal gain \(P_t\).

### Variables
- \(x_t\): observed state / datapoint at step \(t\)
- \(R_t \ge 0\): Reality bandwidth (measurable structure in \(x_t\))
- \(P_t \ge 0\): Perception gain (adaptive internal multiplier)
- \(dt_{\mathrm{eff}}(t)\): effective TRP time-step
- \(T_{\mathrm{subj}}(N)\): subjective time over \(N\) steps

### Effective time-step
\[
dt_{\mathrm{eff}}(t) := \frac{1}{1 + P_t R_t}.
\]
\[
T_{\mathrm{subj}}(N) := \sum_{t=1}^{N} dt_{\mathrm{eff}}(t).
\]

### Structure index
\[
I_{\mathrm{struct}} := D_{\mathrm{KL}}(p_{\mathrm{emp}} \Vert p_0),
\]
with preregistered null \(p_0\) (Poisson or Wigner).  
\(I_{\mathrm{struct}}\) is a measurable distance-from-null (not a proof).

### Stability: KL-Leash
\[
D_{\mathrm{KL}}(q_{t+1}\Vert q_t) \le \varepsilon \quad \forall t,
\]
with \(\varepsilon\) preregistered.

### Matched-budget win condition
\[
S :=
\frac{\min\{t: I_{\mathrm{struct}}^B(t)\le \delta\}}
     {\min\{t: I_{\mathrm{struct}}^A(t)\le \delta\}}
\ge 1+\eta,
\]
Null requirement: \(S_{\mathrm{null}}\approx 1\).--


-

## Attribution / Priority
This framework (“VIREON TRP Time: **T = R × P with KL-Leash**”) was first publicly defined and released by **Inkwon Song Jr.** in **November 2025**.  
Canonical record: this repository + tagged releases.

If you use or adapt this work, cite the canonical repo and author:  
- Inkwon Song Jr., *VIREON TRP Time Framework (T = R × P)*, 2025.

---

## What this repo contains
- **Formal definitions** of \(T, R, P\) and admissible measurement protocols
- **Computable kernels**
  - `PEstimator`, `REstimator`, `TEstimator`
  - `KLLeash` (yellow/red zone stability governor)
  - `TRPToyModel` (minimal dynamical toy)
- **Demos**
  - `demo_toy_env.py`
  - `demo_controls.py` (shuffle + Poisson nulls)
- **Preregistered gates**
  - `prereg/DEMO_A_TOY.md`
  - `prereg/DEMO_B_HUMAN.md`
  - `prereg/DEMO_C_STRUCTURED.md`
- **Controls / falsifiers**
  - shuffle proxies
  - Poissonized null series
- **Tests + CI**
  - reproducible, green-checked, no-fluff

---

## Install (local)
```bash
git clone https://github.com/a3fpro-lab/Vireon_TRP_Time.git
cd Vireon_TRP_Time
pip install -e .
