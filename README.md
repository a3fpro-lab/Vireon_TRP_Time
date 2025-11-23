![CI](https://github.com/a3fpro-lab/Vireon_TRP_Time/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-blue.svg)](LICENSE-DOCS)

# VIREON TRP Time (Canonical)

**Core law:**  
\[
T = R \times P
\]
Effective time is the product of external structure (“Reality bandwidth” \(R\)) and internal gain (“Perception” \(P\)), governed by a stability constraint (KL-Leash) and tested through preregistered falsifiers.

---

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
