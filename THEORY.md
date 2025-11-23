# VIREON TRP Time — Theory (Canonical)

This file defines the math objects TRP uses, independent of any dataset.

---

## 1. Observations and time

Let \(x_t \in \mathcal{X}\) be observations indexed by step \(t\).

Define:
- Reality bandwidth \(R_t := \rho(x_t)\), for a measurable structure functional \(\rho:\mathcal{X}\to\mathbb{R}_{\ge0}\).
- Perception gain \(P_t \in \mathbb{R}_{\ge0}\), updated adaptively.

**Effective time-step**
\[
dt_{\mathrm{eff}}(t)=\frac{1}{1+P_tR_t}.
\]

**Subjective time**
\[
T_{\mathrm{subj}}(N)=\sum_{t=1}^N dt_{\mathrm{eff}}(t).
\]

---

## 2. Spectral unfolding (when \(x_t\) is a spectrum)

Given nondecreasing levels \(\{\lambda_n\}\), define unfolded levels
\[
u_n=\bar N(\lambda_n),
\]
where \(\bar N\) is a smooth counting function (local polynomial fit, Weyl law, or Riemann–von Mangoldt).

Spacings:
\[
s_n := u_{n+1}-u_n, \qquad \mathbb{E}[s_n]=1.
\]

---

## 3. Structure defect

Fix a grid \(G=\{g_i\}_{i=1}^m\subset (0,\infty)\).

Let \(p_{\mathrm{emp}}\) be the KDE/histogram of \(\{s_n\}\) on \(G\).  
Let \(p_0\) be a preregistered null distribution.

Define
\[
I_{\mathrm{struct}}
:=D_{\mathrm{KL}}(p_{\mathrm{emp}}\Vert p_0)
=\sum_{i=1}^m p_{\mathrm{emp}}(g_i)\log\frac{p_{\mathrm{emp}}(g_i)}{p_0(g_i)}.
\]

**Null choices**
- Poisson: \(p_0(s)=e^{-s}\)
- Wigner surmise (β-ensemble):
\[
p_\beta(s)=a_\beta s^\beta\exp(-b_\beta s^2), \quad \beta\in\{1,2,4\}.
\]

---

## 4. TRP update family

Let \(L_t=L(x_t,P_t,\theta)\) be an energy/loss.

General adaptive update:
\[
P_{t+1}=P_t\exp(-\alpha\nabla_P L_t),
\quad \alpha>0
\]
with KL-Leash enforcing stability.

Minimal dataset-agnostic choice:
\[
\rho(x_t)=I_{\mathrm{struct}}(x_t).
\]

Then
\[
dt_{\mathrm{eff}}(t)=\frac{1}{1+P_t I_{\mathrm{struct}}(x_t)}.
\]

---

## 5. KL-Leash (stability discipline)

Let \(q_t\) be TRP’s internal predictive distribution.

Forward KL leash:
\[
\Delta_{\mathrm{KL}}(t):=
D_{\mathrm{KL}}(q_{t+1}\Vert q_t)
\le \varepsilon \quad \forall t.
\]

Purpose: blocks perception spikes and guarantees bounded update energy.

---

## 6. Matched-budget evaluation

TRP agent \(A\) vs baseline \(B\).

Matched budgets:
\[
N_A=N_B,
\qquad
\sum_{t=1}^{N_A}\tau_A(t)
=
\sum_{t=1}^{N_B}\tau_B(t)
\]
where \(\tau\) is wall-clock per step.

Speedup ratio:
\[
S :=
\frac{\min\{t:I_{\mathrm{struct}}^B(t)\le\delta\}}
     {\min\{t:I_{\mathrm{struct}}^A(t)\le\delta\}}.
\]

Acceptance preregisters \(\delta\) and a minimum \(\eta>0\) such that
\[
S\ge 1+\eta.
\]

Null requirement:
\[
S_{\mathrm{null}}\approx 1.
\]

---

## 7. Sanity properties

### Theorem 1 (Scale invariance)
If \(R_t\mapsto cR_t\) and \(P_t\mapsto P_t/c\) for any \(c>0\), then \(dt_{\mathrm{eff}}\) is unchanged.

*Proof:* substitute into \(dt_{\mathrm{eff}}=1/(1+P_tR_t)\).

### Lemma 2 (Monotone freezing)
If \(P_t\) is nondecreasing and \(R_t\ge0\), then
\[
dt_{\mathrm{eff}}(t+1)\le dt_{\mathrm{eff}}(t).
\]

So TRP time can only “speed up” if perception or structure drops.

---

End.
