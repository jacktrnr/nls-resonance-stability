# Corrected Derivation of $\nu_0$ for Full-Line Transmission Resonances

This document gives the complete derivation of the eigenvalue splitting formula $\mu(\varepsilon) = \nu_0 \varepsilon^2 + \mathcal{O}(\varepsilon^3)$ for the full-line transmission case, correcting a sign error in the Variation of Parameters Lemma in main-Apr7.tex (line 332).

**Root cause:** The VoP Lemma has a sign error on the $\alpha$ term. This single error propagates to produce wrong $\dot{E}$, wrong $\mathcal{C}$, and wrong $\nu_0$ for the full-line case. The half-line (Dirichlet, $U_a = 0$) is unaffected since $\alpha_1 = 0$.

**Verified by:** `verify_lemma_vp.jl` (numerical), `verify_nu0_sympy.py` (symbolic).

---

## 1. Setup and Notation

Transmission resonance at $k_\star = -i\gamma$, $\gamma > 0$, with eigenfunction $U_\star$ on $[a,b]$:
$$L_0 U_\star = 0, \quad U_\star'(\pm b) = \gamma U_\star(\pm b), \quad U_\star(a) \equiv U_a = 1, \quad U_\star(b) \equiv U_b.$$

The bifurcated bound state is $\psi_\varepsilon = \sqrt{\varepsilon}\,(U_\star + \mathcal{O}(\varepsilon))$ on $[a,b]$, matched to soliton tails outside, with $E = -\gamma^2 + \varepsilon\dot{E} + \mathcal{O}(\varepsilon^2)$.

**Key quantities:**
- $\kappa = \sqrt{-E} = \gamma - \varepsilon\dot{E}/(2\gamma) + \cdots$
- $\lambda = \sqrt{\kappa^2 - \mu}$, $m = \lambda/\kappa$
- $I_2 = \int_a^b U_\star^2$, $I_4 = \int_a^b U_\star^4$
- $\Omega = (U_b^4 + U_a^4)/(2\gamma) - 2I_4$
- $\mathcal{B} = I_2 - (U_b^2 + U_a^2)/(2\gamma)$

**Eigenvalue problem:** Find $\mu(\varepsilon)$ such that $L_+^\varepsilon \phi = \mu\phi$ on $\mathbb{R}$, determined by matching inner (on $[a,b]$) and outer (Pöschl-Teller Jost) log-derivatives at $x = b$.

---

## 2. VoP Lemma (Corrected)

**Lemma.** If $L_0 f = g$ on $[a,b]$ with $f(a) = 0$, $f'(a) = \alpha$, then:
$$\boxed{f'(b) - \gamma f(b) = \frac{\alpha\,U_\star(a) - \int_a^b g\,U_\star\,dx}{U_\star(b)}.}$$

**Tex error (line 332):** Claims $f'(b) - \gamma f(b) = -[\alpha\,U_\star(a) + \int g\,U_\star] / U_b$. The sign on $\alpha$ is wrong.

**Proof sketch.** The VoP representation using the fundamental system $(U_\star, \tilde{U})$ with $W[U_\star, \tilde{U}] \equiv 1$ gives $f = \alpha\,\tilde{U} + \text{convolution}$. At $x = b$, using $U_\star'(b) = \gamma U_b$ and $\tilde{U}'(b) U_b - \tilde{U}(b) \gamma U_b = W[U_\star, \tilde{U}]|_b \cdot U_b / U_b = 1$... the key is that $\tilde{U}'(b) - \gamma\tilde{U}(b) = 1/U_b$, giving $f'(b) - \gamma f(b) = \alpha\,U_a/U_b - (\int g U_\star)/U_b$.

**Numerical verification:** `verify_lemma_vp.jl` integrates $L_0 f = g$ for multiple $\alpha$ values and confirms the Wronskian formula matches for all $\alpha$, while the tex formula only matches at $\alpha = 0$.

**Consequence:** For the half-line (Dirichlet at $x = 0$), $\alpha_1 = 0$, so the sign error is invisible. For the full-line ($\alpha_1 \neq 0$), the error propagates to $\mathcal{C}$ and $\nu_0$.

---

## 3. Right Outer Log-Derivative $w_{\mathrm{out}}^R$ (at $x = b$)

For $x > b$, $\phi$ is the $L^2$-decaying Jost solution $f_+$ of the right soliton's $\ell = 2$ PT potential. The PT denominator at the right boundary is:
$$D(m, u_R) = m^2 - 1 + 3mu_R + 3u_R^2, \quad u_R = \tanh(\kappa(b - y_R)) \to -1.$$

Setting $\sigma_R = 1 + u_R = U_b^2\varepsilon/(4\gamma^2)$:
$$D(-1+\sigma_R, 1-p) \to 0 \quad \text{as } \sigma_R, p \to 0 \quad \textbf{(DEGENERATE)}.$$

This produces the **singular** $\mu/\varepsilon$ term (tex Eq. 5.5, correct):
$$\boxed{w_{\mathrm{out}}^R = \gamma - \left(\frac{3U_b^2}{4\gamma} + \frac{\dot{E}}{2\gamma}\right)\varepsilon + \frac{4\gamma\mu}{3U_b^2\varepsilon} + \mathcal{O}(\varepsilon^2).}$$

---

## 4. Left Outer Log-Derivative $w_{\mathrm{out}}^L$ (at $x = a$) — Free Decay

For $x < a$, $\phi$ is the left Jost function $f_-$ of the left soliton's PT potential. By reflection symmetry, $f_-(\xi) = f_+(-\xi)$, so $w_{\mathrm{out}}^L = -w_+(y_L - a)$.

The evaluation point is at $\tilde{u} = \tanh(\kappa(y_L - a)) \to +1$ (far tail), giving:
$$D(1 - \tilde{\sigma}, 1 - p) \to 6 \quad \textbf{(NON-DEGENERATE)}.$$

At $\tilde{u} = +1$: $w_+ = -\kappa m = -\lambda$, so $w_{\mathrm{out}}^L = +\lambda$. The $\mathcal{O}(\varepsilon)$ PT correction from the far soliton tail is $-3\kappa\tilde{\sigma}$ with $\tilde{\sigma} = U_a^2\varepsilon/(4\gamma^2)$, but for the **free-decay D̃ approach** (which integrates L₊φ = μφ on $[a,b]$ with BC $\phi'(a) = +\lambda$ only), this correction is not included in the left BC.

**Result:** The effective left BC for the numerical D̃ scan is:
$$\phi'(a)/\phi(a) = +\lambda = \gamma - \frac{\dot{E}}{2\gamma}\varepsilon - \frac{\mu}{2\gamma} + \mathcal{O}(\varepsilon^2).$$

**No singular $\mu/\varepsilon$ term.** The tex's claim (line 617) that $w_{\mathrm{out}}^L$ has a $4\gamma\mu/(3U_a^2\varepsilon)$ term is **wrong** because $D \to 6 \neq 0$ at the left boundary.

---

## 5. Inner Solution and Matching

### 5a. Inner expansion

Write $\phi_{\mathrm{in}} = U_\star + \varepsilon\phi_1 + \mu\phi_\mu$ with $L_0\phi_1 = (3U_\star^2 + \dot{E})U_\star$, $L_0\phi_\mu = U_\star$, and $\phi_j(a) = 0$, $\phi_j'(a) = \alpha_j$.

### 5b. Matching at $x = a$

From the free-decay BC ($\phi'/\phi = +\lambda$ at $x = a$):
$$\gamma + \frac{\varepsilon\alpha_1 + \mu\alpha_\mu}{U_a} = \gamma - \frac{\dot{E}\varepsilon}{2\gamma} - \frac{\mu}{2\gamma} + \cdots$$

$$\boxed{\alpha_1 = -\frac{U_a\dot{E}}{2\gamma}, \qquad \alpha_\mu = -\frac{U_a}{2\gamma}.}$$

### 5c. Inner log-derivative at $x = b$ via correct VoP

Applying the **correct** VoP (Section 2):
$$\frac{\phi_j'(b) - \gamma\phi_j(b)}{U_b} = \frac{\alpha_j U_a - \int g_j U_\star}{U_b^2}$$

So the inner log-derivative at $b$ is:
$$w_{\mathrm{in}}(b) = \gamma + \frac{\varepsilon(\alpha_1 U_a - 3I_4 - \dot{E}I_2) + \mu(\alpha_\mu U_a - I_2)}{U_b^2} + \cdots$$

### 5d. Rescaled matching

Setting $\mu = \varepsilon^2\nu$ and forming $\tilde{D} = (w_{\mathrm{in}} - w_{\mathrm{out}}^R)/\varepsilon$:
$$\tilde{D}(\nu, 0) = \mathcal{C} - \frac{4\gamma\nu}{3U_b^2},$$

where:
$$\mathcal{C} = \frac{\alpha_1 U_a - 3I_4 - \dot{E}I_2}{U_b^2} + \frac{3U_b^2}{4\gamma} + \frac{\dot{E}}{2\gamma}.$$

The eigenvalue is $\nu_0 = 3U_b^2\mathcal{C}/(4\gamma)$.

**Note:** With the **wrong** (tex) VoP, the $\alpha_1 U_a$ term enters with the opposite sign: $-\alpha_1 U_a$ instead of $+\alpha_1 U_a$. This is where the error propagates.

---

## 6. Energy Slope $\dot{E}$ from Existence Matching

The energy correction $\dot{E}$ comes from the **existence** problem (matching the nonlinear bound state to soliton tails). The inner equation is $L_0\psi_1 = (U_\star^2 + \dot{E})U_\star$ with:
- **Left BC** (soliton tail): $\alpha_1^E = -U_a(\dot{E}/(2\gamma) + U_a^2/(4\gamma))$
  (Note: soliton coefficient $U_a^2/(4\gamma)$, not PT coefficient $3U_a^2/(4\gamma)$)
- **Right BC** (soliton tail): $w_{\mathrm{sol}}(b) = \gamma - U_b^2\varepsilon/(4\gamma) - \dot{E}\varepsilon/(2\gamma)$

Applying the **correct** VoP:
$$\frac{\alpha_1^E U_a - I_4 - \dot{E}I_2}{U_b^2} = -\frac{\dot{E}}{2\gamma} - \frac{U_b^2}{4\gamma}$$

Solving for $\dot{E}$:

$$\boxed{\dot{E} = -\frac{I_4 + (U_a^4 - U_b^4)/(4\gamma)}{I_2 + (U_a^2 - U_b^2)/(2\gamma)}.}$$

**Tex error:** The tex uses the **wrong** VoP for the existence matching too, getting $\dot{E} = \Omega/(2\mathcal{B})$. Both are internally consistent with their respective VoP signs. The correct VoP gives $\dot{E} = -0.994$; the tex gives $\dot{E} = -2.134$.

**Half-line check ($U_a = 0$):** The correct formula reduces to $\dot{E} = -(I_4 - U_b^4/(4\gamma))/(I_2 - U_b^2/(2\gamma)) = \Omega/(2\mathcal{B})$, matching the tex. ✓

---

## 7. Final Formula for $\nu_0$

Substituting $\alpha_1 = -U_a\dot{E}/(2\gamma)$ and $\dot{E}$ from Section 6 into $\nu_0 = 3U_b^2\mathcal{C}/(4\gamma)$:

$$\boxed{\nu_0 = \frac{3\Omega}{4\gamma} - \frac{3U_a^4}{16\gamma^2} = \frac{3(2U_b^4 + U_a^4)}{16\gamma^2} - \frac{3I_4}{2\gamma}.}$$

The algebra is verified symbolically in `verify_nu0_sympy.py` (SymPy, 18/18 checks pass).

**Key structure:**
- The $\dot{E}$-dependent terms partially cancel in $\mathcal{C}$, but do **not** cancel completely (unlike the tex, where $\dot{E} = \Omega/(2\mathcal{B})$ makes them cancel exactly).
- The final formula is independent of $\dot{E}$ — after substituting the correct $\dot{E}$, all $I_2$ dependence drops out.

**Half-line ($U_a = 0$):** $\nu_0 = 3\Omega/(4\gamma)$ — tex is correct. ✓

**Full-line ($U_a = 1$):** Correction $= -3/(16\gamma^2)$. The tex's $\nu_0 = 3\Omega/(4\gamma) = -2.855$ is wrong; the correct value is $-3.241$.

---

## 8. Numerical Verification

For $V(x) = 3\sin(\pi x)$ on $[-1, 1]$:

| Quantity | Corrected | Tex | Numerical |
|----------|-----------|-----|-----------|
| $\gamma$ | 0.6971 | 0.6971 | — |
| $U_a$ | 1.0 | 1.0 | — |
| $U_b$ | 0.6235 | 0.6235 | — |
| $I_2$ | 1.618 | 1.618 | — |
| $I_4$ | 1.740 | 1.740 | — |
| $\Omega$ | -2.654 | -2.654 | — |
| $\dot{E}$ | **-0.994** | -2.134 | -0.994 |
| $\nu_0$ | **-3.241** | -2.855 | **-3.24** |
| $N_0$ | **$4\gamma \approx 2.79$** | $8\gamma$ | $\approx 2.79$ |

The corrected $\nu_0 = -3.241$ matches the numerical D̃ scan (from `transmission_figures.jl`) at small $\varepsilon$. The numerical $\dot{E}$ is computed from the continuation slope $dE/d\varepsilon$.

---

## Summary of Tex Errors

| # | Location | Tex Claim | Correct | Root Cause |
|---|----------|-----------|---------|------------|
| 1 | Line 332 (VoP) | $-(α U_a + \int gU_\star)/U_b$ | $(α U_a - \int gU_\star)/U_b$ | Sign on α |
| 2 | Line 583 ($\dot{E}$) | $\Omega/(2\mathcal{B})$ | $-[I_4 + (U_a^4-U_b^4)/(4γ)]/[I_2+(U_a^2-U_b^2)/(2γ)]$ | VoP sign in existence matching |
| 3 | Line 617 ($w_{\mathrm{out}}^L$) | Singular $4\gamma\mu/(3U_a^2\varepsilon)$ | No singular term ($D \to 6$) | Left boundary non-degenerate |
| 4 | Line 645 ($\nu_0$) | $3\Omega/(4\gamma)$ | $3\Omega/(4\gamma) - 3U_a^4/(16\gamma^2)$ | Errors 1–2 propagated |
| 5 | Line 650 ($N_0$) | $8\gamma$ (two solitons) | $4\gamma$ (one soliton) | Left soliton center drifts right |
