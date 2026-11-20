# Supplementary results

These two supplementary certificates apply the same reserve dynamics to a small-reserve event and to eventual failure. The reserve notation is reserves $z_{t+1,k}=(z_{t,k}-c_k)/\gamma_k$
with $z_{0,k}=b_k$; failure at $\tau=\inf\{t:\min_kz_{t+1,k}<0\}$; clipped reward $r_m=\max\{-m,u-B(z_{t+1})\}$ with
$-m\le r_m\le\bar u$, equal to $-m$ from failure onward; $V_m=\mathbb E\sum_t\beta^tr_m$.

## 1. Certificate for a small reserve

**Statement.** Fix thresholds $\varepsilon\in(0,\infty)^K$ and suppose
$\underline B(\varepsilon)=\inf\{B(z):z_k\le\varepsilon_k\text{ for some }k\}>0$. Then every admissible policy satisfies
$$\mathbb P\big(\exists\,t\le T,\,k:\ z_{t+1,k}\le\varepsilon_k\big)\le\min\Big\{1,\ \frac{\bar u-(1-\beta)V_m}{(1-\beta)\,\beta^T\,\min\{m+\bar u,\ \underline B(\varepsilon)\}}\Big\}.$$

**Proof.** The per-period costs $\bar u-r_m$ are nonnegative and
$\mathbb E\sum_t\beta^t(\bar u-r_m)=\bar u/(1-\beta)-V_m$. On the event, let $t\le T$ be a time with $z_{t+1,k}\le\varepsilon_k$
for some $k$. If $z_{t+1}$ is not yet a failure, then $B(z_{t+1})\ge\underline B(\varepsilon)$ and
$\bar u-r_m=\min\{\bar u+m,\ \bar u-u+B(z_{t+1})\}\ge\min\{m+\bar u,\underline B(\varepsilon)\}$, since $u\le\bar u$. If it is a
failure, $\bar u-r_m=\bar u+m$. Either way the discounted sum of costs is at least
$\beta^T\min\{m+\bar u,\underline B(\varepsilon)\}$ on the event, and Markov's inequality gives the bound. $\square$

The certificate requires a uniformly positive penalty in the specified small-reserve region. A failure-only price gives $\underline B=0$.
For the log barrier $B(z)=\theta_B\sum_k\log(1+b_k/z_k)$, $\underline B(\varepsilon)=\theta_B\min_k\log(1+b_k/\varepsilon_k)$
(`LogBarrier.lower_level`). The bound is informative only for values $V_m$ close to $\bar u/(1-\beta)$; direct
measurement of the frequency of a small reserve on validation episodes (`Evaluation.near`) is usually sharper.

## 2. Structural certificate for the probability of ever failing

**Statement.** Suppose $0\le c_k\le\bar g_k<\infty$ for every admissible step and $0<\gamma_k<1$. Put
$c_k^\star=\bar g_k/(1-\gamma_k)$ and $E_T=\{\tau>T,\ z_{T+1,k}<c_k^\star\text{ for some }k\}$. Then for every policy
$$\mathbb P(\tau<\infty)\le p_T+\mathbb P(E_T).$$

**Proof.** If $z_{t,k}\ge c_k^\star$ then $z_{t+1,k}=(z_{t,k}-c_k)/\gamma_k\ge(c_k^\star-\bar g_k)/\gamma_k=c_k^\star$, so a reserve at
or above $c_k^\star$ stays there forever and never fails. On $\{\tau>T\}\setminus E_T$ every reserve satisfies
$z_{T+1,k}\ge c_k^\star$, hence $\tau=\infty$. Therefore $\{\tau<\infty\}\subseteq\{\tau\le T\}\cup E_T$. $\square$

Both events on the right are observable from $T+1$ actions. Their disjoint union is a single Bernoulli event that can be evaluated on fresh episodes. Without such a bound on the future consumption, a small geometric risk or a certificate at one horizon does not
bound the probability of ever failing (the paper's example *Delay without eventual feasibility*). The upper bound approaches the true eventual failure probability if the probability of surviving without entering the invariant reserve region tends to zero. This is an additional property of the process. For example, a single resource can stay forever at a reserve $z<c^\star$ if its consumption is constantly $(1-\gamma)z$. It then never fails and remains outside the region used in this certificate.
