# How the policy optimizers work

The examples use three numerical ideas. Backward induction enumerates the outcomes of a known finite model. Proximal policy optimization (PPO) improves a policy using rollouts and a reward signal. Constrained policy optimization (CPO) and the Gaussian CVaR actor update use rollouts, a separate cost signal and a bound on the size of a policy update.

This guide develops these algorithms from the quantities provided to their implementations. Start with the policy and advantage definitions, then follow the update used in the example for your task. [Optimization problems and their assumptions](formulation-comparison.md)

## Policies, values and advantages

At action time $t$, the environment has a physical state $s_t$. The actor observes $o_t$, which may include the time and remaining reserve. A policy $\pi_w(a\mid o)$ assigns probabilities or densities to actions. In the policy-gradient examples, $w$ is a vector of neural network parameters. In a finite Markov model, the optimal policy can be given by a table indexed by state, reserve and time.

Write $r_t$ for the reward supplied to the algorithm. For a priced task this includes any reserve deduction and the failure charge. In barrier-free CPO and CVaR training, $r_t$ is the original utility and a distinct signal specifies the cost. Let $G_t=\sum_{j\ge t}\gamma_{\rm alg}^{j-t}r_j$ be the return from $t$, ending at task termination. The algorithmic discount is $\gamma_{\rm alg}$. In an ordinary simulator it equals the utility discount $\beta$. Independent geometric stopping provides the weights $\beta^{j-t}$ in expectation, so those rollouts use $\gamma_{\rm alg}=1$.

In a fully observed state, the value is $V^\pi(o)=\mathbb E_\pi[G_t\mid o_t=o]$ and the action value is $Q^\pi(o,a)=\mathbb E_\pi[G_t\mid o_t=o,a_t=a]$. Time belongs in $o$ for a finite-horizon value function. With partial observations, a critic is a predictor from the information it receives. The advantage $A^\pi(o,a)=Q^\pi(o,a)-V^\pi(o)$ measures how much better that action is than a draw from the current policy at the same state. The mean advantage under that policy is zero.

To see how an advantage enters a gradient, differentiate the product of action probabilities along a trajectory. For dynamics independent of $w$, this introduces the score $\nabla_w\log\pi_w(a_t\mid o_t)$ at each action. Only subsequent rewards contribute to its expected product. One can also subtract a baseline depending on the observation, because the conditional expected score is zero. For a finite horizon with discounted reward, this gives

$$
\nabla_w \mathbb E_{\pi_w}\!\left[\sum_t\beta^t r_t\right]
=\mathbb E_{\pi_w}\!\left[\sum_t\beta^t\nabla_w\log\pi_w(a_t\mid o_t)\,A^{\pi_w}(o_t,a_t)\right].
$$

In this display the advantage uses return discount $\beta$. An isolated ascent step proportional to a positive advantage times this score raises that action's probability to first order. An actor changes the action distribution. A critic estimates values used to measure the advantage.

## PPO with a scalar reward

### Estimate advantages from a rollout

A rollout is a sequence of transitions collected with the current policy $\pi_{w_0}$. Let $v(o)$ be the critic prediction. The temporal-difference residual compares the realized reward plus estimated continuation with the current prediction. At a genuine terminal step, set the continuation term to zero.

$$
e_t=r_t+\gamma_{\rm alg}v(o_{t+1})-v(o_t).
$$

Generalized advantage estimation (GAE) sums future residuals within the episode or rollout with weights $(\gamma_{\rm alg}\lambda_{\rm GAE})^j$. A smaller $\lambda_{\rm GAE}$ relies more on the critic. Taking $\lambda_{\rm GAE}=1$ to the end of a complete episode makes the sum telescope to $G_t-v(o_t)$. The estimated return used to fit the critic is $\widehat A_t+v(o_t)$, computed with the old value predictions. At a rollout cut inside an episode, the continuation value supplies a bootstrap estimate.

### Form the clipped objective

A minibatch comes from $\pi_{w_0}$. For each sampled action, compare its probability under the candidate policy with its probability when sampled. Let $L_t(w)=\pi_w(a_t\mid o_t)/\pi_{w_0}(a_t\mid o_t)$ and let $\epsilon>0$ be the clipping level. PPO increases the sample objective

$$
\widehat J_{\rm PPO}(w)=\frac1n\sum_{t=1}^{n}
\min\!\left\{L_t(w)\widehat A_t,\quad
\operatorname{clip}\!\big(L_t(w),1-\epsilon,1+\epsilon\big)\widehat A_t\right\}.
$$

When an estimated advantage is positive, increasing the action ratio beyond $1+\epsilon$ no longer improves that sample term. For a negative advantage, decreasing the ratio below $1-\epsilon$ no longer improves it. For $\epsilon=0.2$, $L_t=1.3$ and $\widehat A_t=2$, the contribution is $2.4$. At the same ratio with advantage $-2$, the contribution is $-2.6$. Raising the probability of a disadvantageous action is still penalized.

Stable-Baselines3 collects a rollout, computes advantages, and takes several minibatch gradient steps before collecting another rollout. Its loss combines the negative clipped objective, a squared value prediction error and any configured entropy bonus. Advantage normalization and reward normalization, where configured, affect the numerical update. Ratio clipping modifies the training objective. The final failure probability is measured separately.

In a budget wrapper, only the data supplied to PPO change. For example, if a price penalizes fuel failure, sampled advantages account for the failure charge in the return. PPO sees a priced reward and does not receive a separate failure tolerance. [Applied fuel-budget notebook](../notebooks/01_lander_fuel_tank.ipynb) · [PPO under a geometric clock](../notebooks/under-the-hood/T2_training_against_an_exact_benchmark.ipynb)

## Adaptive failure pricing

For objective $W(\pi)$ and constraint $C(\pi)\le d$, a multiplier $\nu\ge0$ assigns the objective $W(\pi)-\nu(C(\pi)-d)$. At a fixed $\nu$, a policy optimizer can improve this objective. An update in the other direction increases the price when the observed cost exceeds $d$ and decreases it when the cost is below $d$. Projection keeps the price nonnegative.

The toolbox's `DualAscent` callback implements this update for the mean of the recorded failure flags in a rollout. Given $k$ failures among $n$ completed episodes and step size $h$, its update is

$$
\nu_{j+1}=\max\!\{0,\ \nu_j+h(k/n-\alpha_{\rm train})\}.
$$

If $n=0$, the multiplier stays at its previous value. With a geometric clock, these flags estimate $q_\beta$. With a physical horizon they estimate failure by that horizon. The failure charge is applied at the failure time. For a fixed finite horizon with $\gamma_{\rm alg}=\beta<1$, the actor's discounted charge weights a failure at $t$ by $\beta^t$, even though the callback counts that failure once. Thus an adaptive finite-horizon run in the toolbox is a way to search for a candidate, and the direct binomial test supplies evidence for the stated chance constraint. [Retailer notebook](../notebooks/02_retailer_cash.ipynb)

## CPO with separate reward and cost

### Estimate local utility and cost

The task is to maximize expected utility subject to expected trajectory cost at most $d$. The cost may be total use, a discounted total or an event indicator. For example, charging one unit at the first crossing with cost discount one makes the mean cost equal to the crossing probability by the chosen horizon. A barrier is optional. With no reward penalties, the reward advantage is built directly from the task utility.

The native actor update takes a batch of $(o_i,a_i,\widehat A_i^u,\widehat A_i^c)$ from the current actor, a mean cost estimate $\widehat C_0$, and a positive cost scale $s_c$. The caller computes the baselines and advantage estimates. The importance ratios compare action probabilities on the sampled observations. The implementation uses the utility and mean cost estimates

$$
\widehat J(w)=\frac1n\sum_i L_i(w)\widehat A_i^u,\qquad
\widehat C(w)=\widehat C_0+\frac{s_c}{n}\sum_i(L_i(w)-1)\widehat A_i^c.
$$

The subtraction of one preserves the specified cost baseline at $w=w_0$. The scale converts the sampled advantage correction to the units of the chosen trajectory cost. The dispatch example has one action per episode and uses scale one. A sequential task needs estimates and scaling consistent with its own discount and sampling scheme.

### Solve a quadratic trust-region problem

Let $g=\nabla_w\widehat J(w_0)$, $c=\nabla_w\widehat C(w_0)$, and $b=\widehat C(w_0)-d$. A positive $b$ means the current estimate exceeds the cap. Define $F$ as the Hessian at the old actor of the average Kullback–Leibler (KL) divergence from the old action distribution to the candidate distribution. The KL divergence is the old distribution's expectation of the log density ratio of old to new. The implementation adds diagonal damping to the Hessian. For parameter change $v$ and KL tolerance $\delta_{\rm KL}$, the local problem is

$$
\begin{aligned}
\underset{v}{\operatorname{maximize}}\quad &g^\top v\\
\text{subject to}\quad &b+c^\top v\le0,\\
&\tfrac12v^\top Fv\le\delta_{\rm KL}.
\end{aligned}
$$

The utility and cost are first-order approximations. The second-order approximation of the KL divergence limits the change in the action distributions. The toolbox factors $F=LL^\top$. In coordinates $y=L^\top v$, the trust region is the ball $\|y\|\le\sqrt{2\delta_{\rm KL}}$ and the cost limit is a half-space. The reward-only step points along $L^{-1}g$. If it satisfies the cost limit, it is the proposed step. Otherwise the solver maximizes utility on the cost plane inside the ball. If the plane has no feasible intersection, a recovery proposal seeks the greatest predicted cost decrease. With zero cost gradient and positive cost gap, the linearized constraint is infeasible and the solver has no recovery direction.

For $g=(2,1)$, $c=(1,0)$, $b=-0.6$, $F=I$, and $\delta_{\rm KL}=0.5$, the ball has radius one. The reward-only optimum has first coordinate $2/\sqrt5$, which exceeds the cap $v_1\le0.6$. On the cost plane, the largest feasible second coordinate is $\sqrt{1-0.6^2}=0.8$. The local constrained step is therefore $(0.6,0.8)$.

### Recheck the proposal on the batch

The local solver proposes a step. Backtracking tests that step and progressively shorter steps using the sampled utility surrogate, cost surrogate and actual mean KL on the same observations. A standard accepted step respects the sampled cost and KL limits. If the current iterate was feasible, it must also give nonnegative surrogate utility gain. A recovery step must reduce the estimated cost gap and respect the KL limit. Its estimated cost may still exceed the cap. A rejected proposal leaves the actor parameters in place.

`cpo_step` implements the quadratic calculation, `cpo_update` adds backtracking, and `cpo_policy_update` constructs the sampled problem from the actor and rollouts. The OmniSafe adapter uses its own loop for rollout collection, critic fitting and updates. Fresh trajectories are used to estimate the deployed actor's cost. [Runnable constrained update](../examples/train_risk_constrained.py)

## Trust-region training under a Gaussian CVaR surrogate

For trajectory loss $Z$, upper-tail conditional value-at-risk (CVaR) at tail mass $\varepsilon$ is the mean cost in the worst $\varepsilon$ fraction. In a discrete model, split the probability at the tail boundary so that the included mass is exactly $\varepsilon$. The optimization formulation imposes $\operatorname{CVaR}_{\varepsilon}(Z)\le d$ while maximizing expected utility.

When $Z$ is Gaussian with mean $\mu$ and standard deviation $\sigma$, the upper tail starts at $\mu+\sigma\Phi^{-1}(1-\varepsilon)$. If $\varphi$ and $\Phi$ are the standard normal density and distribution function, the standardized tail mean is $\varphi(\Phi^{-1}(1-\varepsilon))/\varepsilon$. This follows from integrating $x\varphi(x)=-\varphi'(x)$ above the threshold. The resulting constraint statistic is

$$
C_{\rm G}=\mu+\frac{\varphi(\Phi^{-1}(1-\varepsilon))}{\varepsilon}\,\sigma.
$$

The native update uses this expression as a surrogate. It estimates $\mu(w)$ with the same centered importance correction as CPO. For variance $S(w)$ it uses the old variance plus a scaled average of $(L_i(w)-1)\widehat A_i^{S}$, where $\widehat A_i^{S}$ estimates a variance contribution. It evaluates $\sigma(w)$ as the square root of this variance, with a numerical floor. The gradient of $C_{\rm G}(w)$ replaces the mean-cost gradient in the quadratic problem, and backtracking checks the surrogate tail cost.

For one-step dispatch, `CostVariance` uses the centered values of $(Z_i-\overline Z)^2$ as the variance advantages. With $x=\Pr(a=1)$, the cost is $1.25$ with probability $x/2$ and $0.25$ otherwise. At $x=0.2$, the mean is $0.35$. The worst quarter contains all $0.1$ probability of the high cost plus $0.15$ probability of the low cost, so exact upper-quarter CVaR is $(0.1\cdot1.25+0.15\cdot0.25)/0.25=0.65$. The Gaussian surrogate uses the variance of those same outcomes and gives a different value. The example therefore compares two policy optimization problems using the same $0.62$ cap in cost units. Its underlying mean-cost problem is

$$
\begin{aligned}
\underset{0\le x\le1}{\operatorname{maximize}}\quad & 0.6+0.4x\\
\text{subject to}\quad & 0.25+x/2\le0.62 .
\end{aligned}
$$

The tail-cost version keeps the same objective and imposes $\operatorname{CVaR}_{0.25}(Z_x)\le0.62$, where $Z_x$ has the two-point law above. The Gaussian training variant replaces this constraint statistic by $C_{\rm G}(x)$. The actor updates improve sampled approximations of these problems. Report the exact or empirically estimated tail risk on the cost distribution relevant to deployment. [Dispatch example](../examples/train_risk_constrained.py) · [Two methods on the wear simulator](wear-benchmark.md)

## Dynamic programming on a known model

Backward induction computes the best value for each time, state and reserve. At the final time the continuation value is zero. At each previous time, evaluate each action by averaging its immediate reward and optimal continuation value over its known transition outcomes. The maximizing action becomes the table entry. In the pathwise problem, an action is admissible only when every positive-probability successor has nonnegative reserve and an admissible continuation. The recurrence and a two-period calculation are given in the [dynamic programming tutorial](dynamic-programming.md).
