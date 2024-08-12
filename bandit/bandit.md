### The motivation
We map the tensition between estimation and operation into the exploitation-exploration problem for the multi-arm bandit. More-concretely problem of online optimisation of phase-flip gate, can be seen as the non-stationary, correlated multi-armed bandit problem.

### NCMBP
As the action of the agent, we pick the waiting time $a_t = \tau$. The reward is defined as the result of projective measrument after an attempt of phase-flip. This can be quantified with Binomial distribution, i.e. the reward at time $t$ associated with taking action a is given by: $r_t(\tau)\sim B(\theta_\tau(t))$, where $\theta_\tau(t)$ is the average success probability of the action $\tau$ at time $t$. Note that, in the context of Binoial distribution, the probabilities are typically correlated, for instance in the ideal case:
\[
\theta_\tau(t) = [1+\cos(\omega \tau)]/2
\]

**Compute the update**