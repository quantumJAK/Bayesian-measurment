

In most quantum hardware, interactions between qubits and their environment lead to fluctuations in Hamiltonian parameters, degrading gate fidelity [] and causing decoherence []. While environmental noise is often modeled with a Markovian approach [], which assumes no temporal correlations between experimental runs [], temporally correlated noise dominates in many solid-state qubits []. In this case, the decoherence can be modeled by a classical stochastic process in the qubit Hamiltonian [], which evolves slowly relative to qubit operation times. As a result, the relatively short \( T_2^* \) decoherence timescale [] is limited by the lack of knowledge of Hamiltonian parameters, and hence could strongly benefit from suitable feedback control protocols [].

The presence of correlated noise may require mitigation techniques beyond the standard Quantum Error Correction (QEC) [] and Quantum Error Mitigation (QEM) [] strategies. The most popular methods are Dynamical Decoupling and Pauli Twirling, which require additional control pulses []. An alternative approach involves Bayesian tracking of Hamiltonian parameters [], which has recently been demonstrated in spin qubits []. This allows for feedback-loop control of time-varying parameters but introduces an overhead due to additional single shots needed for estimation.

Unlike in well-established methods for estimating static fields [], online tracking introduces the risk of parameter drift during the estimation period []. This can be naturally mitigated by improving estimation speed. One way of decreasing the number of estimation shots is a physics-informed approach [], in which knowledge about the field is propagated between estimation rounds []. However, this method requires statistical knowledge about the noise, necessitating resource-intensive methods like noise spectroscopy [] or gate-set tomography []. Moreover, the method assumes the Markovian property of the stochastic process, which, for instance, does not hold for 1/f charge noise dominant in solid-state qubits [].

Another method to balance estimation accuracy and speed is the adaptive approach [], facilitated by the fast operation of FPGA-based qubit control []. This approach has reduced the number of shots by adaptively choosing the next probing time based on the current knowledge about the field []. So far, however, only greedy and heuristic adaptive methods have been pursued [], due to the difficulty of finding an optimal scheme that accounts for multiple estimation shots.

The task of finding an adaptive policy to achieve the goal in an unknown environment is a typical setup for Reinforcement Learning (RL) []. In this technique, the agent discovers the policy that maximizes the reward signal through interaction with the environment. So far, RL techniques have been used to improve qubit initialization [], qubit tuning [], and discover new quantum circuits []. By construction, RL relies on a Markov Decision Process [], i.e., actions depend on the current state only; however, techniques for partially observable environments have been developed []. In such cases, the state accumulates knowledge from past observations [], making it compatible with the Bayesian tracking scheme.

In this work, we propose to combine physics-informed and adaptive Bayesian Estimation with the Reinforcement Learning (RL) approach. We explore the ability of RL methods to tackle exploration-exploitation problems [], which closely resemble the balance between noise estimation and qubit operation. In each round, the agent decides whether to probe the field with a selected probing time or to perform the qubit flip, which is used to generate the reward signal. To account for finite readout and initialization time, which typically exceeds qubit operation [], after each round, the agent increases the field uncertainty by an adaptive factor. To handle stochastic environments in the form of experimentally relevant 1/f and Brownian noise trajectories, we use the state-of-the-art RL method of Proximal Policy Optimization [].

Our main contribution is the demonstration of an RL agent that:
\begin{itemize}
\item Balances resources between estimation and operation,
\item Efficiently tracks the trajectory of the noise,
\item Discovers a non-trivial probing time strategy,
\item Learns the noise statistics from the environment,
\item Is not limited to Markovian stochastic processes.
\end{itemize}
We report the agent's performance on simulated data, compare it against baseline methods, and discuss the potential for experimental implementation.

The structure of the paper is as follows: In Section 2, we introduce the workings of the Reinforcement Learning scheme, including the model of the noise, states, and actions. In Section 3, we introduce metrics and discuss baseline methods. In Section 4, we discuss the interplay between the agent's performance and the correlation time of the noise, using the Ornstein-Uhlenbeck process as an illustrative example. In Section 5, we discuss the agent's performance against experimentally relevant 1/f noise, which can be seen as an ensemble of OU processes. In Section 6, we discuss the potential for the experimental implementation of the agent. In Section 7, we conclude the paper.

## Section 2: Model
We assume the dynamics of the qubit at any point in time, is given by the Hamiltonian:
\[
    H(t) = \frac{\omega_0 + \delta \omega(t)}{2} \sigma_z/2 
\]
where \(\omega_0\) is the base qubit frequency, and \(\delta \omega(t)\) is the stochastic process of zero average. The classical bit, a single shot, is obtained from the system by performing coherent oscillation around the Hamiltonian axes, for instance after initialisation and readout along eigenstate of $\sigma_x$. To include correlation between the shots, we introduce two timescales an evolution time $\tau$ and the lab time $t$, such that the probability of obtaining the bit $x$ for the shot measured around time $t$ is given by:
\[
    p(x_t|\delta;\tau) = \frac{1}{2} \left(1 + x \cos(\omega_0 \tau + \int_t^{t+\tau} \delta \omega(t') dt')\right).
\]