The idea is to minimise the influence of low-frequency noise on the performance of quantum algorithm. 

#### Actions
In the experiment on can perform three actions:
- Run quantum algorithm
- Estimate the field
- Check the field
The working of each of them is described below:
##### Quantum-algorithm
For the proof-of-principle we use the spin-flip algorithm, in which the desire outcome is the flip of the initial state $\ket{0}$. We aim at maximising the figure of merit in $N$ shots,
$$

Q_{N} = |\sum_i^{N} y_i|, \text{where}\,\, y_i = \pm 1

$$
where $y_i$ is the outcome of the measurment performed in the z-basis. For such, a simple algorithm the probability of measurement can be related to an error in estimation, since we have:
$$

 p(y_i = 0| \tau, B) = \frac{1}{2} + \frac{1}{2}\cos(B \tau),
 \tag{1}
$$

Now for the estimate of $\hat B$, the time is given by $\hat \tau = \pi/\hat B$, which gives:
$$
p_\text{err} = \frac{1}{2} + \frac{1}{2}\cos\left(\pi + \frac{\pi \delta B}{\hat B}\right)
$$
Clearly the error probability depends on the estimation error $\delta B$ as well as the estimate of $\hat B$ . For the estimation of the error in the long experiments see: [[Error-estimation]]
##### Estimation-protocol
The second action, which can be performed is the estimation protocol, in which the evolution time is selected to increase information about the field, i.e. reduce the uncertainty. This is done by the equation (1), plugged sequentially into the Bayesian scheme:
$$
	p(B_{n}) = \frac{p(y_n|B_{n}, \tau_{n})P(B_{n-1})}{p(y_{n})}
$$
Where, in principle the $\tau_n$ can be choose adaptively, i.e. as a function of the distribution $P(B_{n-1})$. 

There are several methods of adaptive estimation. For this paper we use the adaptive scheme in which the next time is based on the standard deviation of the posterior, i.e.
$$
\tau_{n+1} = \frac{1}{c\sigma_{n}}
$$
where $c=4$. We characterise [[performance-of-estimation-protocol]].

##### Checking-algorithm
We highlight that in the proof of principle case the result of the flip can be used to update the probability using Eq. (1), however this will be not in general true, as the results of quantum algorithms are often untrackable. 

However since fliping spin can be used to check the estimation protocol, i.e. it reject the current estimate if the outcome of the experiment is 0, we include the flip as possible action. 

For the time-being the outcome is not counted as a reward. 

Q: Should we reward the algorithm here, as in general the reward scheme is not obvious? 


