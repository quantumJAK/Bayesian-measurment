## Description of Estimation Models
This note explores various estimation models suitable for real-time estimation. We can't use multi-shot models like linear time or Shulman. Instead, we focus on adaptive protocols where the next shot's evolution time depends on the previous shot's distribution, $p(\omega)$.

### Adaptive protocols
The evolution time is chosen such that the next shot will give the most information about the unknown parameter $\omega$. 

##### Inverse sigma approach
In the first attempt we use adaptive scheme, in which 
$$
\tau_{n+1} = \frac{1}{c \sigma_n}
$$
where c is the constant to be optimized. We perform [[Optimization-of-c]] to find the c that maximizes reward. 

##### Check or drift approach
In the seconde attempt we either try to flip the spin (check) or create a superpositon of the measurment basis (drift). They correspond to the evolution times:
$$
\tau^{\text{check}}_{n+1} = 1/2\mu_n,\quad \tau^{\text{drift}}_{n+1} = 1/4\mu_n
$$
The agent can use either of them at will. The names are associated with the evolution of the posterior distribution (See picture below)

((PICTURE))


#### Combination of inverse sigma and check
In the third attempt we combine the two approaches above, i.e. the agent can choose between the estimation with the inverse of sigma and check. 


### Agent driven method
In principle, what can also happen is the strategy picked by the agent. One can imagine the action space can correspond to the evolution times. In this way the agent is the most flexible. The problem is it might not learn the optimal strategy. 

Let's try that! 