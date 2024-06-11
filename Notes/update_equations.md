# Update equation for the mu and sigma

We write the update equation for the mean and variance. 
v
$$
\mu' = m_1 = \int \text{d}\omega p(\omega|\mathbf{x}) \omega
$$

$$
m_2 = \int \text{d}\omega p(\omega|\mathbf{x}) \omega^2
$$

where the posterior is given byb

$$
p(\omega|\mathbf{x}) = \frac{p(\mathbf{x}|\omega)p(\omega)}{p(\mathbf{x})}
$$

We consider the likelihood function as
$$
p(\mathbf{x}|\omega) = \frac{1}{2}\big(1 + \cos(\omega t\big)\big)
$$

So for the Gaussian prior we have:
$$
p(\omega) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(\omega-\mu)^2}{2\sigma^2}\right)
$$
which gives:
$$p(x) = (1 + x \cos\mu t\, e^{-\sigma^2t^2/2})/2$$
$$m_1 = \frac{1 + [x \mu \cos\mu t - x \sigma^2 t \sin\mu t]e^{-\sigma^2 t^2/2}}{1 + x \cos\mu t\, e^{-\sigma^2v^2/2}}$$
$$m_2 = \frac{1 + [x (\mu^2+\sigma^2-\sigma^4t^2) \cos\mu t - 2x \mu\sigma^2 t \sin\mu t]e^{-\sigma^2 t^2/2}}{1 + x \cos\mu t\, e^{-\sigma^2t^2/2}}$$


