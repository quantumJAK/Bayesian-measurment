import numpy as np

ueV_to_MHz = 1e3/4

def get_spectrum(signal, time_step , total_time):
    '''
    This function calculates the spectrum of a signal.
    ----------------
    Parameters:
    signal: the signal
    time_step: time step
    total_time: total time of the signal
    ----------------
    Returns:
    f: the frequencies
    Sxx: the spectrum
    '''
    N = len(signal)
    f = np.fft.fftfreq(N, time_step)
    xf = np.fft.fft(signal)
    Sxx = 2*time_step**2/total_time*(xf*np.conj(xf))
    Sxx = Sxx.real
    Sxx = Sxx[:int(N/2)]
    return f[:int(N/2)], Sxx

class Telegraph_Noise():
    def __init__(self, sigma, gamma, x0=None):
        self.gamma = gamma
        self.x0 = x0
        self.sigma = sigma
        if x0 is None:
            self.x = self.sigma*(2*np.random.randint(0, 2)-1)

    def update(self, dt):
        # update telegraph noise
        switch_probability = 1/2 - 1/2*np.exp(-2*self.gamma*dt)
        r = np.random.rand() 
        if r < switch_probability:
            self.x = -self.x    
        return self.x
    
    def reset(self):
        if self.x0 is None:
            self.x = self.sigma*(2*np.random.randint(0, 2)-1)
        else:
            self.x = self.x0




class OU_noise():
    def __init__(self, sigma, gamma, x0=None):
        self.tc = 1/gamma
        self.sigma = sigma
        self.x0 = x0
        if x0 is None:
            self.x = np.random.normal(0,sigma)
        else:
            self.x = x0
  
    def update(self, dt):
        self.x = self.x*np.exp(-dt/self.tc) + np.sqrt(1-np.exp(-2*dt/self.tc))*np.random.normal(0,self.sigma)
        return self.x

    def reset(self, x0):
        if x0 is None:
            self.x = np.random.normal(0,self.sigma)
        else:
            self.x = x0
        return self.x

    def set_x(self, x0):
        self.x = x0
        return self.x

    def update_mu(self, dt, mu, std):
        return mu*np.exp(-dt/self.tc)
    
    def update_std(self, dt, mu, std):
        return np.sqrt(self.sigma**2 + (std**2 - self.sigma**2)*np.exp(-2*dt/self.tc))

class Over_f_noise():

    def __init__(self, n_fluctuators, S1 ,sigma_couplings, ommax, ommin,
                    fluctuator_class = OU_noise, x0=None):
        self.n_telegraphs = n_fluctuators
        self.S1 = S1 * ueV_to_MHz
        self.sigma_couplings = sigma_couplings
        self.ommax = ommax
        self.ommin = ommin
        self.sigma = np.sqrt(2*self.S1*np.log(ommax/ommin))
        self.fluctuator_class = fluctuator_class
        self.spawn_fluctuators(n_fluctuators, sigma_couplings)
        self.cs = [0,0,0]
        if x0 is None:
            self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        else:
            self.x = x0
        
    def spawn_fluctuators(self, n_fluctuator, sigma_couplings):
        uni = np.random.uniform(0,1,size = n_fluctuator)
        gammas = self.ommax*np.exp(-np.log(self.ommax/self.ommin)*uni)
        sigmas = self.sigma/np.sqrt(n_fluctuator)*np.random.normal(1,sigma_couplings, size=n_fluctuator)
        self.fluctuators = []
        for n, gamma in enumerate(gammas):
            self.fluctuators.append(self.fluctuator_class(sigmas[n], gamma))
        
    def update(self, dt):
        self.x = np.sum([fluctuator.update(dt) for fluctuator in self.fluctuators])
        return self.x
    
    def reset(self):
        for fluctuator in self.fluctuators:
            fluctuator.reset()
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

    def update_mu(self, dt, mu, std):
        return mu
    
    def update_std(self, dt, mu, std):
        a,b,c = self.cs   
        std = std + a*np.log(1+1/(np.exp((std-c)/a)/b))
        return std
        
    def set_x(self, x0):
        x0s = x0*np.random.normal(1, 0.5, size = len(self.fluctuators))/len(self.fluctuators)
        for n,fluctuator in enumerate(self.fluctuators):
            fluctuator.set_x(x0s[n])
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

      
    def gen_trajectory(self, times):
        trajectory = []
        for time in times:
            trajectory.append(self.update(time))
        return trajectory
    