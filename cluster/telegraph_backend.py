import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple, MultiDiscrete, Dict
import matplotlib.pyplot as plt
from noise import Telegraph_Noise, OU_noise, Over_f_noise

def update_p(x, prob, oms, t):
    prob = (1 + x*np.cos(2*np.pi*oms*t))*prob
    return prob/np.sum(prob)


def estimate(om, prob, oms, t, rng_shot = np.random.default_rng()):
    x = rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*om*t))
    x = 2*x-1
    prob = update_p(x, prob, oms, t) 
    return np.hstack([oms, prob])

def get_gauss(x, state):
    return 1/np.sqrt(2*np.pi*state[1]**2)*np.exp(-(x-state[0])**2/(2*state[1]**2))



class Telegraph_detection(Env):
    def __init__(self, length, oms,
                 avg_om = 50,
                 std_om = 10,
                 seed_field=None, 
                 seed_shot = None):

        self.N = len(oms)
        self.avg_om = 50,
        self.std_om = 20,
        self.is_om_fixed = False
        if oms[0] == None:
            self.oms = np.random.normal(avg_om,std_om,size = self.N)
        else:
            self.oms = oms
            self.is_om_fixed = True
        self.seed_shot = seed_shot
        self.seed_field = seed_field
        self.rng_shot = np.random.default_rng(seed_shot)
        self.rng_field = np.random.default_rng(seed_field)
        self.estimaiton_length0 = length
        self.estimation_length = length

        #choose one value from oms array at random
        self.om0 = self.oms[np.random.randint(0,len(self.oms))]

        self.observation_space = Box(
            low = 0, high = np.array([100]*self.N+[1]*self.N), shape = (2*self.N,),dtype=np.float32)
                            
        self.action_space = Box(low = 0, high = 50, shape = (1,),dtype=np.float32)

        self.state = np.hstack([self.oms, np.ones(self.N)/self.N])
        

    def step(self, action):
        # How log to estimate

        reward = 0
        b = None
        #plt.figure()
        #plt.plot(self.freq_grid, self.weigths)
        
        self.state = estimate(self.om0, self.state[:self.N], self.oms,
                                t = action*1e-3, rng_shot = self.rng_shot)
        

        reward = np.sum(-np.log(self.state[self.N:]**4))
        #END THE STEP
        self.estimation_length -= 1
        # Check if estimation is done
        if self.estimation_length <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        truncated = False #?
        info = {"om0":self.om0}
        # Return step information
        return  self.state, reward, done, truncated, info

    def reset(self,seed=None, options=None):
        if self.is_om_fixed:
            1+2
        else:
            self.oms = np.random.normal(self.avg_om,self.std_om,size = self.N)
        self.om0 = self.oms[np.random.randint(0,len(self.oms))]
        self.state = np.hstack([self.oms, np.ones(self.N)/self.N])

        self.estimation_length = self.estimaiton_length0
        info = {"om0":self.om0}

        return self.state, info

