import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple, MultiDiscrete, Dict
import matplotlib.pyplot as plt
from noise import Telegraph_Noise, OU_noise, Over_f_noise

def update_p(x, mu, std, t):

    t = 2*np.pi * t
    denom = 1 + x*np.cos(mu*t)*np.exp(-std**2*t**2/2)
    m1 = (mu+x*(mu*np.cos(mu*t)-np.sin(mu*t)*std**2*t
             )*np.exp(-std**2*t**2/2))/denom
    m20 = std**2+mu**2
    m2 = (m20+x*((mu**2+std**2-std**4*t**2)*np.cos(mu*t
          )-2*np.sin(mu*t)*std**2*t*mu
             )*np.exp(-std**2*t**2/2))/denom
    return m1, (np.sqrt(m2-m1**2))


def get_gauss(x, state):
    return 1/np.sqrt(2*np.pi*state[1]**2)*np.exp(-(x-state[0])**2/(2*state[1]**2))

class Filter():
    def __init__(self, N):
        self.N = N
        self.raw = []
        self.filtered = []

    def update(self, mu):
        self.raw.append(mu)
        self.filtered.append(mu)
        return mu
    
class Moving_average_filter(Filter):
    def update(self, mu):
        self.raw.append(mu)
        if len(self.raw)<self.N:
            self.filtered.append(np.average(self.raw))
        else:
            self.filtered.append(self.filtered[-1] + (self.raw[-1] - self.raw[-self.N])/self.N)
        return self.filtered[-1]

def diffuse_state(dt, mu, std, noise):
    mu = noise.update_mu(dt, mu, std)
    std = noise.update_std(dt, mu, std)
    return mu, std/mu


def J(eps):
    J0 = 20
    J1 = 5
    eps0 = -3.3
    return J0 + J1*np.exp(eps/eps0)



class Moments_estimation(Env):
    def __init__(self, length, om0, 
                 noise,buffer_size,
                 seed_field=None, 
                 seed_shot = None, 
                 penalty = -1, 
                 max_time = 100,
                 time_step = 1,
                 min_time = 1, 
                 filter = Filter(1),
                 std_step = 0.005,
                 mu_step = 0.1,
                 ):
        
        self.penalty = penalty
        self.om0 = om0
        self.time_step = time_step
        self.min_time = min_time
        self.std_step = std_step
        self.mu_step = mu_step
        
        self.noise = noise
        self.init_error = self.noise.sigma
        self.noise.set_x(self.init_error*np.random.normal(0,1))
        self.om = self.noise.x + om0
        self.mu_history = []

        self.seed_shot = seed_shot
        self.seed_field = seed_field
        self.rng_shot = np.random.default_rng(seed_shot)
        self.rng_field = np.random.default_rng(seed_field)
        self.estimaiton_length0 = length
        self.estimation_length = length
        self.filter = filter
        self.buffer_size = buffer_size
        self.observation_space = Box(low = np.array([0]*2*self.buffer_size,dtype=np.float64), 
                                     high = np.array([30,2]*self.buffer_size,dtype=np.float64), shape = (2*self.buffer_size,),dtype=np.float64) 
        
        
        self.action_space = MultiDiscrete([int((max_time-min_time)/time_step)+1, 20])
        
        self.state = np.array([self.om0,self.init_error/self.om0]*buffer_size, dtype=np.float64)



    def estimate(self, om, mu, std, t, rng_shot = np.random.default_rng()):
        x = rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*om*t))
        x = 2*x-1
        mu, std = update_p(x, mu, std, t) 
        return mu, std/mu

    def step(self, action):
        # Apply action
        # 0 - not estimate
        # 1 - estimate 
        # 2 - nothing

        reward = 0
        b = None
        #plt.figure()
        #plt.plot(self.freq_grid, self.weigths)
        mu = self.state[-2]
        if action[0] == 0:
            if mu==0:
                b = 1
            else:
                t = 1/mu/2. # change np.pi in numerator to have different angle    
                b = self.rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*(self.om)*t))
            if b==0:
                reward = 1
            else:
                reward = self.penalty
       
        else:


            #print(self.min_time+action*self.time_step)
            #print(self.min_time)
            #print(action)
            #print(self.time_step)
            #print("pre",self.state[1,-1], self.state[0,-1])
            mu, sig = self.estimate(self.om, mu = self.state[-2], std = self.state[-1]*self.state[-2], 
                                    t = self.min_time*1e-3 + action[0]*self.time_step*1e-3, rng_shot = self.rng_shot)
            #print("post",self.om,self.state[1,-1], self.state[0,-1])
            self.filter.update(mu)

            # push all elements of the matrix by on index
            self.state[:-2] = self.state[2:] #push by two indices
            self.state[-2] = self.filter.filtered[-1]
            self.state[-1] = sig
        
        self.state[-1] += np.abs(action[1])*self.std_step
        #self.state[0, -1] += np.abs(2-action[2])*self.mu_step


        plot_weights = False
        if plot_weights:
            plt.figure()
            grid = np.linspace(0,100,201)
            plt.plot(grid, get_gauss(grid, self.state))

        #self.state = diffuse_state(dt = 1, mu = self.state[0], 
        #                           std = self.state[1]*self.state[0], noise = self.noise)
        if plot_weights:
            plt.plot(grid, get_gauss(grid, self.state))
            plt.vlines(np.abs(self.om),0,1,"k")

            plt.vlines(mu,0,1,"green")


        self.noise.update(1)

        self.om = self.noise.x + self.om0

        #END THE STEP
        self.estimation_length -= 1
        # Check if estimation is done
        if self.estimation_length <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        truncated = False #?
        info = {"om":self.om}
        # Return step information
        return  np.array(self.state).astype(np.float32), reward, done, truncated, info

    def reset(self,seed=None, options=None):

        self.noise.set_x(self.init_error*np.random.normal(0,1))
        self.om = self.noise.x + self.om0

        self.state = np.array([self.om0,self.init_error/self.om0]*self.buffer_size, dtype=np.float64)



        self.estimation_length = self.estimaiton_length0
        info = {"om":self.om}

        return self.state, info

class Moments_estimation_c(Moments_estimation):
    def __init__(self, length, om0, 
                 noise, cs,
                 seed_field=None, 
                 seed_shot = None, 
                 penalty = -1, 
                 max_time = 100,
                 time_step = 1,
                 min_time = 0,
                 c= 10):
        

        super().__init__(length, om0, noise, cs, seed_field, 
                         seed_shot, penalty, max_time, time_step,
                           min_time, filter = Filter(5))
        self.action_space = Discrete(2)
        self.c = c
        

    def estimate(self,om, mu, std, t, rng_shot = np.random.default_rng()):
        t = 1/self.c/std
        x = rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*om*t))
        x = 2*x-1
        mu, std = update_p(x, mu, std, t) 
        return mu, std/mu






class two_qubit_game(Env):
    def __init__(self, length, om0,
                 noise,
                 seed_field=None, 
                 seed_shot = None, 
                 penalty = -1, 
                 max_time = [100,100,100],
                 time_step = [0.25,0.25,0.25],
                 min_time = [1,1,1]):
        
        self.penalty = penalty
        self.om0 = om0
        self.time_step = time_step
        self.noise = noise
        self.om = [self.noise[k].x + om0[k] for k in range(3)]

        self.seed_shot = seed_shot
        self.seed_field = seed_field
        self.rng_shot = np.random.default_rng(seed_shot)
        self.rng_field = np.random.default_rng(seed_field)
        self.estimaiton_length0 = length
        self.estimation_length = length

        self.observation_space = Box(low = 0, high = [20,1,20,1,100,1], shape = (6,),dtype=np.float32) 
        self.action_space = MultiDiscrete([int((max_time[k]-min_time[k])/time_step[k])+1 for k in range(3)])
        self.state = np.array([self.om,self.noise.sigma])

    def step(self, action):
        # Apply action
        # 0 - not estimate
        # 1 - estimate 
        # 2 - nothing

        reward = 0
        b = None
        #plt.figure()
        #plt.plot(self.freq_grid, self.weigths)
    
        
        action_taken = False
        
        if action[0] == 0 and action[1] == 0 and action [2] == 0:
            if mu==0:
                b = 1
            else:
                t = 1/mu/2 # change np.pi in numerator to have different angle    
                b = self.rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*(self.om)*t))
            if b==0:
                reward = 1
            else:
                reward = self.penalty
            action_taken = True            


        elif action[0] == 0:
            if self.state[0]==0:
                b = 1
            else:
                t = 1/mu/2 # change np.pi in numerator to have different angle    
                b = self.rng_shot.binomial(1, 1/2+1/2*np.cos(2*np.pi*(self.om)*t))
            if b==0:
                reward = 1
            else:
                reward = self.penalty

        else:
            self.state = estimate(self.om, mu = self.state[0], std = self.state[1], 
                                    t = action[1]*self.time_step*1e-3, rng_shot = self.rng_shot)
    


  

        self.state[0], self.state[1] = diffuse_state(dt = 1, mu = self.state[0], 
                                   std = self.state[1], noise = self.noise)
       
    

        #UPDATE NOISE
        for k in range(2):
            self.noiseom[k].update(1)
        self.noiseJ.update(1)

        self.om = self.noise.x + self.om0

        #END THE STEP
        self.estimation_length -= 1
        # Check if estimation is done
        if self.estimation_length <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        truncated = False #?
        info = {"om":self.om}
        # Return step information
        return  np.array(self.state).astype(np.float32), reward, done, truncated, info

    def reset(self,seed=None, options=None):

        self.noise.reset()
        self.om = self.noise.x + self.om0
        
        self.state = [self.om0,self.noise.sigma]
        self.state[1]/=self.state[0]
        self.estimation_length = self.estimaiton_length0
        info = {"om":self.om}

        return self.state, info
