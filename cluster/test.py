import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
print("dupa")
import moments_backend as est
import importlib
import noise as nois
import utils as utl
import optimisation as opt
import importlib
importlib.reload(opt)
importlib.reload(est)
importlib.reload(nois)
importlib.reload(utl)

from stable_baselines3 import PPO





class data():
    def __init__(self, res, penalty):
        self.penalty = penalty
        self.actions = res.actions.flatten()
        self.rewards = res.rewards.flatten()
        self.mus = res.mus.flatten()
        self.oms = res.oms.flatten()
        self.stds = res.stds.flatten()*self.mus
        self.errors = self.mus-self.oms
        self.est_prob = np.sum(self.rewards==0)/len(self.rewards)
        self.succ_prob = np.sum(self.rewards==1)/len(self.rewards)
        self.fidelity = np.sum(self.rewards==1)/(np.sum(self.rewards<0)+np.sum(self.rewards==1))
        self.infidelity = np.sum(self.rewards<0)/(np.sum(self.rewards<0)+np.sum(self.rewards==1))

    def return_row(self):
        row = []
        row.append(self.penalty)
        row.append(np.mean(np.abs(self.errors)))
        row.append(self.est_prob)
        row.append(self.succ_prob)
        row.append(self.fidelity)
        return np.array(row)


noiseou = nois.OU_noise(sigma=2, gamma=1/1e4/2)



om0 = 10
env = est.Moments_estimation(length = 500, 
                                  om0 = om0, 
                                  noise = noiseou,
                                  max_time = 1000,
                                  cs=1,
                                  penalty = -5,
                                  time_step = 1,
                                  min_time = 1)


def policy_flip(state, **kwargs):
    return 0

def policy_random_p(n,state, *args, **kwargs):
    pflip = kwargs["x"][0]
    t_max = kwargs["x"][1]
    r = np.random.choice([0,1], p=[pflip,1-pflip])

    if r == 0:
        return 0
    else:
        return np.random.randint(1,t_max)
    

def policy_random_p_overc(n, state, *args, **kwargs):
    pflip = kwargs["x"][0]
    t_max = kwargs["x"][1]
    r = np.random.choice([0,1], p=[pflip,1-pflip])

    if r == 0:
        return 0
    else:
        #print(state[1])
        #print(state[0])
        #print(1/state[1]/state[0]/10)
        return int(1/state[1]/state[0]*1e3/10)

x = [[0.90,50],[0.95,50],[0.99,50]]
results2 = []
for k in range(1):
    res = opt.game(100, env, policy=policy_random_p_overc, x = x[k])
    results2.append(data(res,0))

for k in range(1):
        plt.plot(res.mus[k])
        plt.plot(res.oms[k])

plt.save("random.png")


