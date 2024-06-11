import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
#import monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


import numpy as np
import optimisation as opt
import moments_backend as est
import importlib
import noise as nois
import utils as utl
importlib.reload(opt)
importlib.reload(est)
importlib.reload(nois)
importlib.reload(utl)


noiseou = nois.OU_noise(sigma=2, gamma=1/1e4/2)
logdir ="./ppo_bayes_tensorboard/"
om0 = 10
cs = 4
penalty = [-0.001,-200]  #-1,-10,-100 are ok
for p in penalty:
    env = est.Moments_estimation(length = 20000, 
                                  om0 = om0, 
                                  noise = noiseou,
                                  max_time = 500,
                                  cs=cs,
                                  penalty = p,
                                  time_step = 50,
                                  min_time = 1)
    env = Monitor(env, logdir)
    eval_callback = EvalCallback(env, best_model_save_path="./logs2/t2"+str(p)+"/",
                                log_path="./logs2/"+str(p)+"/t2", eval_freq=1000000,
                                deterministic=False, render=False, n_eval_episodes=16) #?

    if not os.path.exists("./ppo_bayes_tensorboard/"):
        os.makedirs("./ppo_bayes_tensorboard/")

    #model = model.load("logs/best_model.zip")
    model = PPO("MlpPolicy", env, 
                n_steps = 20000,
                batch_size=100,
                verbose=1, tensorboard_log="./ppo_bayes_tensorboard/", gamma = 1)

    model.learn(total_timesteps=5000000, tb_log_name="PPO",callback=eval_callback )

