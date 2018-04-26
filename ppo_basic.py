import numpy as np
import tensorflow as tf
from retro_contest.local import make
from collections import deque
from policy import Policy
from baseline import Baseline
from baselines.a2c.utils import conv,fc,conv_to_fc,ortho_init
from sonic_util import make_env
import sys
from utils import *

render = True
if len(sys.argv) > 1:
    render = False
sess = tf.Session()
env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
optimizer = tf.train.AdamOptimizer(2e-4)
policy = Policy(sess, optimizer, env.observation_space, env.action_space)
baseline = Baseline(sess, optimizer, env.observation_space)
done = False
sess.run(tf.global_variables_initializer())
obs = env.reset()
alpha = 1e-3  # learning rate for PG
beta = 1e-3 # learning rate for baseline
numtrajs = 10  # num of trajecories to collect at each iteration 
iterations = 1000  # total num of iterations
gamma = .9999

for ite in range(iterations):    
    # trajs records for batch update
    OBS = []  # observations
    ACTS = []  # actions
    ADS = []  # advantages (to update policy)
    VAL = []  # value functions (to update baseline)
    iteration_record = []
    trajectory_record = []
    for num in range(numtrajs):
        # record for each episode
        obss = []  # observations
        acts = []   # actions
        rews = []  # instant rewards
        rsum = 0

        obs = env.reset()
        done = False
      
        numsteps = 0
        while not done:
            prob = policy.compute_prob(np.array([obs]))
            action_index = np.random.choice(8, p=prob[0])
            action = action_map(action_index)
            newobs, reward, done, _ = env.step(action)
            numsteps += 1
            
            # record
            obss.append(obs)
            acts.append(action_index)
            rews.append(max(0,reward))
            rsum += reward
            
            # update
            obs = newobs
            if render:
                env.render()
            if numsteps > 200 and np.mean(rews[-200:]) < 1:
                done = True

        # compute returns from instant rewards
        print(rsum)
        returns = discounted_rewards(rews, gamma)
    
        # record for batch update
        VAL += returns
        OBS += obss
        ACTS += acts
        trajectory_record.append(rsum)        

    baseline.train(OBS, VAL)

    iteration_record.append(np.mean(trajectory_record))
    trajectory_record = []
    
    # update baseline
    VAL = np.array(VAL)
    OBS = np.array(OBS)
    ACTS = np.array(ACTS)
    
    # update policy
    old_prob = policy.compute_prob_act(OBS, ACTS)
    BAS = baseline.compute_val(OBS)  # compute baseline for variance reduction
    ADS = VAL - BAS
    
    policy.train(OBS, ACTS, ADS, old_prob)
