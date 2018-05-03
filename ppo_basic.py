import numpy as np
import tensorflow as tf
from retro_contest.local import make
from policy import Policy
from baseline import Baseline
from sonic_util import make_env, wrap_env
import sys
from utils import *

render = False
if len(sys.argv) > 1:
    render = True
sess = tf.Session()
env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
#env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
env = wrap_env(env)
optimizer = tf.train.AdamOptimizer(2e-4)
policy = Policy(sess, optimizer, env.observation_space, env.action_space)
baseline = Baseline(sess, optimizer, env.observation_space)
done = False
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
obs = env.reset()
alpha = 1e-3  # learning rate for PG
beta = 1e-3 # learning rate for baseline
iterations = 1000  # total num of iterations
gamma = .99
step_limit = 4500 # 5 minutes

for ite in range(iterations):    
    # trajs records for batch update
    OBS = []  # observations
    ACTS = []  # actions
    ADS = []  # advantages (to update policy)
    VAL = []  # value functions (to update baseline)
    iteration_record = []
    trajectory_record = []
    steps_taken = 0
    
    while steps_taken < step_limit:
        # record for each episode
        obss = []  # observations
        acts = []   # actions
        rews = []  # instant rewards
        rsum = 0

        obs = env.reset()
        done = False
      
        while not done:
            prob = policy.compute_prob(np.array([obs]))
            action = np.random.choice(len(prob[0]), p=prob[0])
            newobs, reward, done, _ = env.step(action)
            steps_taken += 1
            
            # record
            obss.append(obs)
            acts.append(action)
            rews.append(reward)
            rsum += reward
            # update
            obs = newobs
            if render:
                env.render()
            if len(rews) > 600 and np.mean(rews[-600:]) < 0.01:
                done = True

        # compute returns from instant rewards
        returns = discounted_rewards(rews, gamma)
    
        # record for batch update
        VAL += returns
        OBS += obss
        ACTS += acts
        trajectory_record.append(rsum)

    print(np.mean(trajectory_record))
    iteration_record.append(np.mean(trajectory_record))
    trajectory_record = []
    baseline.train(OBS, VAL)
    
    # update baseline
    VAL = np.array(VAL)
    OBS = np.array(OBS)
    ACTS = np.array(ACTS)
    
    # update policy
    old_prob = policy.compute_prob_act(OBS, ACTS)
    BAS = baseline.compute_val(OBS)  # compute baseline for variance reduction
    ADS = VAL - BAS
    
    policy.train(OBS, ACTS, ADS, old_prob)

    saver.save(sess, "./labyrinthzone_tf_ppo")
