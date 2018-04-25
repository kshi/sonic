import numpy as np
def action_map(a):
    LEFT   =  (0,0,0,0,0,0,1,0,0,0,0,0)
    RIGHT  =  (0,0,0,0,0,0,0,1,0,0,0,0)
    DOWN   =  (0,0,0,0,0,1,0,0,0,0,0,0)
    JUMP   =  (1,0,0,0,0,0,0,0,0,0,0,0)
    R_DOWN =  (0,0,0,0,0,1,0,1,0,0,0,0)
    L_DOWN =  (0,0,0,0,0,1,1,0,0,0,0,0)
    NONE   =  (0,0,0,0,0,0,0,0,0,0,0,0)
    J_DOWN =  (1,0,0,0,0,1,0,0,0,0,0,0)
    mapping = {0:LEFT, 1:RIGHT, 2:DOWN, 3:R_DOWN, 4:L_DOWN, 5:J_DOWN, 6:JUMP, 7:NONE}
    return mapping[a]
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)
