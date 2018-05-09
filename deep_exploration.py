import numpy as np
def perturb(returns, noise_variance):
    return [x + np.random.normal(0, noise_variance) for x in returns]
