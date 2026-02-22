from rbf import RBF, get_state_bounds, make_centers, make_env
import gymnasium as gym
import numpy as np
def q_values(W, phi_s):
    return W @ phi_s # shape (A,)

env = make_env("MountainCar-v0", seed=0)
A = env.action_space.n
low, high = get_state_bounds(env)
centers = make_centers(7, 7)
rbf = RBF(centers, low, high, sigma=0.15, add_bias=True)

d = rbf.d
rng = np.random.default_rng(0)
W = rng.normal(loc=0.0, scale=0.01, size=(A, d))

s, _ = env.reset()
phi_s = rbf(s)
print("q values:", q_values(W, phi_s))