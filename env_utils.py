import numpy as np
import gymnasium as gym

def make_env (env_id="MountainCar-v0", seed=0):
    env = gym.make(env_id, render_mode="human")
    # seed env+action space for reproducibility
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

env = make_env(seed=123)
s, info = env.reset()

print(s) # should be a 2D state: [position, velocity]
print(env.observation_space.low, env.observation_space.high)
print(env.action_space.n)