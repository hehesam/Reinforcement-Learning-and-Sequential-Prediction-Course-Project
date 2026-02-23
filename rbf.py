import numpy as np
import gymnasium as gym

# even though MountainCar has known ranges, it's cleaner to read them
def get_state_bounds(env):
    low = env.observation_space.low
    high = env.observation_space.high
    return low.astype(float), high.astype(float)

def normalize_state(s, low, high):
    s = np.asarray(s, dtype=float)
    return (s - low) / (high - low + 1e-8)  # avoid division by zero

def make_centers(n_pos=7, n_vel=7):
    xs = np.linspace(0.0, 1.0, n_pos)
    ys = np.linspace(0.0, 1.0, n_vel)
    centers = np.array([(x, y) for x in xs for y in ys], dtype=float)
    return centers # (k, 2)

class RBF:
    def __init__(self, centers, low, high, sigma=0.15, add_bias = True):
        self.centers = np.asarray(centers, dtype=float)
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.sigma = float(sigma)
        self.add_bias = bool(add_bias)

        self.K = self.centers.shape[0]
        self.d = self.K + (1 if self.add_bias else 0)

    def __call__(self, s):
        s_norm = normalize_state(s, self.low, self.high)
        diff = self.centers - s_norm
        dist2 = np.sum(diff * diff, axis=1)
        feats = np.exp(-dist2 / (2.0 * self.sigma * self.sigma))
        if self.add_bias:
            feats = np.concatenate([feats, [1.0]])
        return feats
    

# env = make_env("MountainCar-v0", seed=0)
# low, high = get_state_bounds(env)
# centers = make_centers(7, 7)

# rbf = RBF(centers, low, high, sigma=0.15, add_bias=True)
# s, _ = env.reset()
# phi = rbf(s)
# # Sanity check 1: Feature dimension + value ranges
# print("d =", rbf.d)
# print("phi shape:", phi.shape)
# print("min/max:", phi.min(), phi.max())
# print("bias term:", phi[-1])

# # Sanity check 2: Smoothness
# s1 = np.array([-0.5, 0.02])
# s2 = s1 + np.array([0.01, 0.0])

# diff_norm = np.linalg.norm(rbf(s1) - rbf(s2))
# print("feature difference:", diff_norm)

# # Nearest center fires most
# phi_no_bias = rbf(s)[:-1]
# top = np.argmax(phi_no_bias)
# print("max activation:", phi_no_bias[top])
# print("top index:", top)
