import numpy as np
from env_utils import evaluate, make_env

def sample_states_from_env(env_id, seed, n_states=2000, max_steps=200):
    """
    Collect states by rolling out a random policy. This samples the state distribution you actually see.
    """
    env = make_env(env_id, seed)
    rng = np.random.default_rng(seed)

    states = []
    while len(states) < n_states:
        s, _ = env.reset()
        for _ in range(max_steps):
            states.append(np.array(s, dtype=float))
            a = int(rng.integers(0, env.action_space.n))
            s, r, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
    
    env.close()
    return np.stack(states[: n_states], axis=0) # shape (n_states, 2)

def max_abs_q_over_states(W, rbf, states):
    """
    states: (N, 2)
    """
    max_abs = 0.0

    for s in states:
        q = W @ rbf(s) # shape (A,)
        m = float(np.max(np.abs(q)))
        if m > max_abs:
            max_abs = m
    return max_abs