from rbf import RBF, get_state_bounds, make_centers
from env_utils import evaluate, make_env
from Diagnostics import sample_states_from_env, max_abs_q_over_states
import gymnasium as gym
import numpy as np
import pprint
def q_values(W, phi_s):
    return W @ phi_s # shape (A,)

def epsilon_greedy_action(W, phi_s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, W.shape[0])) # with prob ε choose random action
    return int(np.argmax(q_values(W, phi_s))) # with prob 1-ε choose greedy action

def epsilon_by_episode(ep, eps_start=1.0, eps_end=0.05, decay_episodes=1000):
    # linear decay
    frac = min(1.0, ep / decay_episodes) 
    return eps_start + frac * (eps_end - eps_start) # decays from eps_start to eps_end over decay_episodes

# The semi-grardient Q-Learning update
def q_learning_update(W, phi_s, a, r, phi_s2, done, gamma, alpha):
    q_sa = W[a] @ phi_s
    if done:
        target = r
    else:
        target = r + gamma * np.max(W @ phi_s2) 
    delta = target - q_sa # TD error
    W[a] += alpha * delta * phi_s # update weights for action a in direction of phi_s scaled by TD error and learning rate alpha
    return float(delta), float(q_sa), float(target)

# one training episode (unit test)
def train_one_episode_q(env, W, rbf, rng, gamma=0.99, alpha=0.01, eps=0.1, max_steps=200):
    s, _ = env.reset()
    total_return = 0.0
    deltas = []

    for t in range (max_steps):
        phi_s = rbf(s)
        a = epsilon_greedy_action(W, phi_s, eps, rng)

        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        phi_s2 = rbf(s2)
        delta, q_sa, target = q_learning_update(W, phi_s, a, r, phi_s2, done, gamma, alpha)

        total_return += r
        deltas.append(delta)

        s = s2
        if done:
            break

    return {
        "return": total_return,
        "steps": t + 1,
        "mean_abs_td": float(np.mean(np.abs(deltas))) if deltas else 0.0
    }

# greedy policy for evaluation
def greedy_q_policy_builder(W, rbf):
    def _policy_fn(state):
        phi = rbf(state)
        return int(np.argmax(W @ phi))
    return _policy_fn

# Train loop (single seed)
def train_q_learning(env_id, seed, rbf, train_episodes=2000, eval_every=50,
                     eval_episodes=20, max_steps=200,
                     gamma=0.99, alpha=0.01):
    env = make_env(env_id, seed)
    rng = np.random.default_rng(seed)

    A = env.action_space.n
    d = rbf.d
    W = rng.normal(loc=0.0, scale=0.01, size=(A, d))
    diag_states = sample_states_from_env(env_id, seed + 99_999, n_states=2000, max_steps=max_steps) # separate RNG stream for diagnostics

    history = []
    for ep in range(train_episodes):
        eps = epsilon_by_episode(ep, eps_start = 1.0, eps_end=0.05, decay_episodes=int(train_episodes*0.7))
        
        # Render every 50th episode
        if ep % 100 == 0:
            env_render = make_env(env_id, seed, render_mode="human")
            train_metrics = train_one_episode_q(env_render, W, rbf, rng, gamma=gamma, alpha=alpha, eps=eps, max_steps=max_steps)
            env_render.close()
        else:
            train_metrics = train_one_episode_q(env, W, rbf, rng, gamma=gamma, alpha=alpha, eps=eps, max_steps=max_steps)

        if (ep + 1) % eval_every == 0:
            # evaluation is greedy (no epsilon)
            eval_result = evaluate(
                env_id = env_id,
                seed = seed + 10_000, # separate eval RNG stream
                policy_fn_builder = lambda e, rg: greedy_q_policy_builder(W, rbf),
                eval_episodes = eval_episodes,
                max_steps = max_steps                
            )
            max_abs_Q = max_abs_q_over_states(W, rbf, diag_states)
            history.append({
                "episode": ep + 1,
                "train_return": train_metrics["return"],
                "train_mean_abs_td": train_metrics["mean_abs_td"],
                "max_abs_Q": max_abs_Q,
                **eval_result
            })
            print("EP", ep+1,
                "eval_return", eval_result["return_mean"],
                "success", eval_result["success_rate"],
                "max|Q|", round(max_abs_Q, 2))
    env.close()
    return W, history

# W, history = train_q_learning(
#     env_id="MountainCar-v0",
#     seed=0,
#     rbf=RBF(make_centers(7, 7), *get_state_bounds(make_env("MountainCar-v0", seed=0)), sigma=0.15, add_bias=True),
#     train_episodes=1000,
#     eval_every=50,
#     eval_episodes=20,
#     max_steps=200,
#     gamma=0.99,
#     alpha=0.01,
# )
# pprint.pprint(history)

# env = make_env("MountainCar-v0", seed=0)
# A = env.action_space.n
# low, high = get_state_bounds(env)
# centers = make_centers(7, 7)
# rbf = RBF(centers, low, high, sigma=0.15, add_bias=True)

# d = rbf.d
# rng = np.random.default_rng(0)
# W = rng.normal(loc=0.0, scale=0.01, size=(A, d))

# s, _ = env.reset()
# phi_s = rbf(s)
# print("q values:", q_values(W, phi_s))