from rbf import RBF, get_state_bounds, make_centers
from env_utils import evaluate, make_env
import gymnasium as gym

import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits) # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Policy gradient action probabilities
def policy_prob(Theta, phi_s):
    logits = Theta @ phi_s # shape (A,)
    return softmax(logits)

# sample action from policy
def sample_action(Theta, phi_s, rng):
    probs = policy_prob(Theta, phi_s)
    action = int(rng.choice(len(probs), p=probs))
    return action, probs

# defining the critic
def value_estimate(v, phi_s):
    return float(v @ phi_s) # shape (1,)

# gradient of log policy
def grad_log_policy(phi_s, action, probs):
    """
    Returns gradient with shape (A, d)
    """
    grad = -np.outer(probs, phi_s) # for all actions
    grad[action] += phi_s # chosen action gets +phi_s
    return grad

# Advantage/TD error
def td_advantage(v, phi_s, r, phi_s2, done, gamma):
    V_s = value_estimate(v, phi_s)
    V_s2 = 0.0 if done else value_estimate(v, phi_s2)
    A_t = r + gamma * V_s2 - V_s
    return float(A_t), float(V_s), float(V_s2)

# One-step actor-critic update
def actor_critic_update(Theta, v, phi_s, action, probs, r, phi_s2, done,
                        gamma=0.99, alpha_actor=0.01, alpha_critic=0.01):
    A_t, V_s, V_s2 = td_advantage(v, phi_s, r, phi_s2, done, gamma)
    # critic
    v += alpha_critic * A_t * phi_s # value update
    # actor
    grad = grad_log_policy(phi_s, action, probs)
    Theta += alpha_actor * A_t * grad # policy update
    return float(A_t), float(V_s), float(V_s2)

# one training episode
def train_one_episode_ac(env, Theta, v, rbf,rng,
                        gamma=0.99,alpha_theta=0.01, alpha_v=0.01, 
                        max_steps=1000):
    s, _ = env.reset()
    total_return = 0.0
    advantages = []

    for t in range(max_steps):
        phi_s = rbf(s)
        action, probs = sample_action(Theta, phi_s, rng)
        s2, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        phi_s2 = rbf(s2)

        A_t, V_s, V_s2 = actor_critic_update(Theta, v, phi_s, action, probs, r, phi_s2, done,
                                            gamma, alpha_theta, alpha_v)
        
        total_return += r
        advantages.append(abs(A_t))

        s = s2
        if done:
            break
    return {
        "return": total_return,
        "steps": t + 1,
        "mean_abs_advantage": float(np.mean(advantages)) if advantages else 0.0
    }

# greedy policy for evaluation
def greedy_ac_policy_builder(Theta, rbf):
    def _policy_fn(state):
        phi = rbf(state)
        probs = policy_prob(Theta, phi)
        return int(np.argmax(probs))
    return _policy_fn

# Full training loop
def train_actor_critic(env_id, seed, rbf,
                       train_episodes=1000, eval_every=50, eval_episodes=20,
                       max_steps=1000, gamma=0.99, alpha_theta=0.01, alpha_v=0.01):
    env = make_env(env_id, seed)
    rng = np.random.default_rng(seed)

    A = env.action_space.n
    d = rbf.d

    Theta = rng.normal(loc=0.0, scale=0.01, size=(A, d)) # policy parameters
    v = rng.normal(loc=0.0, scale=0.01, size=d) # value function parameters

    history = []

    for ep in range(train_episodes):
        train_metrics = train_one_episode_ac(
            env, Theta, v, rbf, rng,
            gamma=gamma,
            alpha_theta=alpha_theta,
            alpha_v=alpha_v,
            max_steps=max_steps
        )

        if (ep + 1) % eval_every == 0:
            eval_result = evaluate(
                env_id=env_id,
                seed=seed + 10_000,
                policy_fn_builder=lambda e, rg: greedy_ac_policy_builder(Theta, rbf),
                eval_episodes=eval_episodes,
                max_steps=max_steps
            )

            history.append({
                "episode": ep + 1,
                "train_return": train_metrics["return"],
                "train_mean_abs_advantage": train_metrics["mean_abs_advantage"],
                **eval_result
            })

            print("EP", ep + 1,
                  "eval_return", eval_result["return_mean"],
                  "success", eval_result["success_rate"])
    env.close()
    return Theta, v, history

    
# # first sanity check
# env = make_env("MountainCar-v0")
# s, _ = env.reset()
# A = env.action_space.n
# low, high = get_state_bounds(env)
# centers = make_centers(7, 7)
# rbf = RBF(centers, low, high, sigma=0.15, add_bias=True)
# d = rbf.d
# rng = np.random.default_rng(0)

# Theta = rng.normal(loc=0.0, scale=0.01, size=(A, d))
# phi_s = rbf(s)
# probs = policy_prob(Theta, phi_s)

# # print("Action probabilities:", probs)
# # print("Sum of probabilities:", np.sum(probs))

# # grad sanity check
# grad = grad_log_policy(phi_s, action=1, probs=probs)
# print("grad shape:", grad.shape)
# print("row sums maybe not zero, but ckeck finite: ", np.all(np.isfinite(grad)))
Theta, v, history = train_actor_critic(
    env_id="MountainCar-v0",
    seed=0,
    rbf=RBF(make_centers(7, 7), *get_state_bounds(make_env("MountainCar-v0", seed=0)), sigma=0.15, add_bias=True),
    train_episodes=1000,
    eval_every=50,
    eval_episodes=20,
    max_steps=200,
    gamma=0.99,
    alpha_theta=0.0005, # changing these learning rates can affect stability and convergence
    alpha_v=0.005, # If the policy gets good by episode 200 and then degrades, that usually means updates are too aggressive.
)
import pprint
pprint.pprint(history)