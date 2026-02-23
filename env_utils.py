import numpy as np
import gymnasium as gym

def make_env (env_id="MountainCar-v0", seed=0, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode).unwrapped 
    # seed env+action space for reproducibility
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

def is_success (env, state):
    goal = env.unwrapped.goal_position if hasattr (env.unwrapped, "goal_position") else 0.5
    position = state[0]
    return position >= goal

def run_one_episode (env, policy_fn, max_steps=600, render=False):
    out = env.reset()

    if isinstance(out, tuple):
        state = out[0]
    else:
        state = out

    total_return = 0.0
    steps = 0
    success = False

    for t in range (max_steps):
        if render:
            env.render()
        action = policy_fn(state)
        step_out = env.step(action)

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out
            terminated, truncated = done, False

        total_return += reward
        steps += 1

        if is_success(env, next_state):
            success = True
        state = next_state
        if done:
            break

    return {
        "return": total_return,
        "steps": steps,
        "success": success
    }

def evaluate(env_id, seed, policy_fn_builder, eval_episodes=20, max_steps=200):
    """
    policy_fn_builder(env, rng) -> policty_fn(state) -> action
    this keeps RNG controlled and comparable

    """
    env = make_env(env_id, seed)
    rng = np.random.default_rng(seed)

    returns = []
    successes = []
    steps_to_goal = []

    policy_fn = policy_fn_builder(env, rng)

    for ep in range (eval_episodes):
        m = run_one_episode(env, policy_fn, max_steps=max_steps)
        returns.append(m["return"])
        successes.append(m["success"])
        if m["success"]:
            steps_to_goal.append(m["steps"])

    returns = np.array(returns, dtype=float)
    successes = np.array(successes, dtype=bool)

    results = {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "success_rate": float(successes.mean()),
        "steps_to_goal_mean": float(np.mean(steps_to_goal)) if len(steps_to_goal) > 0 else None,
        "steps_to_goal_std": float(np.std(steps_to_goal, ddof=1)) if len(steps_to_goal) > 1 else None,
    }

    env.close()
    return results

def random_policy (env, rng):
    def _pi(state):
        return int(rng.integers(0, env.action_space.n))
    return _pi

# sanity check
# baseline = evaluate (
#     env_id="MountainCar-v0",
#     seed=0,
#     policy_fn_builder=lambda env, rng: random_policy(env, rng),
#     eval_episodes=20,
#     max_steps=200
# )
# print(baseline)
# env = make_env(seed=0)
# rng = np.random.default_rng(0)
# metrics = run_one_episode(env, random_policy(env, rng))
# print(metrics)

# print(s) # should be a 2D state: [position, velocity]
# print(env.observation_space.low, env.observation_space.high)
# print(env.action_space.n)