from q_learning import train_q_learning
from env_utils import evaluate, make_env
from rbf import RBF, get_state_bounds, make_centers
import numpy as np
import json

def run_multiseed_q(env_id, seeds, rbf,
                    train_episodes=1000, eval_every=50, eval_episodes=20, max_steps=200,
                    gamma=0.99, alpha=0.005):
    all_histories = []
    for seed in seeds:
        print("\n=== SEED", seed, "===\n")
        W, hist = train_q_learning(
            env_id=env_id,
            seed=seed,
            rbf=rbf,
            train_episodes=train_episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            gamma=gamma,
            alpha=alpha
        )
        all_histories.append(hist)

    # Aggregate by checkpoint index (assumes same episodes and eval_every)
    episodes = [h["episode"] for h in all_histories[0]]

    def collect(key):
        # shape (num_seeds, num_checkpoints)
        return np.array([[h[i][key] for i in range(len(episodes))] for h in all_histories], dtype=float)

    ret = collect("return_mean")
    succ = collect("success_rate")
    steps = collect("steps_to_goal_mean")  # some can be None -> will break as float if None
    maxq = collect("max_abs_Q")

    # Handle steps_to_goal_mean (None when no success)
    steps_arr = []
    for seed_hist in all_histories:
        row = []
        for item in seed_hist:
            row.append(np.nan if item["steps_to_goal_mean"] is None else float(item["steps_to_goal_mean"]))
        steps_arr.append(row)
    steps_arr = np.array(steps_arr, dtype=float)

    summary = []
    for j, ep in enumerate(episodes):
        summary.append({
            "episode": ep,
            "return_mean_mean": float(np.mean(ret[:, j])),
            "return_mean_std": float(np.std(ret[:, j], ddof=1)),
            "success_rate_mean": float(np.mean(succ[:, j])),
            "success_rate_std": float(np.std(succ[:, j], ddof=1)),
            "steps_to_goal_mean": float(np.nanmean(steps_arr[:, j])),
            "steps_to_goal_std": float(np.nanstd(steps_arr[:, j], ddof=1)),
            "max_abs_Q_mean": float(np.mean(maxq[:, j])),
            "max_abs_Q_std": float(np.std(maxq[:, j], ddof=1)),
        })

    return all_histories, summary

def make_rbf_for_sigma(env_id, seed, n_pos=7, n_vel=7, sigma=0.15, add_bias=True):
    env = make_env(env_id, seed)
    low, high = get_state_bounds(env)
    env.close()
    centers = make_centers(n_pos, n_vel)
    return RBF(centers, low, high, sigma=sigma, add_bias=add_bias)

env_id = "MountainCar-v0"
seeds = [0,1,2,3,4]

results = []
for sigma in [0.15, 0.20]:
    rbf = make_rbf_for_sigma(env_id, seed=0, sigma=sigma)  # bounds same regardless of seed
    for alpha in [0.003, 0.005]:
        _, summary = run_multiseed_q(env_id, seeds, rbf, train_episodes=1000, eval_every=50,
                                     eval_episodes=20, gamma=0.99, alpha=alpha)
        final = summary[-1]
        results.append({"sigma": sigma, "alpha": alpha, **final})
        print("DONE sigma", sigma, "alpha", alpha,
              "final_return", round(final["return_mean_mean"],2),
              "final_success", round(final["success_rate_mean"],2),
              "final_max|Q|", round(final["max_abs_Q_mean"],2))
        
        with open("multi_seed_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Summary saved to multi_seed_summary.json")