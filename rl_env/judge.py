import json
import importlib
import numpy as np

def load_reward():
    try:
        reward_module = importlib.import_module("reward")
        return reward_module.compute_reward
    except Exception as e:
        print("Import error:", e)
        return None

def evaluate():
    compute_reward = load_reward()
    if compute_reward is None:
        return 0.0

    with open("hidden_data.json", "r") as f:
        data = json.load(f)

    deltas = []

    for item in data:
        try:
            good_score = compute_reward(item["source"], item["good_summary"])
            bad_score = compute_reward(item["source"], item["bad_summary"])
            print("\nSOURCE:", item["source"])
            print("GOOD:", good_score)
            print("BAD :", bad_score)
        except Exception:
            return 0.0

        # Ranking check
        if good_score <= bad_score:
            print("FAILED RANKING")
            return 0.1

        deltas.append(good_score - bad_score)

    # Detect constant reward outputs
    if np.std(deltas) < 1e-6:
        return 0.0

    final_score = float(np.clip(np.mean(deltas), 0.0, 1.0))
    return final_score

if __name__ == "__main__":
    print("Final Score:", evaluate())