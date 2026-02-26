import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"

def parse_experiment_name(name):
    parts = name.split("_")

    data = {}

    for p in parts:
        if p.startswith("lr"):
            data["lr"] = p.replace("lr", "")
        elif p.startswith("bs"):
            data["bs"] = int(p.replace("bs", ""))
        elif p.startswith("buf"):
            data["buffer"] = int(p.replace("buf", ""))
        elif p.startswith("tf"):
            data["train_freq"] = int(p.replace("tf", ""))
        elif p in ["huber", "mse"]:
            data["loss"] = p

    return data


rows = []

for folder in os.listdir(LOG_DIR):

    path = os.path.join(LOG_DIR, folder)

    monitor = os.path.join(path, "monitor.csv")

    if not os.path.exists(monitor):
        continue

    df = pd.read_csv(monitor, skiprows=1)

    rewards = df["r"]

    global_mean = rewards.mean()
    last50 = rewards.tail(50).mean()

    info = parse_experiment_name(folder)

    info["experiment"] = folder
    info["global_mean"] = global_mean
    info["last50"] = last50

    rows.append(info)

results = pd.DataFrame(rows)

os.makedirs("plots", exist_ok=True)

print(results)

def plot_param(param):

    grouped = results.groupby(param)["last50"].mean().reset_index()

    plt.figure()

    plt.bar(grouped[param].astype(str), grouped["last50"])

    plt.title(f"Impact of {param}")
    plt.xlabel(param)
    plt.ylabel("Reward (last 50 episodes)")

    plt.tight_layout()

    plt.savefig(f"plots/{param}.png")

    plt.show()


plot_param("lr")
plot_param("bs")
plot_param("buffer")
plot_param("train_freq")
plot_param("loss")