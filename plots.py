import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("logs/*/monitor.csv")

all_runs = []

for f in files:

    df = pd.read_csv(f, skiprows=1)

    exp = f.split("/")[1]

    df["experiment"] = exp

    all_runs.append(df)

data = pd.concat(all_runs)

# ======================
# Reward vs Steps
# ======================

for exp in data["experiment"].unique():

    subset = data[data["experiment"] == exp]

    plt.plot(subset["l"].cumsum(), subset["r"].rolling(20).mean())

plt.xlabel("Steps")
plt.ylabel("Reward (moving avg)")
plt.title("Training Performance")

plt.savefig("training_comparison.png")
plt.show()