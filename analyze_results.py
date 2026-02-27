import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

results = []

# =====================================================
# LEER TODOS LOS EXPERIMENTOS
# =====================================================

for exp in os.listdir(LOG_DIR):

    monitor_path = os.path.join(LOG_DIR, exp, "monitor.csv")
    time_path = os.path.join(LOG_DIR, exp, "time.txt")

    if not os.path.exists(monitor_path):
        continue

    df = pd.read_csv(monitor_path, skiprows=1)

    total_steps = df["l"].sum()
    global_mean = df["r"].mean()
    global_std = df["r"].std()

    last50 = df["r"].tail(50)
    last50_mean = last50.mean()
    last50_std = last50.std()

    training_time_min = None
    if os.path.exists(time_path):
        with open(time_path, "r") as f:
            training_time_sec = float(f.read())
            training_time_min = training_time_sec / 60.0

    best_score = df["r"].max()

    results.append({
        "experiment": exp,
        "total_steps": total_steps,
        "global_mean": global_mean,
        "global_std": global_std,
        "last50_mean": last50_mean,
        "last50_std": last50_std,
        "best_score": best_score,
        "training_time_min": training_time_min
    })

summary = pd.DataFrame(results)

print("\nResumen general:")
print(
    summary
    .sort_values("last50_mean", ascending=False)
    .reset_index(drop=True)
)

# =====================================================
# FUNCION PARA GRAFICAR
# =====================================================

def plot_group(run_list, title, filename, label_parser=None):

    if len(run_list) == 0:
        print(f"No runs found for {title}")
        return

    plt.figure(figsize=(10,6))

    for run in run_list:

        monitor_file = os.path.join(LOG_DIR, run, "monitor.csv")
        df = pd.read_csv(monitor_file, skiprows=1)

        rewards = df["r"].rolling(20).mean()
        steps = df["l"].cumsum()

        label = label_parser(run) if label_parser else run

        plt.plot(steps, rewards, label=label)

    plt.xlabel("Steps")
    plt.ylabel("Reward (Moving Avg 20)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()

    print(f"Saved {filename}")

# =====================================================
# SEPARAR POR TIMESTEPS
# =====================================================

runs_200k = summary[
    (summary["total_steps"] >= 180000) &
    (summary["total_steps"] <= 220000)
]["experiment"].tolist()

runs_500k = summary[
    (summary["total_steps"] >= 450000) &
    (summary["total_steps"] <= 550000)
]["experiment"].tolist()

# Grandes normales (NO finales)
runs_big = summary[
    (summary["total_steps"] >= 900000) &
    (~summary["experiment"].str.startswith("final"))
]["experiment"].tolist()

# Solo finales
runs_final = summary[
    summary["experiment"].str.startswith("final")
]["experiment"].tolist()

# =====================================================
# ===================== 200K ==========================
# =====================================================

batch_200k = [
    r for r in runs_200k
    if "lr0.0001" in r
    and "buf50000" in r
    and "tf4" in r
    and "huber" in r
    and ("bs32" in r or "bs64" in r or "bs128" in r)
]

plot_group(
    batch_200k,
    "Batch Size - 200k",
    "batch_200k.png",
    label_parser=lambda r: r.split("_")[2].replace("bs","")
)

# =====================================================
# ===================== 500K ==========================
# =====================================================

lr_500k = [
    r for r in runs_500k
    if "bs128" in r
    and "buf50000" in r
    and "tf1" in r
    and "huber" in r
    and ("lr0.0001" in r or "lr0.0003" in r or "lr0.0005" in r)
]

plot_group(
    lr_500k,
    "Learning Rate - 500k",
    "lr_500k.png",
    label_parser=lambda r: r.split("_")[1].replace("lr","")
)

# =====================================================
# ============ 1M vs 2M vs 3M (NORMALES) ==============
# =====================================================

plot_group(
    runs_big,
    "Training Steps Comparison (1M vs 2M vs 3M)",
    "steps_comparison.png",
    label_parser=lambda r: r.split("_")[1]
)

# =====================================================
# ================= MODELOS FINALES ===================
# =====================================================

plot_group(
    runs_final,
    "Final Models Comparison",
    "final_models.png",
    label_parser=lambda r: r.split("_")[1] + "_seed" + r.split("_")[-1].replace("seed","")
)

print("\n✅ ALL EXPERIMENT PLOTS GENERATED CORRECTLY.")