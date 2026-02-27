import os
import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F

from stable_baselines3.dqn.dqn import DQN as SB3_DQN
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register_envs(ale_py)

# =========================
# ARGUMENTOS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    help="Path al modelo .zip (sin extensión si quieres)")
parser.add_argument("--episodes", type=int, default=20)
args = parser.parse_args()

MODEL_PATH = args.model
EPISODES = args.episodes

# =========================
# CUSTOM DQN (IGUAL QUE TRAIN)
# =========================

class CustomDQN(SB3_DQN):

    def __init__(self, *args, loss_type="huber", double_q=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.double_q = double_q

# =========================
# CARGAR MODELO
# =========================

if not MODEL_PATH.endswith(".zip"):
    MODEL_PATH += ".zip"

model = CustomDQN.load(MODEL_PATH, device="cuda")

# =========================
# OBTENER NOMBRE EXPERIMENTO
# =========================

exp_name = os.path.basename(os.path.dirname(MODEL_PATH))
log_dir = os.path.join("logs", exp_name)

print("\nEvaluating:", exp_name)

# =========================
# 1️⃣ METRICAS DESDE MONITOR.CSV
# =========================

monitor_path = os.path.join(log_dir, "monitor.csv")

if os.path.exists(monitor_path):

    df = pd.read_csv(monitor_path, skiprows=1)

    global_mean = df["r"].mean()
    global_std = df["r"].std()

    last50 = df["r"].tail(50)
    last50_mean = last50.mean()
    last50_std = last50.std()

    best_score = df["r"].max()

    print("\n===== TRAINING METRICS =====")
    print(f"Media Global: {global_mean:.2f} ± {global_std:.2f}")
    print(f"Últimos 50: {last50_mean:.2f} ± {last50_std:.2f}")
    print(f"Mejor Score: {best_score:.2f}")

else:
    print("No monitor.csv encontrado")

# =========================
# 2️⃣ TIEMPO DE ENTRENAMIENTO
# =========================

time_path = os.path.join(log_dir, "time.txt")

if os.path.exists(time_path):
    with open(time_path, "r") as f:
        training_time = float(f.read())

    print(f"Tiempo entrenamiento (min): {training_time/60:.2f}")
else:
    print("No time.txt encontrado")

# =========================
# 3️⃣ EVALUACION REAL (DETERMINISTICA)
# =========================

ENV_NAME = "ALE/Pacman-v5"

def make_env():
    return gym.make(ENV_NAME)

env = make_env()

scores = []

for ep in range(EPISODES):

    obs, _ = env.reset()
    done = False
    total = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward

    scores.append(total)

env.close()

print("\n===== EVALUATION METRICS =====")
print(f"Episodes: {EPISODES}")
print(f"Mean: {np.mean(scores):.2f}")
print(f"Std: {np.std(scores):.2f}")
print(f"Max: {np.max(scores):.2f}")
print(f"Min: {np.min(scores):.2f}")