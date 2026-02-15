import argparse
import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


# =============================
# Argumentos de línea de comando
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddqn"])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--timesteps", type=int, default=1_000_000)

args = parser.parse_args()

ALGO = args.algo.lower()
SEED = args.seed
TIMESTEPS = args.timesteps


# =============================
# Configuración reproducible
# =============================
set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 50)
print(f"Algoritmo      : {ALGO}")
print(f"Seed           : {SEED}")
print(f"Timesteps      : {TIMESTEPS}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print("=" * 50)


# =============================
# Registrar Atari
# =============================
gym.register_envs(ale_py)


def make_env():
    env = gym.make("ALE/Pacman-v5")
    env.reset(seed=SEED)
    return env


env = DummyVecEnv([make_env])


# =============================
# Carpetas organizadas
# =============================
log_dir = f"logs/{ALGO}_seed_{SEED}"
model_dir = f"models/{ALGO}_seed_{SEED}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# =============================
# Configuración del modelo
# =============================
double_q_flag = True if ALGO == "ddqn" else False

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log=log_dir,
    seed=SEED,
    device="cuda",
    optimize_memory_usage=False,
    policy_kwargs=None,
)

# Activar Double DQN manualmente
model.policy.double_q = double_q_flag


# =============================
# Entrenamiento
# =============================
start_time = time.time()

model.learn(total_timesteps=TIMESTEPS)

end_time = time.time()
training_time = end_time - start_time

print("=" * 50)
print(f"Tiempo total de entrenamiento: {training_time/60:.2f} minutos")
print("=" * 50)


# =============================
# Guardar modelo
# =============================
model.save(f"{model_dir}/final_model")

env.close()